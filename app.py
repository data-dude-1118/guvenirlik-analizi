import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from io import BytesIO
import requests

from scipy.stats import pearsonr
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import streamlit as st

# ══════════════════════════════════════════════════════════════════════════
#  TEMA / SABİTLER  (orijinal PyQt panosundan aynen alındı)
# ══════════════════════════════════════════════════════════════════════════
DARK_BG    = "#12121f"
CARD_BG    = "#16213e"
BORDER     = "#0f3460"
ACCENT1    = "#e94560"
ACCENT2    = "#a8d8ea"
ACCENT3    = "#f7b731"
ACCENT4    = "#00b894"
TEXT       = "#eaeaea"
TEXT_DIM   = "#8899aa"
CC = ["#e94560", "#a8d8ea", "#f7b731", "#6c5ce7", "#00b894", "#fd79a8", "#55efc4"]

N_CLUSTERS = 5
DEFAULT_CLUSTERS = {
    0: ["aylik_tasarruf", "sanat_katilmama_neden", "guven_ic_skor", "guven_ic_neden",
        "guven_dis_skor", "guven_dis_neden", "gunluk_ogun_sayisi", "beslenme_kalitesi",
        "uyku_suresi", "uyku_kalitesi", "yasama_kalitesi_genel", "yasam_keyfi",
        "yasam_anlami", "psikolojik_negatif_siklik"],
    1: ["harcama_egitim", "harcama_kisisel_bakim", "harcama_saglik",
        "harcama_spor", "harcama_fatura", "harcama_ulasim"],
    2: ["aylik_gelir", "harcama_beslenme", "harcama_sosyallesme",
        "dijital_platform_sayisi", "disaridan_yemek_sayisi"],
    3: ["guven_ic_neden", "guven_dis_neden", "sanat_katilmama_neden", "kampus_zorluklari"],
}

THR = {
    "alpha": [(0.90, "Mükemmel", "#00b894"), (0.80, "İyi", "#6c5ce7"),
              (0.70, "Kabul Ed.", ACCENT3), (0.60, "Zayıf", "#e17055"), (0.0, "Yetersiz", ACCENT1)],
    "omega": [(0.80, "İyi", "#00b894"), (0.70, "Kabul Ed.", ACCENT3),
              (0.60, "Zayıf", "#e17055"), (0.0, "Yetersiz", ACCENT1)],
    "itc":   [(0.50, "Güçlü", "#00b894"), (0.30, "Yeterli", ACCENT3), (0.0, "Düşük ⚠", ACCENT1)],
}


def badge(v, metric):
    if v is None or np.isnan(v):
        return "N/A", TEXT_DIM
    for thr, lbl, clr in THR[metric]:
        if v >= thr:
            return lbl, clr
    return "N/A", TEXT_DIM


# ══════════════════════════════════════════════════════════════════════════
#  GÜVENİRLİK HESAPLAMALARI (orijinal koddan birebir)
# ══════════════════════════════════════════════════════════════════════════
def cronbach_alpha(X):
    k = X.shape[1]
    if k < 2:
        return float("nan"), np.full(k, float("nan"))
    iv = X.var(axis=0, ddof=1)
    tv = X.sum(axis=1).var(ddof=1)
    if tv == 0:
        return float("nan"), np.full(k, float("nan"))
    alpha = (k / (k - 1)) * (1 - iv.sum() / tv)
    aid = np.full(k, float("nan"))
    for i in range(k):
        Xd = np.delete(X, i, axis=1)
        kd = k - 1
        if kd < 2:
            continue
        tv2 = Xd.sum(axis=1).var(ddof=1)
        if tv2 > 0:
            aid[i] = (kd / (kd - 1)) * (1 - Xd.var(axis=0, ddof=1).sum() / tv2)
    return alpha, aid


def item_total_corr(X):
    k = X.shape[1]
    itc = np.full(k, float("nan"))
    for i in range(k):
        try:
            r, _ = pearsonr(X[:, i], np.delete(X, i, axis=1).sum(axis=1))
            itc[i] = r
        except Exception:
            pass
    return itc


def split_half(X, method="odd_even"):
    k = X.shape[1]
    if k < 2:
        return float("nan"), float("nan")
    if method == "odd_even":
        h1, h2 = X[:, 0::2], X[:, 1::2]
    elif method == "first_sec":
        m = k // 2
        h1, h2 = X[:, :m], X[:, m:]
    else:
        o = np.argsort(X.var(axis=0))[::-1]
        ml = min(len(o[0::2]), len(o[1::2]))
        h1, h2 = X[:, o[0::2][:ml]], X[:, o[1::2][:ml]]
    try:
        r, _ = pearsonr(h1.sum(1), h2.sum(1))
        return r, (2 * r / (1 + r) if r > -1 else float("nan"))
    except Exception:
        return float("nan"), float("nan")


def mcdonald_omega(X):
    k = X.shape[1]
    if k < 2:
        return float("nan"), np.full(k, float("nan"))
    try:
        Xs = StandardScaler().fit_transform(X)
        fa = FactorAnalysis(n_components=1, max_iter=2000, tol=1e-6)
        fa.fit(Xs)
        lam = fa.components_[0]
        sl = np.sum(lam)
        su = np.sum(1 - lam ** 2)
        return sl ** 2 / (sl ** 2 + su), lam
    except Exception:
        return float("nan"), np.full(k, float("nan"))


def omega_if_deleted(X):
    k = X.shape[1]
    oid = np.full(k, float("nan"))
    for i in range(k):
        Xd = np.delete(X, i, axis=1)
        if Xd.shape[1] < 2:
            continue
        om, _ = mcdonald_omega(Xd)
        oid[i] = om
    return oid


def run_reliability(df, cluster_map, split_method="odd_even"):
    res = {}
    for cid, cols in cluster_map.items():
        if len(cols) < 2:
            res[cid] = {"vars": cols, "n_items": len(cols), "skipped": True}
            continue
        X = df[cols].values.astype(float)
        al, aid = cronbach_alpha(X)
        itc = item_total_corr(X)
        rh, sb = split_half(X, split_method)
        om, lam = mcdonald_omega(X)
        oid = omega_if_deleted(X)
        res[cid] = {
            "vars": cols, "n_items": len(cols), "skipped": False,
            "alpha": al, "alpha_if_del": aid, "itc": itc,
            "r_half": rh, "sb": sb, "omega": om,
            "fa_loadings": lam, "omega_if_del": oid,
        }
    return res


def apply_isolation_forest(df, cols, n_estimators, contamination, random_state):
    cols = [c for c in cols if c in df.columns]
    X = df[cols].values.astype(float)
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                           random_state=random_state, n_jobs=-1)
    pred = clf.fit_predict(X)
    scores = clf.decision_function(X)
    mask = pred == 1
    n_b, n_a = len(df), int(mask.sum())
    stats = {
        "n_before": n_b, "n_after": n_a, "n_removed": n_b - n_a,
        "pct_removed": (n_b - n_a) / n_b * 100 if n_b else 0,
        "contamination": contamination, "n_estimators": n_estimators,
        "random_state": random_state,
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
    }
    return df[mask].reset_index(drop=True), stats


def load_dataframe(file_or_url, filename):
    fname = filename.lower()
    if fname.endswith(".csv"):
        df_raw = pd.read_csv(file_or_url, header=None)
    elif fname.endswith((".xlsx", ".xlsm")):
        df_raw = pd.read_excel(file_or_url, header=None, engine="openpyxl")
    elif fname.endswith(".xls"):
        df_raw = pd.read_excel(file_or_url, header=None, engine="xlrd")
    else:
        try:
            df_raw = pd.read_excel(file_or_url, header=None, engine="openpyxl")
        except Exception:
            if hasattr(file_or_url, "seek"):
                file_or_url.seek(0)
            df_raw = pd.read_excel(file_or_url, header=None, engine="xlrd")
    vnames = df_raw.iloc[1, :].tolist()
    data = df_raw.iloc[2:].reset_index(drop=True)
    data.columns = vnames
    data = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").dropna()
    return data


# ══════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB YARDIMCI
# ══════════════════════════════════════════════════════════════════════════
def style_ax(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(BORDER)


def plot_summary(res, alpha_thr, itc_thr):
    valid = {c: r for c, r in res.items() if not r.get("skipped")}
    if not valid:
        return None
    cids = sorted(valid.keys())
    x = np.arange(len(cids))
    w = 0.35
    fig = plt.figure(figsize=(13, 5), dpi=100)
    gs = GridSpec(1, 3, figure=fig, wspace=0.42)

    ax1 = fig.add_subplot(gs[0])
    alp = [valid[c]["alpha"] for c in cids]
    omg = [valid[c]["omega"] for c in cids]
    ax1.bar(x - w / 2, alp, w, color=ACCENT2, label="α", edgecolor=BORDER)
    ax1.bar(x + w / 2, omg, w, color=CC[2], label="ω", edgecolor=BORDER)
    ax1.axhline(alpha_thr, color=ACCENT1, linestyle="--", linewidth=1.2)
    ax1.set_xticks(x); ax1.set_xticklabels([f"K{c+1}" for c in cids], color=TEXT)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Cronbach α & McDonald ω", color="white", fontsize=10)
    ax1.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8)
    for i, (a, o) in enumerate(zip(alp, omg)):
        if not np.isnan(a): ax1.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", color="white", fontsize=8)
        if not np.isnan(o): ax1.text(i + w / 2, o + 0.02, f"{o:.2f}", ha="center", color="white", fontsize=8)
    style_ax(fig, ax1)

    ax2 = fig.add_subplot(gs[1])
    rhs = [valid[c]["r_half"] for c in cids]
    sbs = [valid[c]["sb"] for c in cids]
    ax2.bar(x - w / 2, rhs, w, color=CC[3], label="r", edgecolor=BORDER)
    ax2.bar(x + w / 2, sbs, w, color=CC[4], label="SB", edgecolor=BORDER)
    ax2.axhline(0.7, color=ACCENT1, linestyle="--", linewidth=1.2)
    ax2.set_xticks(x); ax2.set_xticklabels([f"K{c+1}" for c in cids], color=TEXT)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Split-Half & Spearman-Brown", color="white", fontsize=10)
    ax2.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8)
    style_ax(fig, ax2)

    ax3 = fig.add_subplot(gs[2])
    bp = ax3.boxplot([valid[c]["itc"] for c in cids], patch_artist=True,
                      medianprops=dict(color="white", linewidth=2))
    for patch, clr in zip(bp["boxes"], [CC[i % len(CC)] for i in range(len(cids))]):
        patch.set_facecolor(clr); patch.set_alpha(0.75)
    ax3.axhline(itc_thr, color=ACCENT1, linestyle="--", linewidth=1.2)
    ax3.set_xticks(range(1, len(cids) + 1))
    ax3.set_xticklabels([f"K{c+1}" for c in cids], color=TEXT)
    ax3.set_title("ITC Dağılımı", color="white", fontsize=10)
    style_ax(fig, ax3)

    fig.tight_layout(pad=1.8)
    return fig


def plot_if_deleted(res):
    valid = {c: r for c, r in res.items() if not r.get("skipped")}
    if not valid:
        return None
    cids = sorted(valid.keys())
    n = len(cids)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(13, 4.5 * nrows), dpi=100)
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.75, wspace=0.38)

    for idx, cid in enumerate(cids):
        r = valid[cid]
        row, col = idx // ncols, idx % ncols
        ax = fig.add_subplot(gs[row, col])
        vars_ = r["vars"]; k = len(vars_)
        aid, oid = r["alpha_if_del"], r["omega_if_del"]
        al_cur, om_cur = r["alpha"], r["omega"]
        x = np.arange(k); w = 0.38
        bars_a = ax.bar(x - w / 2, aid, w, color=ACCENT2, alpha=0.85, edgecolor=BORDER, label="α-if-del")
        bars_o = ax.bar(x + w / 2, oid, w, color=CC[2], alpha=0.85, edgecolor=BORDER, label="ω-if-del")
        ax.axhline(al_cur, color=ACCENT1, linestyle="--", linewidth=1.4, label=f"α={al_cur:.3f}")
        ax.axhline(om_cur, color=ACCENT3, linestyle=":", linewidth=1.4, label=f"ω={om_cur:.3f}")
        for i in range(k):
            if not np.isnan(aid[i]) and aid[i] > al_cur + 0.005:
                bars_a[i].set_edgecolor(ACCENT1); bars_a[i].set_linewidth(2.0)
            if not np.isnan(oid[i]) and oid[i] > om_cur + 0.005:
                bars_o[i].set_edgecolor(ACCENT1); bars_o[i].set_linewidth(2.0)
        ax.set_xticks(x)
        short_names = [v if len(v) <= 16 else v[:14] + ".." for v in vars_]
        ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=7, color=TEXT)
        ax.set_title(f"K{cid+1} — α/ω İf Deleted ({k} madde)", color=CC[cid % len(CC)],
                     fontsize=9, fontweight="bold", pad=6)
        all_vals = np.concatenate([aid, oid, [al_cur, om_cur]])
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals):
            ax.set_ylim(max(0, all_vals.min() - 0.06), min(1.0, all_vals.max() + 0.08))
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=7, loc="lower right", framealpha=0.6)
        style_ax(fig, ax)

    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        ax_empty = fig.add_subplot(gs[row, col]); ax_empty.set_visible(False)
    fig.tight_layout(pad=1.5)
    return fig


def plot_if_report(ifs):
    fig = plt.figure(figsize=(12, 4), dpi=100)
    gs = GridSpec(1, 2, figure=fig, wspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(["Önce", "Sonra"], [ifs["n_before"], ifs["n_after"]],
                    color=[ACCENT2, ACCENT4], edgecolor=BORDER, width=0.5)
    for bar, val in zip(bars, [ifs["n_before"], ifs["n_after"]]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  str(val), ha="center", va="bottom", color="white", fontweight="bold", fontsize=12)
    ax1.set_title("Gözlem Sayısı", color="white", fontsize=11)
    ax1.set_ylabel("N", color=ACCENT2)
    style_ax(fig, ax1)
    ax2 = fig.add_subplot(gs[1])
    wedges, texts, autotexts = ax2.pie(
        [ifs["n_after"], ifs["n_removed"]], labels=["Normal", "Aykırı"], autopct="%1.1f%%",
        colors=[ACCENT4, ACCENT1], startangle=90,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2, "width": 0.55})
    for tt in texts: tt.set_color(TEXT); tt.set_fontsize(9)
    for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
    ax2.set_title("Aykırı Oran", color="white", fontsize=11)
    fig.tight_layout(pad=1.5)
    return fig


def build_text_report(df, res, alpha_thr, itc_thr, split_method_label, rev_info, ifs=None):
    L = ["═" * 70, "  GÜVENİRLİK ANALİZİ RAPORU", "═" * 70,
         f"  Tarih          : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
         f"  Gözlem Sayısı  : {len(df)} (analiz edilen)",
         f"  Alpha Eşiği    : {alpha_thr:.2f}",
         f"  ITC Eşiği      : {itc_thr:.2f}",
         f"  Split Yöntemi  : {split_method_label}"]
    if ifs:
        L += [f"  🌲 IF: {ifs['n_before']}→{ifs['n_after']} "
              f"({ifs['n_removed']} aykırı elendi, cont={ifs['contamination']:.2f})"]
    if rev_info:
        L += ["", "─" * 70, "  TERS ÇEVRİLEN DEĞİŞKENLER", "─" * 70]
        for cid, v, vmin, vmax in rev_info:
            L.append(f"  K{cid+1}  {v:<38} aralık {int(vmin)}–{int(vmax)}  formül: {int(vmin+vmax)} − x")
    L += ["", "─" * 70,
          "  REFERANS:  α/ω ≥0.90 Mükemmel | ≥0.80 İyi | ≥0.70 Kabul | <0.70 Yetersiz",
          "             ITC ≥0.50 Güçlü | ≥0.30 Yeterli | <0.30 Düşük → madde çıkar",
          "─" * 70, ""]
    for cid, r in sorted(res.items()):
        L += [f"═" * 70, f"  KÜME {cid+1}  ({r['n_items']} madde): {', '.join(r['vars'])}", "═" * 70]
        if r.get("skipped"):
            L += ["  ⚠ Az madde (min 2 gerekli) — atlandı.", ""]
            continue
        al, om, rh, sb = r["alpha"], r["omega"], r["r_half"], r["sb"]
        albl, _ = badge(al, "alpha"); olbl, _ = badge(om, "omega")
        L += [f"  1. Cronbach α = {al:.4f}  [{albl}]",
              f"     {'✔ Eşik üzerinde.' if al >= alpha_thr else '✗ Eşik altı — revizyon önerilir.'}",
              "", "  α / ω İf Deleted:"]
        for i, v in enumerate(r["vars"]):
            ai = r["alpha_if_del"][i]; da = ai - al
            oi = r["omega_if_del"][i]; do = oi - om
            flag_a = " ← α ÇIKAR" if da > 0.01 else ""
            flag_o = " ← ω ÇIKAR" if do > 0.01 else ""
            L.append(f"    {v:<38} α-del={ai:.4f}(Δ={da:+.4f}){flag_a}  ω-del={oi:.4f}(Δ={do:+.4f}){flag_o}")
        L += ["", "  2. Madde-Toplam Korelasyonu (ITC):"]
        for i, v in enumerate(r["vars"]):
            itcv = r["itc"][i]; lv = r["fa_loadings"][i]
            ilbl, _ = badge(itcv, "itc")
            flag = " ⚠ ÇIKAR" if itcv < itc_thr else ""
            L.append(f"    {v:<38} ITC={itcv:.4f} [{ilbl}]  λ={lv:.4f}{flag}")
        low = [r["vars"][i] for i in range(len(r["vars"])) if r["itc"][i] < itc_thr]
        if low:
            L += ["", f"  ⚠ Düşük ITC: {', '.join(low)}"]
        L += ["", f"  3. Split-Half: r={rh:.4f}  SB={sb:.4f}  [{'✔ Yeterli' if sb >= 0.7 else '✗ Yetersiz'}]",
              "", f"  4. McDonald ω = {om:.4f}  [{olbl}]", ""]
        probs = []
        if al < alpha_thr: probs.append(f"α={al:.3f}<{alpha_thr}")
        if sb < 0.7: probs.append(f"SB={sb:.3f}<0.70")
        if low: probs.append(f"{len(low)} düşük ITC madde")
        L += ["  GENEL: " + ("✔ Yeterli güvenirlik." if not probs else
                              f"✗ Sorunlar: {' | '.join(probs)}  → Revizyon önerilir."), ""]
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════
#  STREAMLIT ARAYÜZÜ
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Güvenirlik Analizi", layout="wide", page_icon="🔍")
st.markdown(f"""
<style>
.stApp {{ background-color: {DARK_BG}; color: {TEXT}; }}
</style>
""", unsafe_allow_html=True)

st.title("🔍 Güvenirlik Analizi Dashboard")

if "df_orig" not in st.session_state:
    st.session_state.df_orig = None
    st.session_state.df = None
    st.session_state.if_stats = None
    st.session_state.results = None
    st.session_state.rev_info = []

# ── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Veri Yükleme")
    up = st.file_uploader("CSV / Excel dosyası", type=["csv", "xlsx", "xls", "xlsm"])
    url = st.text_input("...veya URL")
    if st.button("⬇ Yükle", use_container_width=True):
        try:
            if up is not None:
                data = load_dataframe(up, up.name)
            elif url.strip():
                raw = BytesIO(requests.get(url.strip(), timeout=30).content)
                data = load_dataframe(raw, url.strip())
            else:
                st.warning("Dosya seçin veya URL girin.")
                data = None
            if data is not None:
                st.session_state.df_orig = data
                st.session_state.df = data
                st.session_state.if_stats = None
                st.session_state.results = None
                st.success(f"{len(data)} satır × {data.shape[1]} değişken yüklendi.")
        except Exception as e:
            st.error(f"Yükleme hatası: {e}")

    st.divider()
    st.header("🌲 Isolation Forest")
    apply_if = st.checkbox("Analiz öncesi uygula")
    contamination = st.slider("Kirlilik (contamination)", 0.01, 0.40, 0.15, 0.01, disabled=not apply_if)
    n_estimators = st.number_input("Ağaç sayısı", 50, 500, 100, disabled=not apply_if)
    random_state = st.number_input("Random state", 0, 9999, 42, disabled=not apply_if)

    st.divider()
    st.header("⚙ Analiz Seçenekleri")
    split_label = st.selectbox("Split-Half yöntemi",
                                ["Tek-Çift (Odd-Even)", "İlk-İkinci Yarı", "Varyans Eşleşmeli"])
    sm_map = {"Tek-Çift (Odd-Even)": "odd_even", "İlk-İkinci Yarı": "first_sec",
              "Varyans Eşleşmeli": "matched"}
    alpha_thr = st.slider("Kritik Alpha eşiği", 0.50, 0.95, 0.70, 0.05)
    itc_thr = st.slider("Kritik ITC eşiği", 0.10, 0.60, 0.30, 0.05)
    use_color = st.checkbox("Renk kodlaması göster", value=True)

# ── ANA ALAN: KÜME DEĞİŞKEN SEÇİMİ ─────────────────────────────────────
if st.session_state.df_orig is None:
    st.info("Başlamak için soldan bir CSV/Excel dosyası yükleyin.")
    st.stop()

df_orig = st.session_state.df_orig
var_names = list(df_orig.columns)

st.subheader("✅ Küme Değişkenleri")
st.caption("Her küme için analiz edilecek değişkenleri seçin; ters puanlanacak maddeleri işaretleyin.")

cluster_map = {}
reverse_map = {}
cols_ui = st.columns(N_CLUSTERS)
for i in range(N_CLUSTERS):
    with cols_ui[i]:
        st.markdown(f"<span style='color:{CC[i % len(CC)]};font-weight:bold;'>● Küme {i+1}</span>",
                    unsafe_allow_html=True)
        defaults = [v for v in DEFAULT_CLUSTERS.get(i, []) if v in var_names]
        sel = st.multiselect("Değişkenler", var_names, default=defaults, key=f"clu_{i}", label_visibility="collapsed")
        rev = st.multiselect("Ters çevir", sel, key=f"rev_{i}", placeholder="Ters puanlanacaklar")
        if sel:
            cluster_map[i] = sel
        if rev:
            reverse_map[i] = rev

st.divider()
run = st.button("▶ ANALİZİ ÇALIŞTIR", type="primary", use_container_width=True)

if run:
    if not cluster_map:
        st.warning("En az bir küme için değişken seçin.")
        st.stop()

    df = df_orig
    if_stats = None
    if apply_if:
        all_cols = list({c for cols in cluster_map.values() for c in cols})
        with st.spinner("Isolation Forest çalışıyor…"):
            df, if_stats = apply_isolation_forest(df_orig, all_cols, n_estimators, contamination, random_state)

    df_scored = df.copy()
    rev_info = []
    for cid, rvars in reverse_map.items():
        for v in rvars:
            if v not in df_scored.columns:
                continue
            col = df_scored[v].dropna()
            if len(col) == 0:
                continue
            vmin, vmax = col.min(), col.max()
            df_scored[v] = (vmin + vmax) - df_scored[v]
            rev_info.append((cid, v, vmin, vmax))

    with st.spinner("Güvenirlik hesaplanıyor…"):
        results = run_reliability(df_scored, cluster_map, sm_map[split_label])

    st.session_state.df = df
    st.session_state.if_stats = if_stats
    st.session_state.results = results
    st.session_state.rev_info = rev_info
    st.session_state.params = {"alpha_thr": alpha_thr, "itc_thr": itc_thr, "split_label": split_label}

# ── SONUÇLAR ─────────────────────────────────────────────────────────────
if st.session_state.results:
    res = st.session_state.results
    params = st.session_state.params
    ifs = st.session_state.if_stats
    rev_info = st.session_state.rev_info
    df = st.session_state.df

    if ifs:
        st.success(f"🌲 IF: {ifs['n_before']} → {ifs['n_after']} gözlem "
                    f"({ifs['n_removed']} aykırı, %{ifs['pct_removed']:.1f} elendi)")

    tab_sum, tab_det, tab_plot, tab_del, tab_if, tab_txt = st.tabs(
        ["📊 Özet Tablo", "🔗 Madde Detayı", "📈 Özet Görsel",
         "🗑️ If Deleted", "🌲 Isolation Forest", "📋 Metin Raporu"])

    # --- Özet tablo ---
    with tab_sum:
        rows = []
        for cid, r in sorted(res.items()):
            if r.get("skipped"):
                rows.append([f"Küme {cid+1}", r["n_items"]] + ["—"] * 8)
                continue
            al, om = r["alpha"], r["omega"]
            albl, _ = badge(al, "alpha"); olbl, _ = badge(om, "omega")
            itc = r["itc"][~np.isnan(r["itc"])]
            mi = itc.min() if len(itc) else float("nan")
            ai = itc.mean() if len(itc) else float("nan")
            rows.append([f"Küme {cid+1}", r["n_items"], f"{al:.4f}", albl, f"{om:.4f}", olbl,
                         f"{r['r_half']:.4f}", f"{r['sb']:.4f}",
                         f"{mi:.4f}" if not np.isnan(mi) else "N/A",
                         f"{ai:.4f}" if not np.isnan(ai) else "N/A"])
        st.dataframe(pd.DataFrame(rows, columns=["Küme", "N", "Cronbach α", "α Yorum", "McDonald ω",
                                                  "ω Yorum", "Split-Half r", "Spearman-Brown",
                                                  "Min ITC", "Ort. ITC"]),
                     use_container_width=True, hide_index=True)

    # --- Madde detayı ---
    with tab_det:
        rows = []
        for cid, r in sorted(res.items()):
            if r.get("skipped"):
                for v in r["vars"]:
                    rows.append([f"K{cid+1}", v] + ["—"] * 8)
                continue
            al, om = r["alpha"], r["omega"]
            for i, v in enumerate(r["vars"]):
                itcv = r["itc"][i]
                aidv = r["alpha_if_del"][i]
                oidv = r["omega_if_del"][i]
                lamv = r["fa_loadings"][i]
                dalp = aidv - al
                domg = oidv - om
                ilbl, _ = badge(itcv, "itc")
                flags = []
                if not np.isnan(itcv) and itcv < params["itc_thr"]: flags.append("ITC ⚠")
                if not np.isnan(dalp) and dalp > 0.01: flags.append("α↑ sil")
                if not np.isnan(domg) and domg > 0.01: flags.append("ω↑ sil")
                status = "✔ OK" if not flags else " | ".join(flags)
                rows.append([f"K{cid+1}", v, f"{itcv:.4f}", ilbl, f"{aidv:.4f}", f"{dalp:+.4f}",
                             f"{oidv:.4f}", f"{domg:+.4f}", f"{lamv:.4f}", status])
        st.dataframe(pd.DataFrame(rows, columns=["Küme", "Değişken", "ITC", "ITC Yorum", "α İf Sil",
                                                  "Δα", "ω İf Sil", "Δω", "FA Yükü", "Durum"]),
                     use_container_width=True, hide_index=True)

    # --- Özet görsel ---
    with tab_plot:
        fig = plot_summary(res, params["alpha_thr"], params["itc_thr"])
        if fig: st.pyplot(fig)

    # --- If deleted ---
    with tab_del:
        fig = plot_if_deleted(res)
        if fig: st.pyplot(fig)

    # --- Isolation forest raporu ---
    with tab_if:
        if ifs:
            st.markdown(f"""
            - Contamination: **{ifs['contamination']:.2f}**
            - Ağaç sayısı: **{ifs['n_estimators']}**
            - Random state: **{ifs['random_state']}**
            - Analiz öncesi: **{ifs['n_before']}**, sonrası: **{ifs['n_after']}**
            - Elenen: **{ifs['n_removed']}** (%{ifs['pct_removed']:.2f})
            - Anomali skoru — ort: {ifs['score_mean']:.4f}, min: {ifs['score_min']:.4f}, maks: {ifs['score_max']:.4f}
            """)
            st.pyplot(plot_if_report(ifs))
        else:
            st.info("Isolation Forest uygulanmadı.")

    # --- Metin raporu ---
    with tab_txt:
        report = build_text_report(df, res, params["alpha_thr"], params["itc_thr"],
                                    params["split_label"], rev_info, ifs)
        st.text(report)
        st.download_button("💾 Raporu İndir (.txt)", report, file_name="guvenirlik_raporu.txt",
                            use_container_width=True)
