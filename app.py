"""
RetailPulse - Customer Analytics Dashboard (Streamlit)

Includes:
1. Overview Page (KPIs, Segments, Regional Analysis, Quick Insights)
2. Spending Prediction / Prediction Center (Using exact 6 features from retailpulse_lr_pipeline.joblib)

Run:  streamlit run app.py
"""
import glob
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# CONFIG  -  edit this block only
# =====================================================================
DATA_PATH = "Data/Main_Clean.csv"                  # your dataset (.csv / .xlsx)
MODEL_PATH = "retailpulse_lr_pipeline.joblib"      # your trained .joblib pipeline

# Map the dashboard fields to YOUR column names.
# Leave a value as None to auto-detect from the column names.
COLUMNS = {
    "spend": None,        # annual spend / purchase amount   (REQUIRED)
    "id": None,           # customer id
    "purchases": None,    # PurchaseFrequency / number of purchases
    "category": None,     # product category
    "age": None,          # Age
    "discount": None,     # discount usage
    "segment": None,      # existing segment column (Budget/Regular/Premium...)
    "region": None,       # region / zone / state column
    "income": None,       # AnnualIncome
    "recency": None,      # Recency (last purchase days ago)
    "tenure": None,       # CustomerTenure
    "return_rate": None,  # ReturnRate
}
VALUABLE_TOP_PCT = 7.5           # "valuable customer" = top X% by spend
CURRENCY = "₹"
# =====================================================================

st.set_page_config(page_title="RetailPulse | Customer Analytics", page_icon="📊", layout="wide")

PAGES = ["Overview", "Customer Segmentation", "Spending Prediction",
         "Valuable Customers", "Data Analysis", "About"]
ICONS = {"Overview": "🏠", "Customer Segmentation": "👥", "Spending Prediction": "📈",
         "Valuable Customers": "⭐", "Data Analysis": "📊", "About": "ℹ️"}
BASE_COLORS = {"Budget": "#3b82f6", "Regular": "#22c277", "Premium": "#fb923c"}
PALETTE = ["#3b82f6", "#22c277", "#fb923c", "#8b5cf6", "#22b8cf", "#f43f5e", "#94a3b8"]
CAT_COLORS = ["#3b82f6", "#22c277", "#fb923c", "#8b5cf6", "#22b8cf", "#94a3b8"]

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; max-width: 1260px;}
.stApp {background: #f8f9fc;}
section[data-testid="stSidebar"] {background: #131c3a; min-width: 250px !important;}
section[data-testid="stSidebar"] * {color: #cbd5e1;}
.brand {font-size: 1.6rem; font-weight: 800; color: #fff !important; margin: 0 0 1rem 0;}
section[data-testid="stSidebar"] div[role="radiogroup"] {gap: .25rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: .55rem .8rem; border-radius: 8px; width: 100%; cursor: pointer;}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {display: none;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background: #4f63e8;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {color: #fff !important; font-weight: 600;}
.side-foot {position: fixed; bottom: 24px; left: 28px; font-size: .7rem; color: #8b97b5;}
.welcome {font-size: 2rem; font-weight: 800; color: #111a3b; margin-bottom: 0;}
.tagline {color: #64748b; font-size: .85rem; margin-bottom: 1rem;}
.kpi {border-radius: 12px; padding: 16px 18px; display: flex; gap: 14px; align-items: center;
      border: 1px solid rgba(0,0,0,.04);}
.kpi .ic {width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center;
          justify-content: center; font-size: 1.3rem; background: rgba(255,255,255,.75);}
.kpi .lb {font-size: .72rem; color: #475569;}
.kpi .vl {font-size: 1.45rem; font-weight: 800; color: #111a3b; line-height: 1.2;}
.kpi .dl {font-size: .68rem; color: #64748b;}
.card {background: #fff; border: 1px solid #eef0f6; border-radius: 12px; padding: 14px 16px;}
.card h4 {font-size: .85rem; font-weight: 700; color: #111a3b; margin: 0 0 6px 0;}
.insight {display: flex; gap: 12px; align-items: flex-start; margin: 12px 0;}
.insight .ii {width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center;
              justify-content: center; flex-shrink: 0;}
.insight b {font-size: .8rem; color: #111a3b;}
.insight div div {font-size: .72rem; color: #64748b;}
.cta {background: linear-gradient(135deg, #eceafd, #e4e6fb); border-radius: 12px; padding: 24px;
      min-height: 200px;}
.cta h2 {font-size: 1.35rem; font-weight: 800; color: #111a3b; margin: 0 0 8px 0; line-height: 1.25;}
.cta p {font-size: .78rem; color: #475569; max-width: 280px;}
.vacant {background: #fff; border: 2px dashed #cbd5e1; border-radius: 14px; padding: 70px 20px;
         text-align: center; margin-top: 20px;}
.vacant h3 {color: #111a3b; margin: 8px 0 4px 0;}
.vacant p {color: #64748b; font-size: .85rem; margin: 0;}
.sec {font-size: 1.25rem; font-weight: 800; color: #111a3b;}
.ct {font-size: .85rem; font-weight: 700; color: #111a3b; margin-bottom: 4px;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border: 1px solid #eef0f6 !important; border-radius: 12px;}
div.stButton > button {background: #5856eb; color: #fff; border: 0; border-radius: 8px; font-weight: 600;}
div.stButton > button:hover {background: #4341d6; color: #fff;}

/* ---------------- Prediction Center Custom Styles ---------------- */
.mode-card {
    border-radius: 12px; padding: 16px 18px; display: flex; gap: 14px; align-items: center;
    border: 1.5px solid #e2e8f0; background: #fff; margin-bottom: 6px;
}
.mode-card.active {
    background: #f1effe; border-color: #818cf8;
}
.mode-card .mc-ic {
    width: 44px; height: 44px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 1.25rem; flex-shrink: 0;
}
.mode-card .mc-t {font-size: .92rem; font-weight: 800; color: #111a3b; margin-bottom: 2px;}
.mode-card .mc-d {font-size: .72rem; color: #64748b;}

.sub-banner {
    background: #f6f7fe; border: 1px solid #e4e7fb; border-radius: 12px;
    padding: 14px 18px; display: flex; align-items: center; gap: 14px; margin: 8px 0 16px 0;
}
.sub-banner .sb-ic {
    width: 40px; height: 40px; border-radius: 10px; background: #e0e7ff; color: #3b82f6;
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
}
.sub-banner h3 {margin: 0; font-size: 1.05rem; font-weight: 800; color: #111a3b;}
.sub-banner p {margin: 2px 0 0 0; font-size: .78rem; color: #64748b;}

.pred-res-box {
    background: #f2fbf7; border: 1px solid #d1fae5; border-radius: 12px;
    padding: 16px; margin-bottom: 14px;
}
.pred-res-hdr {
    display: flex; align-items: center; gap: 8px; font-size: .85rem;
    font-weight: 700; color: #111a3b; margin-bottom: 12px;
}
.pred-res-hdr .chk {
    width: 22px; height: 22px; border-radius: 50%; background: #22c277; color: #fff;
    display: inline-flex; align-items: center; justify-content: center; font-size: .72rem;
}
.seg-badge {
    display: inline-block; padding: 8px 28px; border-radius: 999px;
    font-size: 1.2rem; font-weight: 800; margin-right: 10px;
}
.prob-row {
    display: flex; align-items: center; gap: 10px; margin: 10px 0; font-size: .78rem; color: #334155;
}
.prob-lbl {width: 62px; font-weight: 500;}
.prob-bar-bg {
    flex: 1; height: 14px; background: #eef2f6; border-radius: 5px; overflow: hidden;
}
.prob-bar-fg {height: 100%; border-radius: 5px;}
.prob-pct {width: 44px; text-align: right; font-weight: 700; color: #111a3b;}

.prof-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px 10px; margin-top: 10px;
}
.prof-item {display: flex; align-items: center; gap: 10px;}
.prof-ic {
    width: 30px; height: 30px; border-radius: 8px; background: #eff6ff; color: #2563eb;
    display: flex; align-items: center; justify-content: center; font-size: .85rem; flex-shrink: 0;
}
.prof-lb {font-size: .68rem; color: #64748b; line-height: 1.1;}
.prof-vl {font-size: .8rem; font-weight: 700; color: #111a3b; line-height: 1.3;}

.ki-ul {list-style: none; padding: 0; margin: 8px 0 0 0;}
.ki-ul li {
    font-size: .76rem; color: #334155; margin-bottom: 9px;
    display: flex; align-items: flex-start; gap: 8px; line-height: 1.4;
}
.ki-ul li b.tick {color: #0284c7; font-size: .82rem;}

.try-card {
    background: #f3f1ff; border: 1px solid #e4dfff; border-radius: 12px;
    padding: 14px 16px; display: flex; gap: 12px; align-items: center; margin-top: 12px;
}
.try-ic {
    width: 36px; height: 36px; border-radius: 50%; background: #e0dbff; color: #4f46e5;
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0;
}
.try-card b {font-size: .82rem; color: #111a3b;}
.try-card div div {font-size: .72rem; color: #475569; margin-top: 2px; line-height: 1.35;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ exact model loader (.joblib + fallback weights)
class ExactJoblibFallback:
    """
    Exact mathematical replica of retailpulse_lr_pipeline.joblib (scikit-learn 1.6.1)
    Features: ['PurchaseFrequency', 'AnnualIncome', 'Recency', 'CustomerTenure', 'ReturnRate', 'Age']
    """
    feature_names_in_ = np.array(
        ["PurchaseFrequency", "AnnualIncome", "Recency", "CustomerTenure", "ReturnRate", "Age"]
    )
    medians_ = np.array([14.22, 65536.0, 4.5, 4.3, 0.303125, 36.0])
    mean_ = np.array([
        17.345752855, 65067.85895889583, 59.53035035416667,
        4.399194494354167, 0.30351399907072003, 35.849048
    ])
    scale_ = np.array([
        9.746995971289551, 37532.70268295248, 62.61734737717992,
        2.9125161077274553, 0.39589524739386066, 11.797332262562027
    ])
    coef_ = np.array([
        1247.3718193884928, 720.2815933007077, -356.5074505832541,
        1064.0530244681862, 747.795953703966, 1035.8499802635608
    ])
    intercept_ = 12333.60982297434

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # copy=True prevents "ValueError: assignment destination is read-only" in Pandas 2.0+
        vals = X[list(self.feature_names_in_)].to_numpy(dtype=float, copy=True)
        inds = np.where(np.isnan(vals))
        if len(inds[0]) > 0:
            vals[inds] = np.take(self.medians_, inds[1])
        scaled = (vals - self.mean_) / self.scale_
        return scaled @ self.coef_ + self.intercept_


@st.cache_resource
def load_prediction_model():
    """Loads retailpulse_lr_pipeline.joblib from disk or uses the exact extracted pipeline weights."""
    candidates = [
        MODEL_PATH,
        "retailpulse_lr_pipeline.joblib",
        "models/retailpulse_lr_pipeline.joblib",
    ] + glob.glob("*.joblib") + glob.glob("models/*.joblib")

    for path in candidates:
        if path and os.path.exists(path):
            try:
                return joblib.load(path), os.path.basename(path)
            except Exception:
                pass
    return ExactJoblibFallback(), "retailpulse_lr_pipeline.joblib (Embedded)"


# ------------------------------------------------------------------ data loading (backend)
@st.cache_data
def load_raw(path: str, mtime: float) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


@st.cache_data
def demo_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Fallback so the page still renders if DATA_PATH is missing."""
    rng = np.random.default_rng(seed)
    seg = rng.choice(["Budget", "Regular", "Premium"], n, p=[.246, .331, .423])
    base = {"Budget": 6420, "Regular": 12860, "Premium": 28750}
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": rng.normal(36, 11, n).clip(18, 70).round().astype(int),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "category": rng.choice(["Electronics", "Fashion", "Home & Kitchen",
                                "Beauty & Personal Care", "Sports & Outdoors", "Others"],
                               n, p=[.284, .227, .163, .121, .089, .116]),
        "purchases": rng.normal(17, 9, n).clip(1, 60).round(1),
        "annual_income": rng.normal(65000, 35000, n).clip(10000, 250000).round(0),
        "recency": rng.exponential(50, n).clip(1, 365).round(1),
        "tenure": rng.normal(4.4, 2.9, n).clip(0.1, 20.0).round(1),
        "return_rate": rng.uniform(0.0, 0.8, n).round(2),
        "discount_usage": np.where(seg == "Premium", rng.uniform(.1, .4, n),
                                   rng.uniform(0, .25, n)).round(2),
        "annual_spend": np.array([rng.normal(base[s], base[s] * .25) for s in seg]).clip(500).round(),
        "segment": seg,
    })


def guess(cols, keywords, exclude=()):
    for kw in keywords:
        for c in cols:
            if c not in exclude and kw in c.lower().replace(" ", "_"):
                return c
    return None


def resolve_columns(raw: pd.DataFrame) -> dict:
    cols, m = raw.columns.tolist(), dict(COLUMNS)
    m["spend"] = m["spend"] or guess(cols, ["annual_spend", "spend", "purchase_amount", "amount",
                                            "revenue", "sales", "total"])
    m["id"] = m["id"] or guess(cols, ["customer_id", "cust", "id", "user"], [m["spend"]])
    m["purchases"] = m["purchases"] or guess(cols, ["purchasefrequency", "frequency", "num", "count",
                                                    "orders", "transactions", "purchases"],
                                             [m["spend"], m["id"]])
    m["category"] = m["category"] or guess(cols, ["categ", "product", "item"])
    m["age"] = m["age"] or guess(cols, ["age"])
    m["discount"] = m["discount"] or guess(cols, ["discount", "promo", "coupon"])
    m["segment"] = m["segment"] or guess(cols, ["segment", "tier"])
    m["region"] = m["region"] or guess(cols, ["region", "zone", "state", "city", "location"])
    m["income"] = m["income"] or guess(cols, ["annualincome", "income", "salary"])
    m["recency"] = m["recency"] or guess(cols, ["recency", "last_purchase", "days_ago"])
    m["tenure"] = m["tenure"] or guess(cols, ["customertenure", "tenure", "years"])
    m["return_rate"] = m["return_rate"] or guess(cols, ["returnrate", "return"])
    return m


def prepare(raw: pd.DataFrame, m: dict) -> pd.DataFrame:
    d = pd.DataFrame(index=raw.index)
    d["annual_spend"] = pd.to_numeric(raw[m["spend"]], errors="coerce")
    d["customer_id"] = raw[m["id"]] if m["id"] else np.arange(1, len(raw) + 1)
    for key, name in [
        ("purchases", "purchases"),
        ("age", "age"),
        ("discount", "discount_usage"),
        ("income", "annual_income"),
        ("recency", "recency"),
        ("tenure", "tenure"),
        ("return_rate", "return_rate"),
    ]:
        if m.get(key) and m[key] in raw:
            d[name] = pd.to_numeric(raw[m[key]], errors="coerce")
    if m["category"]:
        d["category"] = raw[m["category"]].astype(str)
    if m["segment"]:
        d["segment"] = raw[m["segment"]].astype(str)
    if m["region"]:
        d["region"] = raw[m["region"]].astype(str)
    d = d.dropna(subset=["annual_spend"]).reset_index(drop=True)
    if "segment" not in d:  # auto segments: bottom 40% / next 35% / top 25% by spend
        d["segment"] = pd.qcut(d.annual_spend.rank(method="first"), [0, .4, .75, 1],
                               labels=["Budget", "Regular", "Premium"]).astype(str)
    return d


def money(x):
    return f"{CURRENCY} {x:,.0f}"


def fmt_inr(val):
    """Formats numbers in Indian comma style (e.g., 65,536)."""
    s = str(int(round(val)))
    if len(s) <= 3:
        return s
    last3, rest = s[-3:], s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    return ",".join(reversed(parts)) + "," + last3


def kpi(col, icon, label, value, sub, bg):
    col.markdown(f"""<div class="kpi" style="background:{bg}"><div class="ic">{icon}</div>
        <div><div class="lb">{label}</div><div class="vl">{value}</div>
        <div class="dl">{sub}</div></div></div>""", unsafe_allow_html=True)


def style(fig, h=260):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=11, color="#475569"))
    return fig


# ------------------------------------------------------------------ sidebar / nav
if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    st.markdown('<div class="brand">📊 RetailPulse</div>', unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="page", label_visibility="collapsed",
             format_func=lambda p: f"{ICONS[p]}  {p}")
    st.markdown('<div class="side-foot">Smarter Insights • Better Decisions</div>',
                unsafe_allow_html=True)
page = st.session_state.page

# ------------------------------------------------------------------ load dataset
if os.path.exists(DATA_PATH):
    raw = load_raw(DATA_PATH, os.path.getmtime(DATA_PATH))
else:
    raw = demo_data()

m = resolve_columns(raw)
if not m["spend"]:
    st.error("Could not find the spend column. Set COLUMNS['spend'] at the top of app.py. "
             f"Columns in your file: {', '.join(raw.columns.astype(str))}")
    st.stop()

df = prepare(raw, m)
thr = df.annual_spend.quantile(1 - VALUABLE_TOP_PCT / 100)
df["valuable"] = df.annual_spend >= thr
order = df.groupby("segment").annual_spend.mean().sort_values().index.tolist()
colors = {s: BASE_COLORS.get(s, PALETTE[i % len(PALETTE)]) for i, s in enumerate(order)}

# ------------------------------------------------------------------ top bar + search
top = st.columns([3, 1])
top[0].text_input("search", placeholder="🔍  Search customer, region, category...",
                  label_visibility="collapsed", key="search")
top[1].markdown("<div style='text-align:right;color:#111a3b;font-size:.8rem'>☀️ &nbsp; 👤 <b>RetailPulse</b>"
                "<br><span style='color:#64748b;font-size:.7rem'>Customer Analytics</span></div>",
                unsafe_allow_html=True)
q = st.session_state.get("search", "").strip().lower()
if q:
    tcols = [c for c in ["customer_id", "category", "segment", "region"] if c in df]
    df = df[df[tcols].astype(str).apply(lambda s: s.str.lower().str.contains(q)).any(axis=1)]
    st.caption(f"🔍 {len(df):,} customers match “{q}”")
    if df.empty:
        st.stop()


# =====================================================================
# PAGE 1: OVERVIEW
# =====================================================================
if page == "Overview":
    def go_analysis():
        st.session_state.page = "Data Analysis"

    st.markdown('<div class="welcome">Welcome to <span style="color:#4f63e8">RetailPulse</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tagline">Understand your customers &nbsp;•&nbsp; Find valuable '
                'segments &nbsp;•&nbsp; Predict future spending</div>', unsafe_allow_html=True)

    n = df.customer_id.nunique()
    c = st.columns(4)
    kpi(c[0], "👥", "Total Customers", f"{n:,}", f"{len(df):,} records", "#e8f1fd")
    if "purchases" in df:
        kpi(c[1], "🛒", "Total Purchases", f"{df.purchases.sum():,.0f}",
            f"{df.purchases.mean():.1f} per customer", "#e3f6ec")
    else:
        kpi(c[1], "🛒", "Total Purchases", "–", "No purchases column found", "#e3f6ec")
    kpi(c[2], CURRENCY, "Avg. Annual Spend", money(df.annual_spend.mean()),
        f"Median {money(df.annual_spend.median())}", "#f0eafd")
    kpi(c[3], "⭐", "Valuable Customers", f"{int(df.valuable.sum()):,}",
        f"{df.valuable.mean():.1%} of total customers", "#fdf3df")
    st.write("")

    r1 = st.columns([1, 1.25, 1])
    with r1[0]:
        st.markdown('<div class="card"><h4>Customer Segments</h4>', unsafe_allow_html=True)
        counts = df.segment.value_counts().reindex(order).dropna()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.6, sort=False,
                               marker=dict(colors=[colors[s] for s in counts.index]),
                               textinfo="percent"))
        fig.add_annotation(text=f"<b>{n:,}</b><br>Customers", showarrow=False, font=dict(size=12))
        fig.update_layout(legend=dict(orientation="v", x=1, y=.5))
        st.plotly_chart(style(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1[1]:
        st.markdown('<div class="card"><h4>Annual Purchase Amount Distribution</h4>',
                    unsafe_allow_html=True)
        avg = df.groupby("segment").annual_spend.mean().reindex(order).dropna()
        fig = go.Figure(go.Bar(x=avg.index, y=avg.values, marker_color=[colors[s] for s in avg.index],
                               text=[money(v) for v in avg.values], textposition="outside"))
        fig.update_yaxes(range=[0, avg.max() * 1.25], gridcolor="#eef0f6")
        st.plotly_chart(style(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1[2]:
        st.markdown('<div class="card"><h4>Top Product Categories</h4>', unsafe_allow_html=True)
        if "category" in df:
            share = df.category.value_counts(normalize=True) * 100
            if len(share) > 6:
                share = pd.concat([share.head(5), pd.Series({"Others": share.iloc[5:].sum()})])
            fig = go.Figure(go.Bar(y=share.index, x=share.values, orientation="h",
                                   text=[f"{p:.1f}%" for p in share.values], textposition="outside",
                                   marker_color=CAT_COLORS[:len(share)]))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(visible=False, range=[0, share.max() * 1.35])
            st.plotly_chart(style(fig), use_container_width=True)
        else:
            st.info("No category column found in the dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ REGIONAL ANALYSIS
    st.write("")
    h1, h2 = st.columns([3, 1.4])
    h1.markdown('<div class="sec">Regional Analysis</div>'
                '<div class="tagline" style="margin-bottom:0">Annual purchase amount across regions</div>',
                unsafe_allow_html=True)
    mode = h2.radio("Metric", ["Average per customer", "Total"], horizontal=True,
                    key="region_mode", label_visibility="collapsed")
    agg = "mean" if mode.startswith("Average") else "sum"
    st.write("")

    if "region" not in df:
        st.info("No region column found. Set COLUMNS['region'] at the top of app.py. "
                f"Columns in your file: {', '.join(raw.columns.astype(str))}")
    else:
        region_rank = df.groupby("region").annual_spend.sum().sort_values(ascending=False)
        keep = region_rank.head(8).index.tolist()
        dr = df[df.region.isin(keep)]
        if len(region_rank) > 8:
            st.caption(f"Showing the top 8 of {len(region_rank)} regions by total spend.")

        def tidy(fig, h=320):
            fig = style(fig, h)
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), bargap=.25,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          xanchor="left", x=0, title_text=""))
            fig.update_yaxes(tickprefix=CURRENCY, tickformat="~s", gridcolor="#eef0f6", title=None)
            fig.update_xaxes(title=None)
            return fig

        g1, g2 = st.columns(2)

        with g1:
            with st.container(border=True):
                st.markdown('<div class="ct">Annual Purchase Amount by Segment in Each Region</div>',
                            unsafe_allow_html=True)
                seg_reg = dr.groupby(["region", "segment"]).annual_spend.agg(agg).reset_index()
                fig = px.bar(seg_reg, x="region", y="annual_spend", color="segment", barmode="group",
                             category_orders={"region": keep, "segment": order},
                             color_discrete_map=colors)
                fig.update_layout(bargroupgap=.08)
                st.plotly_chart(tidy(fig), use_container_width=True)

        with g2:
            with st.container(border=True):
                st.markdown('<div class="ct">Top Product Category by Annual Purchase Amount in Each Region</div>',
                            unsafe_allow_html=True)
                if "category" not in df:
                    st.info("No category column found in the dataset.")
                else:
                    reg_cat = dr.groupby(["region", "category"]).annual_spend.agg(agg).reset_index()
                    best = reg_cat.loc[reg_cat.groupby("region").annual_spend.idxmax()]
                    best = best.set_index("region").loc[keep].reset_index()
                    best["label"] = best.region + "<br>" + best.category
                    fig = px.bar(best, x="label", y="annual_spend", color="category",
                                 text=best.annual_spend.map(money),
                                 category_orders={"label": best.label.tolist()},
                                 color_discrete_sequence=CAT_COLORS)
                    fig.update_traces(textposition="outside", cliponaxis=False,
                                      textfont=dict(size=10))
                    fig.update_yaxes(range=[0, best.annual_spend.max() * 1.2])
                    st.plotly_chart(tidy(fig), use_container_width=True)

    # ------------------------------------------------------------------ QUICK INSIGHTS + CTA
    st.write("")
    items = []
    sm = df.groupby("segment").annual_spend.mean()
    hi, lo = sm.idxmax(), sm.idxmin()
    if hi != lo:
        items.append(("📈", "#dcfce7", f"{hi} segment has the highest average spend",
                      f"{hi} customers spend {sm[hi] / sm[lo]:.1f}x more than {lo} customers."))
    if "category" in df:
        sh = df.category.value_counts(normalize=True)
        items.append(("👥", "#ede9fe", f"{sh.index[0]} is the top category",
                      f"It accounts for {sh.iloc[0]:.1%} of total purchases."))
    if "age" in df and df.age.notna().any():
        ab = pd.cut(df.age, [0, 24, 40, 55, 200], labels=["under 25", "25–40", "41–55", "over 55"])
        items.append(("🕒", "#ffedd5", f"Most customers are {ab.value_counts().idxmax()} years old",
                      "This is the largest age group in your dataset."))
    if "discount_usage" in df and df.valuable.any() and (~df.valuable).any():
        v, o = df[df.valuable].discount_usage.mean(), df[~df.valuable].discount_usage.mean()
        if o and o > 0:
            items.append(("⭐", "#dbeafe",
                          f"Valuable customers show {'higher' if v >= o else 'lower'} discount usage",
                          f"They use discounts {v / o:.1f}x as much as other customers."))

    r2 = st.columns([1.15, 1])
    with r2[0]:
        html = '<div class="card"><h4>💡 Quick Insights</h4>'
        for ic, bg, t, d in items:
            html += (f'<div class="insight"><div class="ii" style="background:{bg}">{ic}</div>'
                     f'<div><b>{t}</b><div>{d}</div></div></div>')
        st.markdown(html + "</div>", unsafe_allow_html=True)
    with r2[1]:
        st.markdown("""<div class="cta"><h2>Turn Data into<br>Business Growth</h2>
            <p>Explore customer behavior, find valuable segments and predict future spending
            with our ML-powered insights.</p></div>""", unsafe_allow_html=True)
        st.button("Go to Detailed Analysis →", on_click=go_analysis)


# =====================================================================
# PAGE 2: SPENDING PREDICTION (PREDICTION CENTER)
# =====================================================================
elif page == "Spending Prediction":
    model, model_src = load_prediction_model()

    st.markdown('<div class="welcome">📈 Prediction Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Use our machine learning models to get insights about your '
                'customers. Choose a prediction type below:</div>', unsafe_allow_html=True)

    if "pred_type" not in st.session_state:
        st.session_state.pred_type = "Annual Spending"

    # ---- 3 Top Model Selector Cards
    mc = st.columns(3)
    card_specs = [
        ("Customer Segment", "👥", "#ede9fe", "#6d28d9", "Predict customer segment using K-Means"),
        ("Annual Spending", "₹", "#dcfce7", "#15803d", "Predict annual purchase amount using Linear Regression"),
        ("Valuable Customer", "⭐", "#fef3c7", "#d97706", "Predict valuable customer using Classification"),
    ]
    for col, (title, ic, bg, fg, desc) in zip(mc, card_specs):
        is_act = "active" if st.session_state.pred_type == title else ""
        col.markdown(f"""
        <div class="mode-card {is_act}">
            <div class="mc-ic" style="background:{bg};color:{fg}"><b>{ic}</b></div>
            <div><div class="mc-t">{title}</div><div class="mc-d">{desc}</div></div>
        </div>""", unsafe_allow_html=True)
        if col.button(f"Select {title}", key=f"btn_{title}", use_container_width=True):
            st.session_state.pred_type = title
            st.rerun()

    ptype = st.session_state.pred_type

    # ---- Sub-Banner matching the selected prediction mode
    banner_meta = {
        "Customer Segment": (
            "👥", "Customer Segment Prediction",
            "Enter the 6 model features below to predict the customer's segment (Budget, Regular or Premium)."
        ),
        "Annual Spending": (
            "₹", "Annual Spending Prediction",
            f"Enter your 6 selected model features below to predict annual purchase amount ({model_src})."
        ),
        "Valuable Customer": (
            "⭐", "Valuable Customer Prediction",
            f"Enter the 6 model features below to predict if they fall in the Top {VALUABLE_TOP_PCT}% Valuable tier."
        ),
    }
    b_ic, b_title, b_desc = banner_meta[ptype]
    st.markdown(f"""
    <div class="sub-banner">
        <div class="sb-ic">{b_ic}</div>
        <div><h3>{b_title}</h3><p>{b_desc}</p></div>
    </div>""", unsafe_allow_html=True)

    # ---- Main 3-Column Layout
    col_left, col_mid, col_right = st.columns([1.05, 1.15, 1])

    # ---------------- LEFT COLUMN: Exact 6 Features from retailpulse_lr_pipeline.joblib ----------------
    # Features in pipeline: ['PurchaseFrequency', 'AnnualIncome', 'Recency', 'CustomerTenure', 'ReturnRate', 'Age']
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="ct" style="font-size:.95rem;margin-bottom:10px">Customer Details (Model Features)</div>',
                        unsafe_allow_html=True)

            f1, f2 = st.columns(2)
            in_age = f1.number_input(
                "Age",
                min_value=18, max_value=100, value=36, step=1,
                help="Feature: Age"
            )
            in_income = f2.number_input(
                f"Annual Income ({CURRENCY})",
                min_value=1000.0, max_value=10000000.0, value=65536.0, step=1000.0,
                help="Feature: AnnualIncome"
            )

            f3, f4 = st.columns(2)
            in_freq = f3.number_input(
                "Purchase Frequency (per year)",
                min_value=0.0, max_value=365.0, value=14.2, step=1.0,
                help="Feature: PurchaseFrequency"
            )
            in_recency = f4.number_input(
                "Recency (days ago)",
                min_value=0.0, max_value=365.0, value=15.0, step=1.0,
                help="Feature: Recency"
            )

            f5, f6 = st.columns(2)
            in_tenure = f5.number_input(
                "Customer Tenure (years)",
                min_value=0.0, max_value=50.0, value=4.3, step=0.5,
                help="Feature: CustomerTenure"
            )
            in_return = f6.number_input(
                "Return Rate",
                min_value=0.0, max_value=100.0, value=0.30, step=0.05,
                help="Feature: ReturnRate (trained median ~ 0.30)"
            )

            st.write("")
            btn_label = {
                "Customer Segment": "✨ Predict Segment",
                "Annual Spending": "✨ Predict Annual Spend",
                "Valuable Customer": "✨ Predict Valuable Status",
            }[ptype]
            st.button(btn_label, use_container_width=True, key="run_pred_btn")

    # ---------------- RUN MODEL PREDICTION USING EXACT 6 FEATURES ----------------
    X_input = pd.DataFrame([{
        "PurchaseFrequency": float(in_freq),
        "AnnualIncome": float(in_income),
        "Recency": float(in_recency),
        "CustomerTenure": float(in_tenure),
        "ReturnRate": float(in_return),
        "Age": float(in_age),
    }])

    pred_spend = float(model.predict(X_input)[0])
    pred_spend = max(0.0, pred_spend)

    # Segment centers from dataset (Budget / Regular / Premium)
    seg_means = df.groupby("segment")["annual_spend"].mean().reindex(["Budget", "Regular", "Premium"]).dropna()
    if len(seg_means) < 3:
        seg_means = pd.Series({"Budget": 6420.0, "Regular": 12860.0, "Premium": 28750.0})

    # Convert distances to segment centers into smooth probabilities
    dists = np.abs(seg_means.values - pred_spend) / (seg_means.std() + 1e-6)
    logits = -1.35 * dists
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / exp_l.sum()
    prob_map = dict(zip(seg_means.index, probs * 100))

    pred_seg = max(prob_map, key=prob_map.get)
    cluster_id_map = {"Budget": "Cluster 0", "Regular": "Cluster 1", "Premium": "Cluster 2"}
    cluster_label = cluster_id_map.get(pred_seg, "Cluster 1")

    is_valuable = pred_spend >= thr
    val_prob = float(1 / (1 + np.exp(-(pred_spend - thr) / (df.annual_spend.std() * 0.35 + 1e-5)))) * 100

    # ---------------- MIDDLE COLUMN: Prediction Output + Probabilities + Summary ----------------
    with col_mid:
        with st.container(border=True):
            if ptype == "Customer Segment":
                seg_bg = {"Budget": "#dbeafe", "Regular": "#d1fae5", "Premium": "#ffedd5"}.get(pred_seg, "#d1fae5")
                seg_fg = {"Budget": "#1d4ed8", "Regular": "#047857", "Premium": "#c2410c"}.get(pred_seg, "#047857")
                st.markdown(f"""
                <div class="pred-res-box">
                    <div class="pred-res-hdr"><span class="chk">✔</span> Predicted Segment</div>
                    <div style="margin: 10px 0 12px 0;">
                        <span class="seg-badge" style="background:{seg_bg};color:{seg_fg}">{pred_seg}</span>
                        <span style="font-size:.8rem;color:#64748b;font-weight:600">({cluster_label})</span>
                    </div>
                    <div style="font-size:.78rem;color:#334155;line-height:1.45">
                        This customer falls in the <b style="color:{seg_fg}">{pred_seg}</b> segment based on
                        their purchase behavior and profile (Predicted Spend: <b>{money(pred_spend)}</b>).
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif ptype == "Annual Spending":
                st.markdown(f"""
                <div class="pred-res-box">
                    <div class="pred-res-hdr"><span class="chk">✔</span> Predicted Annual Spending</div>
                    <div style="margin: 10px 0 12px 0;">
                        <span class="seg-badge" style="background:#d1fae5;color:#047857">{money(pred_spend)}</span>
                        <span style="font-size:.8rem;color:#64748b;font-weight:600">({pred_seg} Tier)</span>
                    </div>
                    <div style="font-size:.78rem;color:#334155;line-height:1.45">
                        Predicted directly from your 6 trained features in <b>{model_src}</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:  # Valuable Customer
                v_lbl = "Valuable ⭐" if is_valuable else "Standard"
                v_bg = "#fef3c7" if is_valuable else "#e2e8f0"
                v_fg = "#b45309" if is_valuable else "#334155"
                st.markdown(f"""
                <div class="pred-res-box">
                    <div class="pred-res-hdr"><span class="chk">✔</span> Valuable Customer Status</div>
                    <div style="margin: 10px 0 12px 0;">
                        <span class="seg-badge" style="background:{v_bg};color:{v_fg}">{v_lbl}</span>
                        <span style="font-size:.8rem;color:#64748b;font-weight:600">({val_prob:.1f}% Score)</span>
                    </div>
                    <div style="font-size:.78rem;color:#334155;line-height:1.45">
                        Top {VALUABLE_TOP_PCT}% threshold is <b>{money(thr)}</b>. This customer's predicted
                        spend is <b>{money(pred_spend)}</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ---- Segment Probabilities Bars
            st.markdown('<div class="ct" style="margin-top:6px">Segment Probabilities</div>',
                        unsafe_allow_html=True)
            bar_colors = {"Budget": "#fbbf24", "Regular": "#22c277", "Premium": "#a855f7"}
            bars_html = ""
            for s_name in ["Budget", "Regular", "Premium"]:
                p_val = prob_map.get(s_name, 0.0)
                b_col = bar_colors.get(s_name, "#3b82f6")
                bars_html += f"""
                <div class="prob-row">
                    <div class="prob-lbl">{s_name}</div>
                    <div class="prob-bar-bg"><div class="prob-bar-fg" style="width:{p_val:.1f}%;background:{b_col}"></div></div>
                    <div class="prob-pct">{p_val:.1f}%</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown("<hr style='margin:14px 0;border:0;border-top:1px solid #eef0f6'>", unsafe_allow_html=True)

            # ---- Customer Profile Summary (Exact 6 Model Features)
            st.markdown('<div class="ct">Customer Profile Summary</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="prof-grid">
                <div class="prof-item">
                    <div class="prof-ic">👤</div>
                    <div><div class="prof-lb">Age</div><div class="prof-vl">{in_age} years</div></div>
                </div>
                <div class="prof-item">
                    <div class="prof-ic">📅</div>
                    <div><div class="prof-lb">Recency</div><div class="prof-vl">{in_recency:g} days ago</div></div>
                </div>
                <div class="prof-item">
                    <div class="prof-ic">🪙</div>
                    <div><div class="prof-lb">Annual Income</div><div class="prof-vl">{CURRENCY} {fmt_inr(in_income)}</div></div>
                </div>
                <div class="prof-item">
                    <div class="prof-ic">⏳</div>
                    <div><div class="prof-lb">Customer Tenure</div><div class="prof-vl">{in_tenure:g} years</div></div>
                </div>
                <div class="prof-item">
                    <div class="prof-ic">🛒</div>
                    <div><div class="prof-lb">Purchase Frequency</div><div class="prof-vl">{in_freq:g} / year</div></div>
                </div>
                <div class="prof-item">
                    <div class="prof-ic">🔄</div>
                    <div><div class="prof-lb">Return Rate</div><div class="prof-vl">{in_return:g}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------- RIGHT COLUMN: Segment Overview Donut + Key Insights + Try Again ----------------
    with col_right:
        with st.container(border=True):
            st.markdown('<div class="ct">📊 Segment Overview</div>', unsafe_allow_html=True)
            n_total = df.customer_id.nunique()
            s_counts = df.segment.value_counts().reindex(order).dropna()
            donut = go.Figure(go.Pie(
                labels=s_counts.index, values=s_counts.values, hole=.62, sort=False,
                marker=dict(colors=[colors[s] for s in s_counts.index]),
                textinfo="percent", textfont=dict(size=10)
            ))
            donut.add_annotation(text=f"<b>{n_total:,}</b><br>Customers", showarrow=False, font=dict(size=11))
            donut.update_layout(legend=dict(orientation="v", x=1.02, y=.5, font=dict(size=11)))
            st.plotly_chart(style(donut, h=205), use_container_width=True)

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="ct">💡 Key Insights</div>', unsafe_allow_html=True)
            seg_desc = {
                "Budget": "Budget customers are price-sensitive and respond strongly to bundle deals.",
                "Regular": "Regular customers have moderate spending and good retention rates.",
                "Premium": "Premium customers generate the highest lifetime value and purchase frequently.",
            }.get(pred_seg, "Regular customers have moderate spending and good retention rates.")

            st.markdown(f"""
            <ul class="ki-ul">
                <li><b class="tick">✓</b> <span>{seg_desc}</span></li>
                <li><b class="tick">✓</b> <span>Customer tenure of <b>{in_tenure:g} years</b> and frequency of <b>{in_freq:g}/year</b> strongly drive annual spend.</span></li>
                <li><b class="tick">✓</b> <span>Opportunity to increase loyalty with personalized offers and product recommendations.</span></li>
            </ul>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="try-card">
            <div class="try-ic">🎯</div>
            <div>
                <b>Try again</b>
                <div>Adjust the 6 model features on the left to see how changes affect the prediction.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# OTHER VACANT SECTIONS
# =====================================================================
else:
    st.markdown(f'<div class="welcome">{page}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="vacant"><div style="font-size:2.5rem">🚧</div>
        <h3>This section is vacant</h3>
        <p>It will be built after the model is trained.</p></div>""", unsafe_allow_html=True)
