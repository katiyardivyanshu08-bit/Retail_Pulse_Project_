"""
RetailPulse - Customer Analytics Dashboard (Streamlit)

Single-file setup with:
- Sidebar Navigation & CSV/Excel Uploader
- Overview Page (KPIs, Age/Spend/Category Distributions, Regional Analysis, Quick Insights)
- All other sections kept vacant until requested

Run:  streamlit run app.py
"""
import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# CONFIG  -  edit this block only
# =====================================================================
DEFAULT_DATA_PATH = "Data/Main_Clean.csv"      # optional default dataset if no CSV is uploaded

# Map the dashboard fields to YOUR column names.
# Leave a value as None to auto-detect from the CSV column names.
COLUMNS = {
    "spend": None,        # annual spend / purchase amount   (REQUIRED)
    "id": None,           # customer id
    "purchases": None,    # PurchaseFrequency / number of purchases
    "category": None,     # product category
    "age": None,          # Age
    "discount": None,     # discount usage
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
PALETTE = ["#3b82f6", "#22c277", "#fb923c", "#8b5cf6", "#22b8cf", "#f43f5e", "#94a3b8"]
CAT_COLORS = ["#3b82f6", "#22c277", "#fb923c", "#8b5cf6", "#22b8cf", "#94a3b8"]

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
/* Hide only MainMenu and footer; leave Streamlit's native header and sidebar toggle untouched */
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {    
    background: transparent !important;
}
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] {
    pointer-events: auto !important;
    color: #172033 !important;
}
[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapsedControl"] * {
    color: #172033 !important;
    fill: #172033 !important;
}

.block-container {padding-top: 2.2rem; max-width: 1260px;}
.stApp {background: #f8f9fc;}
section[data-testid="stSidebar"] {background: #131c3a;}
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
.card h4 {font-size: .9rem; font-weight: 700; color: #172033; margin: 0 0 6px 0;}
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
.ct {font-size: .9rem; font-weight: 700; color: #172033; margin-bottom: 4px;}
div[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border: 1px solid #eef0f6 !important; border-radius: 12px;}
div.stButton > button {background: #5856eb; color: #fff; border: 0; border-radius: 8px; font-weight: 600;}
div.stButton > button:hover {background: #4341d6; color: #fff;}

/* Sidebar Upload Dropzone Contrast */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
    background: #1e294b; border: 1px dashed #475569; border-radius: 10px; padding: 8px 10px;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {
    background: #4f63e8; color: #fff; border: none;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data loading & helpers
@st.cache_data
def load_file_from_path(path: str, mtime: float) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def get_active_raw():
    if "uploaded_bytes" in st.session_state:
        name = st.session_state["uploaded_name"]
        buf = io.BytesIO(st.session_state["uploaded_bytes"])
        if name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(buf), name
        return pd.read_csv(buf), name
    if os.path.exists(DEFAULT_DATA_PATH):
        return load_file_from_path(DEFAULT_DATA_PATH, os.path.getmtime(DEFAULT_DATA_PATH)), DEFAULT_DATA_PATH
    return None, None


def guess(cols, keywords, exclude=()):
    for kw in keywords:
        for c in cols:
            if c not in exclude and kw in c.lower().replace(" ", "_"):
                return c
    return None


def resolve_columns(raw: pd.DataFrame) -> dict:
    cols, m = raw.columns.tolist(), dict(COLUMNS)
    m["spend"] = m["spend"] or guess(cols, ["annual_spend", "annualspend", "spend", "purchase_amount",
                                            "amount", "revenue", "sales", "total"])
    m["id"] = m["id"] or guess(cols, ["customer_id", "customerid", "cust", "id", "user"], [m["spend"]])
    m["purchases"] = m["purchases"] or guess(cols, ["purchasefrequency", "frequency", "num", "count",
                                                    "orders", "transactions", "purchases"],
                                             [m["spend"], m["id"]])
    m["category"] = m["category"] or guess(cols, ["categ", "product", "item"])
    m["age"] = m["age"] or guess(cols, ["age"])
    m["discount"] = m["discount"] or guess(cols, ["discount", "promo", "coupon"])
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
    if m["region"]:
        d["region"] = raw[m["region"]].astype(str)
    d = d.dropna(subset=["annual_spend"]).reset_index(drop=True)
    return d


def money(x):
    return f"{CURRENCY} {x:,.0f}"

def format_inr(x):
    """Format numeric value in Indian numbering style (lakhs/crores) with ₹ symbol."""
    if pd.isna(x):
        return "–"
    neg = x < 0
    s = str(int(round(abs(x))))
    if len(s) <= 3:
        res = s
    else:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        res = ",".join(reversed(parts)) + "," + last3
    return f"{'-' if neg else ''}{CURRENCY} {res}"
def kpi(col, icon, label, value, sub, bg):
    col.markdown(f"""<div class="kpi" style="background:{bg}"><div class="ic">{icon}</div>
        <div><div class="lb">{label}</div><div class="vl">{value}</div>
        <div class="dl">{sub}</div></div></div>""", unsafe_allow_html=True)


def style_chart(fig, h=260):
    """Applies high-contrast dark text styling across all Plotly charts."""
    fig.update_layout(
        height=h,
        margin=dict(l=12, r=16, t=16, b=12),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(size=12, color="#172033"),
        legend=dict(
            font=dict(size=12, color="#172033"),
            title_font=dict(size=12, color="#172033"),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            font=dict(size=12, color="#172033"),
        ),
    )
    fig.update_xaxes(
        tickfont=dict(size=12, color="#334155"),
        title_font=dict(size=12, color="#172033"),
    )
    fig.update_yaxes(
        tickfont=dict(size=12, color="#334155"),
        title_font=dict(size=12, color="#172033"),
    )
    return fig


# ------------------------------------------------------------------ sidebar / nav
if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    st.markdown('<div class="brand">📊 RetailPulse</div>', unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="page", label_visibility="collapsed",
             format_func=lambda p: f"{ICONS[p]}  {p}")
    st.markdown("---")
    sidebar_file = st.file_uploader("Upload Dataset (CSV / Excel)", type=["csv", "xlsx", "xls"], key="side_csv")
    st.markdown('<div class="side-foot">Smarter Insights • Better Decisions</div>',
                unsafe_allow_html=True)

if sidebar_file is not None:
    st.session_state["uploaded_bytes"] = sidebar_file.getvalue()
    st.session_state["uploaded_name"] = sidebar_file.name

page = st.session_state.page

# ------------------------------------------------------------------ top bar + search
top = st.columns([3, 1])
top[0].text_input("search", placeholder="🔍  Search customer, category, region...",
                  label_visibility="collapsed", key="search")
top[1].markdown("<div style='text-align:right;color:#111a3b;font-size:.8rem'>☀️ &nbsp; 👤 <b>RetailPulse</b>"
                "<br><span style='color:#64748b;font-size:.7rem'>Customer Analytics</span></div>",
                unsafe_allow_html=True)


# =====================================================================
# PAGE 1: OVERVIEW
# =====================================================================
if page == "Overview":
    def go_analysis():
        st.session_state.page = "Data Analysis"

    st.markdown('<div class="welcome">Welcome to <span style="color:#4f63e8">RetailPulse</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tagline">Understand your customers &nbsp;•&nbsp; Explore purchase '
                'behavior &nbsp;•&nbsp; Predict future spending</div>', unsafe_allow_html=True)

    raw, _ = get_active_raw()

    if raw is None:
        st.markdown("""<div class="vacant"><div style="font-size:2.5rem">📂</div>
            <h3>No CSV file uploaded yet</h3>
            <p>Use the <b>Upload Dataset (CSV / Excel)</b> uploader in the left sidebar to load your data and view live KPIs, distributions, and insights.</p></div>""",
            unsafe_allow_html=True)
        st.stop()

    m = resolve_columns(raw)
    if not m["spend"]:
        st.error("Could not auto-detect the spend/purchase amount column in your CSV. "
                 f"Set `COLUMNS['spend']` at the top of `app.py`. Columns found: {', '.join(raw.columns.astype(str))}")
        st.stop()

    df = prepare(raw, m)

    # Apply Top Search Filter if typed
    q = st.session_state.get("search", "").strip().lower()
    if q:
        tcols = [c for c in ["customer_id", "category", "region"] if c in df]
        if tcols:
            df = df[df[tcols].astype(str).apply(lambda s: s.str.lower().str.contains(q)).any(axis=1)]
            st.caption(f"🔍 {len(df):,} customers match “{q}”")
            if df.empty:
                st.stop()

    thr = df.annual_spend.quantile(1 - VALUABLE_TOP_PCT / 100)
    df["valuable"] = df.annual_spend >= thr

    # ---------------- ROW 1: 4 KPI CARDS ----------------
    n = df.customer_id.nunique()
    c = st.columns(4)
    kpi(c[0], "👥", "Total Customers", f"{n:,}", f"{len(df):,} records", "#e8f1fd")
    if "purchases" in df and df.purchases.notna().any():
        kpi(c[1], "🛒", "Total Purchases", f"{df.purchases.sum():,.0f}",
            f"{df.purchases.mean():.1f} avg per customer", "#e3f6ec")
    else:
        kpi(c[1], "🛒", "Total Purchases", "–", "No frequency/purchases column", "#e3f6ec")
    kpi(c[2], CURRENCY, "Avg. Annual Spend", money(df.annual_spend.mean()),
        f"Median {money(df.annual_spend.median())}", "#f0eafd")
    rev_series = (
        pd.to_numeric(raw.loc[df.index, "AnnualPurchaseAmount"], errors="coerce")
        if "AnnualPurchaseAmount" in raw.columns
        else df["annual_spend"]
    )
    if rev_series.notna().any():
        kpi(c[3], "💰", "Total Annual Revenue", format_inr(rev_series.sum()),
            f"Across {int(rev_series.notna().sum()):,} valid orders", "#fdf3df")
    else:
        kpi(c[3], "💰", "Total Annual Revenue", "–",
            "No valid AnnualPurchaseAmount values", "#fdf3df")
    st.write("")

    # ---------------- ROW 2: 3 CHARTS ----------------
    r1 = st.columns([1, 1.25, 1])

    # Chart 1 (Donut)
    with r1[0]:
        if "age" in df and df.age.notna().any():
            st.markdown('<div class="card"><h4>Customer Age Distribution</h4>', unsafe_allow_html=True)
            age_groups = pd.cut(
                df.age,
                bins=[0, 25, 35, 50, 150],
                labels=["18–25 yrs", "26–35 yrs", "36–50 yrs", "50+ yrs"]
            )
            counts = age_groups.value_counts().reindex(["18–25 yrs", "26–35 yrs", "36–50 yrs", "50+ yrs"]).dropna()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=.6, sort=False,
                marker=dict(colors=PALETTE[:len(counts)]),
                textinfo="percent",
                textfont=dict(size=12, color="#ffffff"),
            ))
            fig.add_annotation(
                text=f"<b>{n:,}</b><br>Customers",
                showarrow=False,
                font=dict(size=13, color="#172033"),
            )
            fig.update_layout(legend=dict(orientation="v", x=1, y=.5, font=dict(size=12, color="#172033")))
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        elif "region" in df:
            st.markdown('<div class="card"><h4>Customers by Region</h4>', unsafe_allow_html=True)
            counts = df.region.value_counts().head(5)
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=.6, sort=False,
                marker=dict(colors=PALETTE[:len(counts)]),
                textinfo="percent",
                textfont=dict(size=12, color="#ffffff"),
            ))
            fig.add_annotation(
                text=f"<b>{n:,}</b><br>Customers",
                showarrow=False,
                font=dict(size=13, color="#172033"),
            )
            fig.update_layout(legend=dict(orientation="v", x=1, y=.5, font=dict(size=12, color="#172033")))
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><h4>Customer Value Split</h4>', unsafe_allow_html=True)
            v_counts = pd.Series({
                f"Top {VALUABLE_TOP_PCT}%": int(df.valuable.sum()),
                "Standard": int((~df.valuable).sum())
            })
            fig = go.Figure(go.Pie(
                labels=v_counts.index, values=v_counts.values, hole=.6, sort=False,
                marker=dict(colors=["#fb923c", "#3b82f6"]),
                textinfo="percent",
                textfont=dict(size=12, color="#ffffff"),
            ))
            fig.add_annotation(
                text=f"<b>{n:,}</b><br>Customers",
                showarrow=False,
                font=dict(size=13, color="#172033"),
            )
            fig.update_layout(legend=dict(orientation="v", x=1, y=.5, font=dict(size=12, color="#172033")))
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Chart 2 (Vertical Bar)
    with r1[1]:
        st.markdown('<div class="card"><h4>Annual Purchase Amount Distribution</h4>', unsafe_allow_html=True)
        q_bins = pd.qcut(df.annual_spend, q=4, duplicates="drop")
        bracket_labels = [f"{CURRENCY}{int(iv.left):,}–{int(iv.right):,}" for iv in q_bins.cat.categories]
        q_named = q_bins.cat.rename_categories(bracket_labels)
        avg_bracket = df.groupby(q_named, observed=False).annual_spend.mean().dropna()

        fig = go.Figure(go.Bar(
            x=avg_bracket.index.astype(str),
            y=avg_bracket.values,
            marker_color=PALETTE[:len(avg_bracket)],
            text=[money(v) for v in avg_bracket.values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=11, color="#172033"),
        ))
        fig.update_yaxes(
            range=[0, avg_bracket.max() * 1.28],
            gridcolor="#e2e8f0",
            tickfont=dict(size=11, color="#334155"),
        )
        fig.update_xaxes(tickfont=dict(size=11, color="#172033"))
        st.plotly_chart(style_chart(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Chart 3 (Horizontal Bar)
    with r1[2]:
        if "category" in df:
            st.markdown('<div class="card"><h4>Top Product Categories</h4>', unsafe_allow_html=True)
            share = df.category.value_counts(normalize=True) * 100
            if len(share) > 6:
                share = pd.concat([share.head(5), pd.Series({"Others": share.iloc[5:].sum()})])
            fig = go.Figure(go.Bar(
                y=share.index,
                x=share.values,
                orientation="h",
                text=[f"{p:.1f}%" for p in share.values],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=12, color="#172033"),
                marker_color=CAT_COLORS[:len(share)],
            ))
            fig.update_yaxes(autorange="reversed", tickfont=dict(size=12, color="#172033"))
            fig.update_xaxes(visible=False, range=[0, share.max() * 1.38])
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><h4>Key Behavioral Averages</h4>', unsafe_allow_html=True)
            metrics = {}
            if "purchases" in df and df.purchases.notna().any():
                metrics["Avg Frequency"] = df.purchases.mean()
            if "tenure" in df and df.tenure.notna().any():
                metrics["Avg Tenure (yrs)"] = df.tenure.mean()
            if "recency" in df and df.recency.notna().any():
                metrics["Avg Recency (d)"] = df.recency.mean()
            if "age" in df and df.age.notna().any():
                metrics["Avg Age (yrs)"] = df.age.mean()
            if metrics:
                s_met = pd.Series(metrics)
                fig = go.Figure(go.Bar(
                    y=s_met.index,
                    x=s_met.values,
                    orientation="h",
                    text=[f"{v:.1f}" for v in s_met.values],
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=12, color="#172033"),
                    marker_color=CAT_COLORS[:len(s_met)],
                ))
                fig.update_yaxes(autorange="reversed", tickfont=dict(size=12, color="#172033"))
                fig.update_xaxes(visible=False, range=[0, s_met.max() * 1.38])
                st.plotly_chart(style_chart(fig), use_container_width=True)
            else:
                st.info("Upload a CSV with Category, Age, or Frequency columns for breakdown.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- ROW 3: REGIONAL ANALYSIS ----------------
    if "region" in df:
        st.write("")
        h1, h2 = st.columns([3, 1.4])
        h1.markdown('<div class="sec">Regional Analysis</div>'
                    '<div class="tagline" style="margin-bottom:0">Annual purchase amount across regions</div>',
                    unsafe_allow_html=True)
        mode = h2.radio("Metric", ["Average per customer", "Total"], horizontal=True,
                        key="region_mode", label_visibility="collapsed")
        agg = "mean" if mode.startswith("Average") else "sum"
        st.write("")

        region_rank = df.groupby("region").annual_spend.sum().sort_values(ascending=False)
        keep = region_rank.head(8).index.tolist()
        dr = df[df.region.isin(keep)]

        def tidy(fig, h=320):
            fig = style_chart(fig, h)
            fig.update_layout(
                margin=dict(l=12, r=16, t=44, b=14),
                bargap=.25,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, title_text="",
                    font=dict(size=12, color="#172033"),
                ),
            )
            fig.update_yaxes(
                tickprefix=CURRENCY, tickformat="~s",
                gridcolor="#e2e8f0", title=None,
                tickfont=dict(size=12, color="#334155"),
            )
            fig.update_xaxes(
                title=None,
                tickfont=dict(size=12, color="#172033"),
            )
            return fig

        g1, g2 = st.columns(2)
        with g1:
            with st.container(border=True):
                st.markdown('<div class="ct">Annual Purchase Amount by Region</div>', unsafe_allow_html=True)
                reg_spend = dr.groupby("region").annual_spend.agg(agg).reindex(keep).reset_index()
                fig = px.bar(reg_spend, x="region", y="annual_spend", color="region",
                             text=reg_spend.annual_spend.map(money), color_discrete_sequence=PALETTE)
                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=11, color="#172033"),
                )
                fig.update_yaxes(range=[0, reg_spend.annual_spend.max() * 1.24])
                fig.update_layout(showlegend=False)
                st.plotly_chart(tidy(fig), use_container_width=True)

        with g2:
            with st.container(border=True):
                st.markdown('<div class="ct">Top Product Category by Purchase Amount in Each Region</div>', unsafe_allow_html=True)
                if "category" not in df:
                    st.info("No category column found in the uploaded CSV.")
                else:
                    reg_cat = dr.groupby(["region", "category"]).annual_spend.agg(agg).reset_index()
                    best = reg_cat.loc[reg_cat.groupby("region").annual_spend.idxmax()]
                    best = best.set_index("region").loc[keep].reset_index()
                    best["label"] = best.region + "<br>" + best.category
                    fig = px.bar(best, x="label", y="annual_spend", color="category",
                                 text=best.annual_spend.map(money),
                                 category_orders={"label": best.label.tolist()},
                                 color_discrete_sequence=CAT_COLORS)
                    fig.update_traces(
                        textposition="outside",
                        cliponaxis=False,
                        textfont=dict(size=11, color="#172033"),
                    )
                    fig.update_yaxes(range=[0, best.annual_spend.max() * 1.24])
                    st.plotly_chart(tidy(fig), use_container_width=True)

    # ---------------- ROW 4: QUICK INSIGHTS + CTA ----------------
    st.write("")
    items = []
    top_q = df[df.valuable].annual_spend.mean()
    std_q = df[~df.valuable].annual_spend.mean()
    if std_q and std_q > 0:
        items.append(("📈", "#dcfce7", f"Top {VALUABLE_TOP_PCT}% customers drive outsized revenue",
                      f"Valuable customers spend {top_q / std_q:.1f}x more on average than standard customers."))
    if "category" in df:
        sh = df.category.value_counts(normalize=True)
        items.append(("👥", "#ede9fe", f"{sh.index[0]} is the top category",
                      f"It accounts for {sh.iloc[0]:.1%} of total records in your CSV."))
    if "age" in df and df.age.notna().any():
        ab = pd.cut(df.age, [0, 24, 40, 55, 200], labels=["under 25", "25–40", "41–55", "over 55"])
        items.append(("🕒", "#ffedd5", f"Most customers are {ab.value_counts().idxmax()} years old",
                      "This is the largest age group in your uploaded dataset."))
    if "discount_usage" in df and df.valuable.any() and (~df.valuable).any():
        v, o = df[df.valuable].discount_usage.mean(), df[~df.valuable].discount_usage.mean()
        if o and o > 0:
            items.append(("⭐", "#dbeafe",
                          f"Valuable customers show {'higher' if v >= o else 'lower'} discount usage",
                          f"They use discounts {v / o:.1f}x as much as other customers."))
    if "income" in df and df.annual_income.notna().any():
        items.append(("💰", "#e0f2fe", f"Average annual customer income is {money(df.annual_income.mean())}",
                      "Income and purchase frequency strongly correlate with annual spend."))

    r2 = st.columns([1.15, 1])
    with r2[0]:
        html = '<div class="card"><h4>💡 Quick Insights</h4>'
        for ic, bg, t, d in items[:4]:
            html += (f'<div class="insight"><div class="ii" style="background:{bg}">{ic}</div>'
                     f'<div><b>{t}</b><div>{d}</div></div></div>')
        st.markdown(html + "</div>", unsafe_allow_html=True)
    with r2[1]:
        st.markdown("""<div class="cta"><h2>Turn Data into<br>Business Growth</h2>
            <p>Explore customer behavior, train unsupervised segments and predict future spending
            with our ML-powered insights.</p></div>""", unsafe_allow_html=True)
        st.button("Go to Detailed Analysis →", on_click=go_analysis)


# =====================================================================
# OTHER VACANT SECTIONS
# =====================================================================
else:
    st.markdown(f'<div class="welcome">{page}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="vacant"><div style="font-size:2.5rem">🚧</div>
        <h3>This section is vacant</h3>
        <p>It will be built when requested.</p></div>""", unsafe_allow_html=True)