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
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# CONFIG  -  edit this block only
# =====================================================================
DEFAULT_DATA_PATH = "Data/Main_Clean.csv"
KMEANS_MODEL_PATH = "Models/kmeans_model.pkl"     # optional default dataset if no CSV is uploaded
SPENDING_MODEL_PATH = "Models/annual_spending_model.joblib"

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
/* High-contrast Regional Analysis radio labels ("Average per customer" / "Total") */
section.main div[data-testid="stRadio"] label,
section.main div[data-testid="stRadio"] label p,
section.main div[data-testid="stRadio"] div[role="radiogroup"] label * {
    color: #111a3b !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
/* High-contrast dark text for main-area expanders ("View Dataset Preview") & radios ("Average per customer" / "Total") */
[data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stExpander"] summary,
[data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stExpander"] summary:hover,
[data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stExpander"] details[open] > summary {
    background-color: #ffffff !important;
    color: #172033 !important;
    border-radius: 8px !important;
}
[data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stExpander"] summary *,
[data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stRadio"] label,
[data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stRadio"] label *,
[data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stMultiSelect"] label,
[data-testid="stAppViewContainer"] [data-testid="stMain"] [data-testid="stSelectbox"] label {
    color: #172033 !important;
    fill: #172033 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}
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

@st.cache_resource
def load_kmeans_bundle(path: str = KMEANS_MODEL_PATH):
    """Loads and caches the pre-trained K-Means dictionary bundle once."""
    if not os.path.exists(path):
        return None, f"Model file not found at `{path}`."
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or not {"pipeline", "cluster_names", "features"}.issubset(bundle.keys()):
            return None, "Invalid K-Means bundle: expected keys 'pipeline', 'cluster_names', and 'features'."
        return bundle, None
    except Exception as exc:
        return None, f"Failed to load K-Means model: {exc}"


def render_customer_segmentation():
    """Renders the Customer Segmentation page using the active dataset and Models/kmeans_model.pkl."""
    st.markdown('<div class="welcome">Customer <span style="color:#4f63e8">Segmentation</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tagline">Behavioral clustering powered by pre-trained K-Means &nbsp;•&nbsp; '
                'Segment telemetry &amp; profiling</div>', unsafe_allow_html=True)

    raw, source_name = get_active_raw()
    if raw is None or raw.empty:
        st.markdown("""<div class="vacant"><div style="font-size:2.5rem">📂</div>
            <h3>No dataset available</h3>
            <p>Ensure <b>Data/Main_Clean.csv</b> exists or upload a CSV/Excel file in the left sidebar.</p></div>""",
            unsafe_allow_html=True)
        return

    st.markdown(
        f'<div style="color:#334155;font-size:0.84rem;font-weight:600;margin:-4px 0 12px 0;">'
        f'📁 Active Dataset: <span style="color:#111a3b;">{source_name}</span> ({len(raw):,} total records)</div>',
        unsafe_allow_html=True,
    )

    bundle, model_err = load_kmeans_bundle()
    if model_err:
        st.error(f"❌ {model_err}")
        return

    pipeline = bundle["pipeline"]
    cluster_names = bundle["cluster_names"]
    required_features = list(bundle["features"])  # ['AnnualIncome', 'AnnualPurchaseAmount', 'PurchaseFrequency']

    # Validate that all 3 required K-Means features exist (never use CustomerSegment for predictions)
    missing_req = [col for col in required_features if col not in raw.columns]
    if missing_req:
        st.warning(
            f"⚠️ Cannot run K-Means segmentation because required feature(s) are missing: "
            f"**{', '.join(missing_req)}**. Expected features: `{required_features}`."
        )
        return

    # Preserve all rows; coerce numeric types and let the saved pipeline impute & scale nulls
    work_df = raw.copy()
    for feat in required_features:
        work_df[feat] = pd.to_numeric(work_df[feat], errors="coerce")

    if work_df[required_features].isna().all().all():
        st.warning("⚠️ None of the three required K-Means features contain valid numeric values.")
        return

    raw_clusters = pipeline.predict(work_df[required_features])
    work_df["PredictedSegment"] = pd.Series(raw_clusters, index=work_df.index).map(cluster_names).fillna("Unknown")

    # Apply Top Search Filter if typed
    q = st.session_state.get("search", "").strip().lower()
    if q:
        search_cols = [c for c in ["CustomerID", "Region", "Gender", "ProductCategory", "PredictedSegment"] if c in work_df.columns]
        if search_cols:
            work_df = work_df[work_df[search_cols].astype(str).apply(lambda s: s.str.lower().str.contains(q)).any(axis=1)]
            if work_df.empty:
                st.warning(f"No segmented customers match “{q}”.")
                return

    # ---------------- E. FILTERS (Region, Gender, ProductCategory) ----------------
    with st.container(border=True):
        st.markdown('<div class="ct">🎛️ Segment Filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        filtered_df = work_df.copy()

        with f1:
            if "Region" in filtered_df.columns:
                reg_opts = sorted(filtered_df["Region"].dropna().astype(str).unique().tolist())
                sel_reg = st.multiselect("Region", options=reg_opts, default=[], placeholder="All Regions", key="seg_reg")
                if sel_reg:
                    filtered_df = filtered_df[filtered_df["Region"].astype(str).isin(sel_reg)]
            else:
                st.selectbox("Region", options=["Column not in dataset"], disabled=True, key="seg_reg_na")

        with f2:
            if "Age" in filtered_df.columns:
                age_options = ["18–25 years", "26–35 years", "36–50 years", "50+ years"]
                sel_ages = st.multiselect(
                    "Age Group",
                    options=age_options,
                    default=[],
                    placeholder="All Age Groups",
                    key="seg_age_group",
                )
                if sel_ages:
                    age_num = pd.to_numeric(filtered_df["Age"], errors="coerce")
                    age_mask = pd.Series(False, index=filtered_df.index)
                    if "18–25 years" in sel_ages:
                        age_mask |= (age_num >= 18) & (age_num <= 25)
                    if "26–35 years" in sel_ages:
                        age_mask |= (age_num >= 26) & (age_num <= 35)
                    if "36–50 years" in sel_ages:
                        age_mask |= (age_num >= 36) & (age_num <= 50)
                    if "50+ years" in sel_ages:
                        age_mask |= (age_num > 50)
                    filtered_df = filtered_df[age_mask]
            else:
                st.selectbox("Age Group", options=["Column not in dataset"], disabled=True, key="seg_age_na")

        with f3:
            if "ProductCategory" in filtered_df.columns:
                cat_opts = sorted(filtered_df["ProductCategory"].dropna().astype(str).unique().tolist())
                sel_cat = st.multiselect("Product Category", options=cat_opts, default=[], placeholder="All Categories", key="seg_cat")
                if sel_cat:
                    filtered_df = filtered_df[filtered_df["ProductCategory"].astype(str).isin(sel_cat)]
            else:
                st.selectbox("Product Category", options=["Column not in dataset"], disabled=True, key="seg_cat_na")

    if filtered_df.empty:
        st.warning("⚠️ No customers match the selected filters. Clear one or more filters to view segments.")
        return

    st.write("")

    # ---------------- A. 4 KPI CARDS ----------------
    total_cust = len(filtered_df)
    seg_counts = filtered_df["PredictedSegment"].value_counts()
    num_segments = int(seg_counts.shape[0])
    spend_s = pd.to_numeric(filtered_df["AnnualPurchaseAmount"], errors="coerce")
    avg_spend = float(spend_s.mean()) if spend_s.notna().any() else 0.0
    top_seg = str(seg_counts.index[0])
    top_pct = (seg_counts.iloc[0] / total_cust) * 100

    c = st.columns(4)
    kpi(c[0], "👥", "Total Customers", f"{total_cust:,}", "Filtered active cohort", "#e8f1fd")
    kpi(c[1], "🧩", "Number of Segments", f"{num_segments}", "K-Means behavioral clusters", "#e3f6ec")
    kpi(c[2], CURRENCY, "Average Annual Spend", money(avg_spend), "Per customer annual mean", "#f0eafd")
    kpi(c[3], "🏆", "Top Segment", top_seg, f"{seg_counts.iloc[0]:,} customers ({top_pct:.1f}%)", "#fdf3df")
    st.write("")

    # ---------------- B. DONUT CHART & C. BAR CHART ----------------
    segment_order = ["Premium", "Regular", "Budget"]
    segment_colors = {
        "Budget": "#fb923c",   # Orange
        "Regular": "#22c277",  # Green
        "Premium": "#3b82f6",  # Blue
    }
    ordered_segs = [s for s in segment_order if s in seg_counts.index] + [
        s for s in seg_counts.index if s not in segment_order
    ]

    r_charts = st.columns([1, 1.2])

    with r_charts[0]:
        st.markdown('<div class="card"><h4>Segment Distribution</h4>', unsafe_allow_html=True)
        donut_vals = [int(seg_counts[s]) for s in ordered_segs]
        donut_colors = [segment_colors.get(s, "#94a3b8") for s in ordered_segs]

        legend_labels = [f"{s} ({seg_counts[s]:,})" for s in ordered_segs]
        fig_donut = go.Figure(go.Pie(
            labels=legend_labels,
            values=donut_vals,
            hole=0.56,
            sort=False,
            marker=dict(colors=donut_colors, line=dict(color="#ffffff", width=2)),
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont=dict(size=12, color="#ffffff"),
            customdata=ordered_segs,
            hovertemplate="<b>%{customdata}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total_cust:,}</b><br>Customers",
            showarrow=False,
            font=dict(size=13, color="#172033"),
        )
        fig_donut.update_layout(
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.08,
                font=dict(size=11, color="#172033"),
            )
        )
        fig_donut = style_chart(fig_donut, h=320)
        fig_donut.update_layout(margin=dict(l=16, r=16, t=24, b=42))
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r_charts[1]:
        st.markdown('<div class="card"><h4>Average Annual Purchase Amount by Segment</h4>', unsafe_allow_html=True)
        avg_by_seg = (
            filtered_df.groupby("PredictedSegment")["AnnualPurchaseAmount"]
            .mean()
            .reindex(ordered_segs)
            .fillna(0.0)
        )
        bar_colors = [segment_colors.get(s, "#94a3b8") for s in ordered_segs]

        fig_bar = go.Figure(go.Bar(
            x=ordered_segs,
            y=avg_by_seg.values,
            marker_color=bar_colors,
            text=[money(v) for v in avg_by_seg.values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=12, color="#172033"),
            hovertemplate="<b>%{x}</b><br>Avg Annual Spend: %{text}<extra></extra>",
        ))
        max_val = float(avg_by_seg.max()) if len(avg_by_seg) > 0 and avg_by_seg.max() > 0 else 1000.0
        fig_bar.update_yaxes(
            range=[0, max_val * 1.28],
            gridcolor="#e2e8f0",
            tickprefix=CURRENCY,
            tickformat="~s",
        )
        st.plotly_chart(style_chart(fig_bar, h=300), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ---------------- D. SEGMENT CHARACTERISTICS TABLE ----------------
    with st.container(border=True):
        st.markdown('<div class="ct">📋 Segment Characteristics</div>', unsafe_allow_html=True)
        metric_cols = {
            "Age": "Average Age",
            "AnnualIncome": "Average Annual Income",
            "PurchaseFrequency": "Average Purchase Frequency",
            "AnnualPurchaseAmount": "Average Annual Purchase Amount",
            "ReturnRate": "Average Return Rate",
            "DiscountUsage": "Average Discount Usage",
        }
        rows = []
        for seg_name in ordered_segs:
            sub = filtered_df[filtered_df["PredictedSegment"] == seg_name]
            row = {"Segment": seg_name, "Customer Count": f"{len(sub):,}"}
            for raw_col, label in metric_cols.items():
                if raw_col in sub.columns:
                    s_num = pd.to_numeric(sub[raw_col], errors="coerce")
                    if s_num.notna().any():
                        val = float(s_num.mean())
                        if raw_col in ("AnnualIncome", "AnnualPurchaseAmount"):
                            row[label] = money(val)
                        elif raw_col == "Age":
                            row[label] = f"{val:.1f} yrs"
                        elif raw_col == "PurchaseFrequency":
                            row[label] = f"{val:.1f}"
                        else:
                            row[label] = f"{val:.2f}"
                    else:
                        row[label] = "N/A"
                else:
                    row[label] = "N/A"
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.write("")

    # ---------------- F. SEGMENT INSIGHTS CARDS (Premium, Regular, Budget) ----------------
    st.markdown('<div class="sec" style="margin-bottom:8px;">Segment Insights</div>', unsafe_allow_html=True)
    ins_cols = st.columns(3)
    meta = {
        "Premium": ("👑", "#e8f1fd", "#3b82f6", "High-spending loyal customers driving top-line revenue."),
        "Regular": ("⭐", "#e3f6ec", "#22c277", "Consistent mid-tier buyers with strong repeat potential."),
        "Budget": ("🏷️", "#fdf3df", "#fb923c", "Value-conscious shoppers responsive to targeted promotions."),
    }

    for idx, seg_name in enumerate(["Premium", "Regular", "Budget"]):
        icon, bg, accent, desc = meta[seg_name]
        sub = filtered_df[filtered_df["PredictedSegment"] == seg_name]
        cnt = len(sub)
        inc_s = pd.to_numeric(sub["AnnualIncome"], errors="coerce")
        spd_s = pd.to_numeric(sub["AnnualPurchaseAmount"], errors="coerce")
        frq_s = pd.to_numeric(sub["PurchaseFrequency"], errors="coerce")

        inc_txt = money(inc_s.mean()) if cnt > 0 and inc_s.notna().any() else f"{CURRENCY} 0"
        spd_txt = money(spd_s.mean()) if cnt > 0 and spd_s.notna().any() else f"{CURRENCY} 0"
        frq_txt = f"{frq_s.mean():.1f} / yr" if cnt > 0 and frq_s.notna().any() else "0.0 / yr"

        with ins_cols[idx]:
            st.markdown(f"""
            <div class="card" style="border-top: 4px solid {accent};">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    <div style="width:36px;height:36px;border-radius:50%;background:{bg};display:flex;align-items:center;justify-content:center;font-size:1.1rem;">{icon}</div>
                    <div>
                        <div style="font-size:1rem;font-weight:800;color:#111a3b;">{seg_name} Segment</div>
                        <div style="font-size:.72rem;color:#64748b;">{desc}</div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;padding-top:10px;border-top:1px solid #eef0f6;font-size:.78rem;">
                    <div><span style="color:#64748b;">Customers:</span><br><b style="color:#111a3b;">{cnt:,}</b></div>
                    <div><span style="color:#64748b;">Avg Income:</span><br><b style="color:#111a3b;">{inc_txt}</b></div>
                    <div><span style="color:#64748b;">Avg Spending:</span><br><b style="color:#111a3b;">{spd_txt}</b></div>
                    <div><span style="color:#64748b;">Purchase Freq:</span><br><b style="color:#111a3b;">{frq_txt}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
SPENDING_FEATURES = [
    "PurchaseFrequency",
    "AnnualIncome",
    "Recency",
    "CustomerTenure",
    "ReturnRate",
    "Age",
]


@st.cache_resource
def load_spending_model(path: str = SPENDING_MODEL_PATH):
    """Loads and caches the pre-trained Annual Spending regression pipeline."""
    if not os.path.exists(path):
        return None, f"Regression model file not found at `{path}`."
    try:
        reg_model = joblib.load(path)
        if not hasattr(reg_model, "predict"):
            return None, f"Loaded object at `{path}` does not expose a `.predict()` method."
        return reg_model, None
    except Exception as exc:
        return None, f"Failed to load spending prediction model: {exc}"


def render_spending_prediction():
    """Renders the Spending Prediction page using the active dataset and Models/annual_spending_model.joblib."""
    st.markdown(
        '<div class="welcome">Spending <span style="color:#4f63e8">Prediction</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tagline">Forecast customer annual purchase amounts &nbsp;•&nbsp; '
        'HistGradientBoosting regression pipeline &nbsp;•&nbsp; Individual &amp; cohort telemetry</div>',
        unsafe_allow_html=True,
    )

    raw, source_name = get_active_raw()
    if raw is None or raw.empty:
        st.markdown(
            """<div class="vacant"><div style="font-size:2.5rem">📂</div>
            <h3>No dataset available</h3>
            <p>Ensure <b>Data/Main_Clean.csv</b> exists or upload a CSV/Excel file in the left sidebar.</p></div>""",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div style="color:#334155;font-size:0.84rem;font-weight:600;margin:-4px 0 12px 0;">'
        f'📁 Active Dataset: <span style="color:#111a3b;">{source_name}</span> ({len(raw):,} total records)</div>',
        unsafe_allow_html=True,
    )

    reg_model, model_err = load_spending_model()
    if model_err:
        st.error(f"❌ {model_err}")
        return

    expected_features = (
        list(reg_model.feature_names_in_)
        if hasattr(reg_model, "feature_names_in_")
        else list(SPENDING_FEATURES)
    )

    # ---------------- B. INDIVIDUAL CUSTOMER PREDICTION ----------------
    with st.container(border=True):
        st.markdown('<div class="ct">🧮 Individual Customer Prediction</div>', unsafe_allow_html=True)

        def _col_median(col_name: str, fallback: float) -> float:
            if col_name in raw.columns:
                s = pd.to_numeric(raw[col_name], errors="coerce").dropna()
                if not s.empty:
                    return float(s.median())
            return fallback

        with st.form("individual_spending_form"):
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                in_freq = st.number_input(
                    "Purchase Frequency (orders/yr)",
                    min_value=0.0,
                    max_value=500.0,
                    value=round(_col_median("PurchaseFrequency", 6.0), 1),
                    step=1.0,
                )
            with r1c2:
                in_income = st.number_input(
                    f"Annual Income ({CURRENCY})",
                    min_value=0.0,
                    max_value=50_000_000.0,
                    value=round(_col_median("AnnualIncome", 60000.0), 2),
                    step=1000.0,
                )
            with r1c3:
                in_recency = st.number_input(
                    "Recency (days since last purchase)",
                    min_value=0,
                    max_value=3650,
                    value=int(round(_col_median("Recency", 45.0))),
                    step=1,
                )

            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                in_tenure = st.number_input(
                    "Customer Tenure (years)",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(_col_median("CustomerTenure", 3.5), 2),
                    step=0.5,
                )
            with r2c2:
                in_return = st.number_input(
                    "Return Rate (0.00 – 1.00)",
                    min_value=0.0,
                    max_value=1.0,
                    value=round(min(max(_col_median("ReturnRate", 0.08), 0.0), 1.0), 4),
                    step=0.01,
                    format="%.4f",
                )
            with r2c3:
                in_age = st.number_input(
                    "Age (years)",
                    min_value=18,
                    max_value=110,
                    value=int(round(min(max(_col_median("Age", 35.0), 18), 110))),
                    step=1,
                )

            submitted = st.form_submit_button("Predict Annual Spending")

        if submitted:
            input_values = {
                "PurchaseFrequency": in_freq,
                "AnnualIncome": in_income,
                "Recency": in_recency,
                "CustomerTenure": in_tenure,
                "ReturnRate": in_return,
                "Age": in_age,
            }
            if any(v is None or pd.isna(v) for v in input_values.values()):
                st.error("❌ All 6 input fields are required and must contain valid numeric values.")
            elif in_income <= 0:
                st.error("❌ Annual Income must be greater than ₹ 0.")
            else:
                single_df = pd.DataFrame([input_values])[expected_features]
                pred_single = float(reg_model.predict(single_df)[0])
                pred_single = max(0.0, pred_single)
                st.markdown(
                    f"""
                    <div style="background:#eef2ff;border-left:4px solid #4f63e8;border-radius:10px;
                                padding:14px 18px;margin-top:8px;display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-size:0.78rem;color:#475569;font-weight:600;">ESTIMATED ANNUAL SPENDING</div>
                            <div style="font-size:1.65rem;font-weight:800;color:#111a3b;">{format_inr(pred_single)}</div>
                        </div>
                        <div style="font-size:0.78rem;color:#475569;text-align:right;">
                            Exact Value: <b>{CURRENCY} {pred_single:,.2f}</b><br>
                            Pipeline: HistGradientBoostingRegressor
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")

    # ---------------- DATASET FEATURE VALIDATION & BATCH PREDICTION ----------------
    missing_cols = [col for col in expected_features if col not in raw.columns]
    if missing_cols:
        st.warning(
            f"⚠️ Dataset-level prediction skipped because required model feature(s) are missing from "
            f"**{source_name}**: **{', '.join(missing_cols)}**. Expected features: `{expected_features}`."
        )
        return

    # Preserve all customer rows; coerce features to numeric and let pipeline SimpleImputer handle nulls
    work_df = raw.copy()
    feat_df = pd.DataFrame(index=work_df.index)
    for feat in expected_features:
        feat_df[feat] = pd.to_numeric(work_df[feat], errors="coerce")

    if feat_df.isna().all().all():
        st.warning("⚠️ None of the required model features contain valid numeric values in the active dataset.")
        return

    # Apply Top Search Filter if typed
    q = st.session_state.get("search", "").strip().lower()
    if q:
        search_cols = [c for c in ["CustomerID", "Region", "Gender", "ProductCategory"] if c in work_df.columns]
        if search_cols:
            mask = work_df[search_cols].astype(str).apply(lambda s: s.str.lower().str.contains(q)).any(axis=1)
            work_df = work_df.loc[mask].copy()
            feat_df = feat_df.loc[mask].copy()
            if work_df.empty:
                st.warning(f"No customers match “{q}”.")
                return

    preds = np.clip(reg_model.predict(feat_df[expected_features]), 0.0, None)
    work_df["PredictedAnnualSpending"] = preds

    has_actual = "AnnualPurchaseAmount" in work_df.columns
    if has_actual:
        work_df["_actual_spend"] = pd.to_numeric(work_df["AnnualPurchaseAmount"], errors="coerce")
        has_actual = bool(work_df["_actual_spend"].notna().any())

    # ---------------- A. PREDICTION OVERVIEW (4 KPI CARDS) ----------------
    st.markdown('<div class="sec" style="margin-bottom:8px;">Prediction Overview</div>', unsafe_allow_html=True)
    total_cust = len(work_df)
    avg_pred = float(np.mean(preds))

    c = st.columns(4)
    kpi(c[0], "👥", "Total Customers", f"{total_cust:,}", "Active cohort evaluated", "#e8f1fd")

    if has_actual:
        valid_comp = work_df.dropna(subset=["_actual_spend"])
        avg_actual = float(valid_comp["_actual_spend"].mean())
        avg_diff = float((valid_comp["PredictedAnnualSpending"] - valid_comp["_actual_spend"]).mean())
        mae_val = float((valid_comp["PredictedAnnualSpending"] - valid_comp["_actual_spend"]).abs().mean())
        diff_sign = "+" if avg_diff >= 0 else "-"

        kpi(c[1], "🧾", "Avg. Actual Spend", money(avg_actual), f"Across {len(valid_comp):,} actual records", "#e3f6ec")
        kpi(c[2], "📈", "Avg. Predicted Spend", money(avg_pred), f"Median {money(float(np.median(preds)))}", "#f0eafd")
        kpi(
            c[3],
            "⚖️",
            "Avg. Prediction Diff",
            f"{diff_sign}{CURRENCY} {abs(avg_diff):,.0f}",
            f"Mean Abs Error: {money(mae_val)}",
            "#fdf3df",
        )
    else:
        kpi(c[1], "🧾", "Avg. Actual Spend", "N/A", "Actual target column absent", "#e3f6ec")
        kpi(c[2], "📈", "Avg. Predicted Spend", money(avg_pred), f"Median {money(float(np.median(preds)))}", "#f0eafd")
        kpi(c[3], "⚖️", "Avg. Prediction Diff", "N/A", "Requires AnnualPurchaseAmount", "#fdf3df")

    st.write("")

    # ---------------- C. PREDICTION ANALYSIS (PLOTLY CHARTS + SUMMARY) ----------------
    st.markdown('<div class="sec" style="margin-bottom:8px;">Prediction Analysis</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="card"><h4>Predicted Annual Spending Distribution</h4>', unsafe_allow_html=True)
        hist_sample = (
            work_df["PredictedAnnualSpending"].sample(min(50_000, len(work_df)), random_state=42)
        )
        fig_hist = px.histogram(
            hist_sample,
            x="PredictedAnnualSpending",
            nbins=30,
            color_discrete_sequence=["#4f63e8"],
        )
        fig_hist.add_vline(
            x=avg_pred,
            line_dash="dash",
            line_color="#22c277",
            line_width=2,
            annotation_text=f"Mean: {money(avg_pred)}",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#172033"),
        )
        fig_hist.update_xaxes(title_text=f"Predicted Annual Spend ({CURRENCY})", tickprefix=CURRENCY, tickformat="~s")
        fig_hist.update_yaxes(title_text="Customer Count", gridcolor="#e2e8f0")
        fig_hist.update_traces(
            marker_line_width=1,
            marker_line_color="#3b82f6",
            hovertemplate="<b>Range:</b> %{x}<br><b>Customers:</b> %{y:,}<extra></extra>",
        )
        st.plotly_chart(style_chart(fig_hist, h=320), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch2:
        if has_actual:
            st.markdown('<div class="card"><h4>Actual vs. Predicted Annual Spending</h4>', unsafe_allow_html=True)
            comp_df = work_df.dropna(subset=["_actual_spend"])[["_actual_spend", "PredictedAnnualSpending"]].copy()
            comp_sample = comp_df.sample(min(4000, len(comp_df)), random_state=42)
            low_v = float(min(comp_sample["_actual_spend"].min(), comp_sample["PredictedAnnualSpending"].min()))
            high_v = float(max(comp_sample["_actual_spend"].max(), comp_sample["PredictedAnnualSpending"].max()))

            fig_scat = go.Figure()
            fig_scat.add_trace(
                go.Scattergl(
                    x=comp_sample["_actual_spend"],
                    y=comp_sample["PredictedAnnualSpending"],
                    mode="markers",
                    name="Customers",
                    marker=dict(color="#3b82f6", size=5, opacity=0.45),
                    hovertemplate=f"<b>Actual:</b> {CURRENCY}%{{x:,.0f}}<br><b>Predicted:</b> {CURRENCY}%{{y:,.0f}}<extra></extra>",
                )
            )
            fig_scat.add_trace(
                go.Scatter(
                    x=[low_v, high_v],
                    y=[low_v, high_v],
                    mode="lines",
                    name="Ideal Fit (y = x)",
                    line=dict(color="#fb923c", width=2, dash="dash"),
                )
            )
            fig_scat.update_xaxes(title_text=f"Actual Annual Spend ({CURRENCY})", tickprefix=CURRENCY, tickformat="~s", gridcolor="#e2e8f0")
            fig_scat.update_yaxes(title_text=f"Predicted Annual Spend ({CURRENCY})", tickprefix=CURRENCY, tickformat="~s", gridcolor="#e2e8f0")
            fig_scat.update_layout(legend=dict(orientation="h", x=0, y=1.08))
            st.plotly_chart(style_chart(fig_scat, h=320), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><h4>Predicted Spending Brackets</h4>', unsafe_allow_html=True)
            p_bins = pd.qcut(work_df["PredictedAnnualSpending"], q=4, duplicates="drop")
            b_labels = [f"{CURRENCY}{int(iv.left):,}–{int(iv.right):,}" for iv in p_bins.cat.categories]
            b_counts = p_bins.cat.rename_categories(b_labels).value_counts().sort_index()
            fig_brk = go.Figure(
                go.Bar(
                    x=b_counts.index.astype(str),
                    y=b_counts.values,
                    marker_color=PALETTE[: len(b_counts)],
                    text=[f"{v:,}" for v in b_counts.values],
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=11, color="#172033"),
                )
            )
            fig_brk.update_yaxes(range=[0, b_counts.max() * 1.25], gridcolor="#e2e8f0", title_text="Customers")
            st.plotly_chart(style_chart(fig_brk, h=320), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Summary Strip
    p25, p50, p75 = np.percentile(preds, [25, 50, 75])
    if has_actual:
        valid_r2 = work_df.dropna(subset=["_actual_spend"])
        ss_res = float(((valid_r2["_actual_spend"] - valid_r2["PredictedAnnualSpending"]) ** 2).sum())
        ss_tot = float(((valid_r2["_actual_spend"] - valid_r2["_actual_spend"].mean()) ** 2).sum())
        r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        summary_html = (
            f"<b>Prediction Summary:</b> Across <b>{total_cust:,}</b> customers, predicted annual spending spans an "
            f"interquartile range of <b>{money(p25)}</b> (25th) to <b>{money(p75)}</b> (75th) with a median of "
            f"<b>{money(p50)}</b>. Against observed `AnnualPurchaseAmount` values, the model achieves an "
            f"<b>R² of {r2_val:.4f}</b> and a mean absolute error of <b>{money(mae_val)}</b>."
        )
    else:
        summary_html = (
            f"<b>Prediction Summary:</b> Across <b>{total_cust:,}</b> customers, predicted annual spending spans an "
            f"interquartile range of <b>{money(p25)}</b> (25th) to <b>{money(p75)}</b> (75th) with a median of "
            f"<b>{money(p50)}</b>."
        )
    st.markdown(
        f'<div class="card" style="font-size:0.84rem;color:#334155;">{summary_html}</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # ---------------- D. CUSTOMER PREDICTION RESULTS TABLE ----------------
    with st.container(border=True):
        st.markdown('<div class="ct">📋 Customer Prediction Results</div>', unsafe_allow_html=True)
        t_col1, t_col2 = st.columns([1.2, 2.8])
        with t_col1:
            row_limit = st.selectbox(
                "Rows to display",
                options=[25, 100, 500, 1000],
                index=1,
                key="pred_table_limit",
            )
        with t_col2:
            st.markdown(
                f'<div style="color:#475569;font-size:0.8rem;padding-top:30px;">'
                f'Showing top <b>{min(row_limit, total_cust):,}</b> of <b>{total_cust:,}</b> customers '
                f'(all rows preserved, including imputed feature nulls).</div>',
                unsafe_allow_html=True,
            )

        display_cols = []
        if "CustomerID" in work_df.columns:
            display_cols.append("CustomerID")
        for feat in expected_features:
            if feat in work_df.columns:
                display_cols.append(feat)

        table_df = work_df[display_cols].head(row_limit).copy()
        if has_actual:
            table_df["ActualAnnualPurchaseAmount"] = work_df["_actual_spend"].head(row_limit).round(2)
        table_df["PredictedAnnualSpending"] = work_df["PredictedAnnualSpending"].head(row_limit).round(2)
        if has_actual:
            table_df["Difference (Pred - Actual)"] = (
                table_df["PredictedAnnualSpending"] - table_df["ActualAnnualPurchaseAmount"]
            ).round(2)

        col_cfg = {
            "AnnualIncome": st.column_config.NumberColumn(f"Annual Income ({CURRENCY})", format=f"{CURRENCY} %.2f"),
            "ActualAnnualPurchaseAmount": st.column_config.NumberColumn(f"Actual Spend ({CURRENCY})", format=f"{CURRENCY} %.2f"),
            "PredictedAnnualSpending": st.column_config.NumberColumn(f"Predicted Spend ({CURRENCY})", format=f"{CURRENCY} %.2f"),
            "Difference (Pred - Actual)": st.column_config.NumberColumn(f"Difference ({CURRENCY})", format=f"{CURRENCY} %.2f"),
        }
        st.dataframe(table_df, hide_index=True, use_container_width=True, column_config=col_cfg)            
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
# ---------------- DATA OVERVIEW (BETWEEN KPI CARDS & CHARTS) ----------------
    st.markdown('<div class="sec" style="margin-bottom:6px;">Data Overview</div>', unsafe_allow_html=True)
    with st.expander("View Dataset Preview", expanded=False):
        st.dataframe(raw.head(5), hide_index=True, use_container_width=True)
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

# # ---------------- ROW 5: EXPANDABLE DATA OVERVIEW ----------------
#     st.write("")
#     st.markdown('<div class="sec">Data Overview</div>', unsafe_allow_html=True)
#     with st.expander("📋 Data Overview (First 5 Rows & Column Schema)", expanded=False):
#         preview_df = raw.head(5).copy()

#         # Derive PredictedSegment on the 5 preview rows using the cached K-Means bundle
#         bundle, model_err = load_kmeans_bundle()
#         if model_err:
#             st.warning(f"⚠️ Could not generate `PredictedSegment` preview: {model_err}")
#         else:
#             req_feats = list(bundle["features"])  # ['AnnualIncome', 'AnnualPurchaseAmount', 'PurchaseFrequency']
#             missing_feats = [col for col in req_feats if col not in preview_df.columns]
#             if missing_feats:
#                 st.warning(
#                     f"⚠️ `PredictedSegment` skipped — missing required model feature(s): "
#                     f"**{', '.join(missing_feats)}**."
#                 )
#             else:
#                 feat_matrix = preview_df[req_feats].apply(pd.to_numeric, errors="coerce")
#                 if feat_matrix.isna().all().all():
#                     st.warning("⚠️ `PredictedSegment` skipped — required feature columns contain no numeric values.")
#                 else:
#                     cluster_ids = bundle["pipeline"].predict(feat_matrix)
#                     preview_df["PredictedSegment"] = (
#                         pd.Series(cluster_ids, index=preview_df.index)
#                         .map(bundle["cluster_names"])
#                         .fillna("Unknown")
#                     )

#         st.caption(
#             f"**Columns ({len(preview_df.columns)}):** "
#             + ", ".join([f"`{c}`" for c in preview_df.columns.astype(str)])
#         )
#         st.dataframe(preview_df, hide_index=True, use_container_width=True)
# # =====================================================================
# PAGE 2: CUSTOMER SEGMENTATION
# =====================================================================
elif page == "Customer Segmentation":
    render_customer_segmentation()

elif page == "Spending Prediction":
    render_spending_prediction()


# =====================================================================
# OTHER VACANT SECTIONS
# =====================================================================
else:
    st.markdown(f'<div class="welcome">{page}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="vacant"><div style="font-size:2.5rem">🚧</div>
        <h3>This section is vacant</h3>
        <p>It will be built when requested.</p></div>""", unsafe_allow_html=True)