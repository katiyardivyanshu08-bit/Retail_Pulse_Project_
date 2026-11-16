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


def _resolve_recency_column(work_df: pd.DataFrame) -> pd.DataFrame:
    """Uses existing Recency column if present; otherwise derives it from LastPurchaseDate using the dataset reference date (2025-12-31)."""
    if "Recency" not in work_df.columns and "LastPurchaseDate" in work_df.columns:
        parsed_dates = pd.to_datetime(work_df["LastPurchaseDate"], errors="coerce")
        work_df["Recency"] = (pd.Timestamp("2025-12-31") - parsed_dates).dt.days
    return work_df


def _assign_spending_potential_groups(pred_series: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Derives mutually exclusive Spending Potential groups using 25th and 75th percentiles
    computed on the full active dataset BEFORE filtering:
      - High Spending Potential:   pred > p75
      - Medium Spending Potential: p25 <= pred <= p75
      - Low Spending Potential:    pred < p25
    """
    valid = pred_series.dropna()
    if valid.empty:
        return pd.Series("Unknown", index=pred_series.index), np.nan, np.nan

    p25 = float(np.percentile(valid, 25))
    p75 = float(np.percentile(valid, 75))

    groups = pd.Series("Medium Spending Potential", index=pred_series.index, dtype="object")
    groups[pred_series.isna()] = "Unknown"
    groups[pred_series > p75] = "High Spending Potential"
    groups[pred_series < p25] = "Low Spending Potential"
    return groups, p25, p75


def _apply_spending_filters(work_df: pd.DataFrame) -> pd.DataFrame:
    """Renders Region, Age Group, and Product Category multiselect filters matching Customer Segmentation."""
    with st.container(border=True):
        st.markdown('<div class="ct">🎛️ Cohort Filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        filtered = work_df.copy()

        with f1:
            if "Region" in filtered.columns:
                reg_opts = sorted(filtered["Region"].dropna().astype(str).unique().tolist())
                sel_reg = st.multiselect(
                    "Region",
                    options=reg_opts,
                    default=[],
                    placeholder="All Regions",
                    key="sp_filter_region",
                )
                if sel_reg:
                    filtered = filtered[filtered["Region"].astype(str).isin(sel_reg)]
            else:
                st.selectbox("Region", options=["Column not in dataset"], disabled=True, key="sp_reg_na")

        with f2:
            if "Age" in filtered.columns:
                age_options = ["18–25 years", "26–35 years", "36–50 years", "50+ years"]
                sel_ages = st.multiselect(
                    "Age Group",
                    options=age_options,
                    default=[],
                    placeholder="All Age Groups",
                    key="sp_filter_age_group",
                )
                if sel_ages:
                    age_num = pd.to_numeric(filtered["Age"], errors="coerce")
                    age_mask = pd.Series(False, index=filtered.index)
                    if "18–25 years" in sel_ages:
                        age_mask |= (age_num >= 18) & (age_num <= 25)
                    if "26–35 years" in sel_ages:
                        age_mask |= (age_num >= 26) & (age_num <= 35)
                    if "36–50 years" in sel_ages:
                        age_mask |= (age_num >= 36) & (age_num <= 50)
                    if "50+ years" in sel_ages:
                        age_mask |= (age_num > 50)
                    filtered = filtered[age_mask]
            else:
                st.selectbox("Age Group", options=["Column not in dataset"], disabled=True, key="sp_age_na")

        with f3:
            if "ProductCategory" in filtered.columns:
                cat_opts = sorted(filtered["ProductCategory"].dropna().astype(str).unique().tolist())
                sel_cat = st.multiselect(
                    "Product Category",
                    options=cat_opts,
                    default=[],
                    placeholder="All Categories",
                    key="sp_filter_cat",
                )
                if sel_cat:
                    filtered = filtered[filtered["ProductCategory"].astype(str).isin(sel_cat)]
            else:
                st.selectbox("Product Category", options=["Column not in dataset"], disabled=True, key="sp_cat_na")

    return filtered


def _build_grouped_potential_table(df_sub: pd.DataFrame, group_col: str, label_col: str) -> pd.DataFrame:
    """Aggregates customer counts, predicted spending stats, and high-potential share by group_col."""
    rows = []
    for grp_val, grp_df in df_sub.groupby(group_col, dropna=True):
        cnt = int(len(grp_df))
        pred_s = grp_df["PredictedAnnualPurchaseAmount"].dropna()
        avg_pred = float(pred_s.mean()) if not pred_s.empty else np.nan
        med_pred = float(pred_s.median()) if not pred_s.empty else np.nan
        tot_pred = float(pred_s.sum()) if not pred_s.empty else 0.0
        hp_cnt = int((grp_df["SpendingPotentialGroup"] == "High Spending Potential").sum())
        hp_pct = (hp_cnt / cnt * 100.0) if cnt > 0 else 0.0

        rows.append({
            label_col: str(grp_val),
            "Customer Count": cnt,
            "_avg_pred": avg_pred,
            "_med_pred": med_pred,
            "_tot_pred": tot_pred,
            "_hp_cnt": hp_cnt,
            "_hp_pct": hp_pct,
        })

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(by="_avg_pred", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def render_spending_prediction():
    """Business Intelligence dashboard analyzing predicted annual spending potential across cohorts, regions, and product categories."""
    # 1. Page Title & Short Description
    st.markdown(
        '<div class="welcome">Spending <span style="color:#4f63e8">Prediction</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tagline">Analyze predicted annual spending potential &nbsp;•&nbsp; '
        'Identify high-potential cohorts &nbsp;•&nbsp; Guide targeted retention &amp; marketing strategy</div>',
        unsafe_allow_html=True,
    )

    # 2. Active Dataset Information & Saved Pipeline Loading
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

    # Preserve original dataset columns & rows; compute Recency only if absent
    work_df = _resolve_recency_column(raw.copy())
    if "CustomerTenure" not in work_df.columns and "CustomerTenureMonths" in work_df.columns:
        work_df["CustomerTenure"] = pd.to_numeric(work_df["CustomerTenureMonths"], errors="coerce") / 12.0

    norm_col_map = {c.lower().replace(" ", "").replace("_", ""): c for c in work_df.columns}
    missing_cols = []
    feat_df = pd.DataFrame(index=work_df.index)

    for feat in expected_features:
        if feat in work_df.columns:
            feat_df[feat] = pd.to_numeric(work_df[feat], errors="coerce")
        else:
            matched = norm_col_map.get(feat.lower().replace(" ", "").replace("_", ""))
            if matched is not None:
                feat_df[feat] = pd.to_numeric(work_df[matched], errors="coerce")
            else:
                missing_cols.append(feat)
                feat_df[feat] = np.nan

    if len(missing_cols) == len(expected_features) or feat_df.isna().all().all():
        st.warning(
            f"⚠️ Cannot run Spending Prediction because none of the required model features "
            f"(`{expected_features}`) are available in **{source_name}**."
        )
        return

    if missing_cols:
        st.info(
            f"ℹ️ **{source_name}** is missing **{', '.join(missing_cols)}**. "
            f"Using the saved regression pipeline's built-in median imputer for those feature(s)."
        )

    if feat_df.isna().all().all():
        st.warning("⚠️ None of the required model features contain valid numeric values in the active dataset.")
        return

    # Run saved pipeline directly without retraining or manual scaling/imputation
    work_df["PredictedAnnualPurchaseAmount"] = np.clip(
        reg_model.predict(feat_df[expected_features]), 0.0, None
    )

    # Calculate 25th & 75th percentile thresholds on the COMPLETE active dataset BEFORE filtering
    work_df["SpendingPotentialGroup"], p25_full, p75_full = _assign_spending_potential_groups(
        work_df["PredictedAnnualPurchaseAmount"]
    )

    # Apply Top Search Bar filter if typed
    q = st.session_state.get("search", "").strip().lower()
    if q:
        search_cols = [
            c for c in ["CustomerID", "Region", "Gender", "ProductCategory", "SpendingPotentialGroup"]
            if c in work_df.columns
        ]
        if search_cols:
            mask = work_df[search_cols].astype(str).apply(lambda s: s.str.lower().str.contains(q)).any(axis=1)
            work_df = work_df.loc[mask].copy()
            if work_df.empty:
                st.warning(f"No customers match “{q}”.")
                return

    # 3. Filters (Region, Age Group, Product Category)
    filtered_df = _apply_spending_filters(work_df)
    if filtered_df.empty:
        st.warning("⚠️ No customers match the selected filters. Clear one or more filters to view spending potential.")
        return

    st.write("")

    # 4. Four KPI Cards
    total_cust = int(len(filtered_df))
    pred_s = filtered_df["PredictedAnnualPurchaseAmount"].dropna()
    avg_pred = float(pred_s.mean()) if not pred_s.empty else 0.0
    tot_pred = float(pred_s.sum()) if not pred_s.empty else 0.0
    high_cnt = int((filtered_df["SpendingPotentialGroup"] == "High Spending Potential").sum())
    high_pct = (high_cnt / total_cust * 100.0) if total_cust > 0 else 0.0

    c = st.columns(4)
    kpi(c[0], "👥", "Total Customers", f"{total_cust:,}", "Filtered customer cohort", "#e8f1fd")
    kpi(c[1], "📈", "Avg. Predicted Annual Spending", money(avg_pred), "Per-customer model estimate", "#e3f6ec")
    kpi(c[2], "💼", "Total Predicted Annual Spending", format_inr(tot_pred), "Estimated cohort spend (not actual revenue)", "#f0eafd")
    kpi(c[3], "🌟", "High-Potential Customers", f"{high_cnt:,}", f"{high_pct:.1f}% of filtered cohort (> {money(p75_full)})", "#fdf3df")

    st.write("")

    # 5. Customer Predicted Annual Spending Distribution
    st.markdown('<div class="sec" style="margin-bottom:8px;">Customer Predicted Annual Spending Distribution</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="ct">📊 Customer Predicted Annual Spending Distribution</div>', unsafe_allow_html=True)
        hist_vals = pred_s.sample(min(50_000, len(pred_s)), random_state=42) if len(pred_s) > 50_000 else pred_s
        fig_dist = px.histogram(
            hist_vals,
            x="PredictedAnnualPurchaseAmount",
            nbins=28,
            color_discrete_sequence=["#4f63e8"],
        )
        fig_dist.add_vline(
            x=avg_pred,
            line_dash="dash",
            line_color="#22c277",
            line_width=2,
            annotation_text=f"Cohort Mean: {money(avg_pred)}",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#172033"),
        )
        fig_dist.update_traces(
            marker_line_width=1,
            marker_line_color="#3b82f6",
            hovertemplate=f"<b>Predicted Spend Range:</b> {CURRENCY}%{{x}}<br><b>Customers:</b> %{{y:,}}<extra></extra>",
        )
        fig_dist = style_chart(fig_dist, h=300)
        fig_dist.update_layout(margin=dict(l=12, r=16, t=28, b=16))
        fig_dist.update_xaxes(
            title_text=f"Predicted Annual Spending ({CURRENCY})",
            tickprefix=CURRENCY,
            tickformat="~s",
            gridcolor="#e2e8f0",
        )
        fig_dist.update_yaxes(title_text="Number of Customers", gridcolor="#e2e8f0")
        st.plotly_chart(fig_dist, use_container_width=True)

    st.write("")

    # 6. Customer Spending Potential Segmentation (Percentile-Based)
    st.markdown('<div class="sec" style="margin-bottom:4px;">Customer Spending Potential Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#475569;font-size:0.82rem;margin-bottom:10px;">'
        f'ℹ️ Derived model-based spending tiers using full-dataset percentile thresholds '
        f'(<b>Low:</b> &lt; {money(p25_full)} &nbsp;|&nbsp; '
        f'<b>Medium:</b> {money(p25_full)}–{money(p75_full)} &nbsp;|&nbsp; '
        f'<b>High:</b> &gt; {money(p75_full)}). These reflect predicted spending potential, not guaranteed revenue.</div>',
        unsafe_allow_html=True,
    )

    pot_order = ["High Spending Potential", "Medium Spending Potential", "Low Spending Potential"]
    pot_short = {
        "High Spending Potential": "High Potential",
        "Medium Spending Potential": "Medium Potential",
        "Low Spending Potential": "Low Potential",
    }
    pot_colors = {
        "High Spending Potential": "#3b82f6",    # Blue
        "Medium Spending Potential": "#22c277",  # Green
        "Low Spending Potential": "#fb923c",     # Orange
    }

    pot_counts = filtered_df["SpendingPotentialGroup"].value_counts()
    sp_c1, sp_c2 = st.columns([1, 1.2])

    with sp_c1:
        st.markdown('<div class="card"><h4>Customer Share by Spending Potential</h4>', unsafe_allow_html=True)
        d_vals = [int(pot_counts.get(g, 0)) for g in pot_order]
        d_labels = [f"{pot_short[g]} ({pot_counts.get(g, 0):,})" for g in pot_order]
        d_colors = [pot_colors[g] for g in pot_order]

        fig_pot_donut = go.Figure(go.Pie(
            labels=d_labels,
            values=d_vals,
            hole=0.56,
            sort=False,
            marker=dict(colors=d_colors, line=dict(color="#ffffff", width=2)),
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont=dict(size=12, color="#ffffff"),
            customdata=pot_order,
            hovertemplate="<b>%{customdata}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pot_donut.add_annotation(
            text=f"<b>{total_cust:,}</b><br>Customers",
            showarrow=False,
            font=dict(size=13, color="#172033"),
        )
        fig_pot_donut = style_chart(fig_pot_donut, h=310)
        fig_pot_donut.update_layout(
            margin=dict(l=16, r=16, t=20, b=44),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08, font=dict(size=11, color="#172033")),
        )
        st.plotly_chart(fig_pot_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with sp_c2:
        st.markdown('<div class="card"><h4>Average Predicted Annual Spending by Group</h4>', unsafe_allow_html=True)
        avg_pot = (
            filtered_df.groupby("SpendingPotentialGroup")["PredictedAnnualPurchaseAmount"]
            .mean()
            .reindex(pot_order)
            .fillna(0.0)
        )
        fig_pot_bar = go.Figure(go.Bar(
            x=[pot_short[g] for g in pot_order],
            y=avg_pot.values,
            marker_color=[pot_colors[g] for g in pot_order],
            text=[money(v) for v in avg_pot.values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=12, color="#172033"),
            customdata=pot_order,
            hovertemplate="<b>%{customdata}</b><br>Avg Predicted Spend: %{text}<extra></extra>",
        ))
        max_pot_y = float(avg_pot.max()) if avg_pot.max() > 0 else 1000.0
        fig_pot_bar = style_chart(fig_pot_bar, h=310)
        fig_pot_bar.update_layout(margin=dict(l=12, r=16, t=24, b=16), showlegend=False)
        fig_pot_bar.update_yaxes(
            range=[0, max_pot_y * 1.28],
            gridcolor="#e2e8f0",
            tickprefix=CURRENCY,
            tickformat="~s",
            title_text=f"Avg Predicted Spend ({CURRENCY})",
        )
        st.plotly_chart(fig_pot_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="ct">📋 Spending Potential Summary</div>', unsafe_allow_html=True)
        pot_rows = []
        for g in pot_order:
            sub = filtered_df[filtered_df["SpendingPotentialGroup"] == g]
            cnt = int(len(sub))
            pct = (cnt / total_cust * 100.0) if total_cust > 0 else 0.0
            s_p = sub["PredictedAnnualPurchaseAmount"].dropna()
            pot_rows.append({
                "Spending Potential Group": g,
                "Customer Count": f"{cnt:,}",
                "Customer Percentage": f"{pct:.1f}%",
                "Average Predicted Annual Spending": money(float(s_p.mean())) if not s_p.empty else "N/A",
                "Median Predicted Annual Spending": money(float(s_p.median())) if not s_p.empty else "N/A",
                "Total Predicted Annual Spending": format_inr(float(s_p.sum())) if not s_p.empty else "N/A",
            })
        st.dataframe(pd.DataFrame(pot_rows), hide_index=True, use_container_width=True)

    st.write("")

    # 7. Regional Spending Potential
    reg_summary = pd.DataFrame()
    if "Region" in filtered_df.columns and filtered_df["Region"].notna().any():
        st.markdown('<div class="sec" style="margin-bottom:8px;">Regional Spending Potential</div>', unsafe_allow_html=True)
        reg_summary = _build_grouped_potential_table(filtered_df, "Region", "Region")

        if not reg_summary.empty:
            r_chart_col, r_tbl_col = st.columns([1.15, 1.25])
            with r_chart_col:
                st.markdown('<div class="card"><h4>Average Predicted Annual Spending by Region</h4>', unsafe_allow_html=True)
                fig_reg = go.Figure(go.Bar(
                    x=reg_summary["Region"],
                    y=reg_summary["_avg_pred"],
                    marker_color=PALETTE[: len(reg_summary)],
                    text=[money(v) if pd.notna(v) else "N/A" for v in reg_summary["_avg_pred"]],
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=11, color="#172033"),
                    hovertemplate="<b>%{x}</b><br>Avg Predicted Spend: %{text}<extra></extra>",
                ))
                max_ry = float(reg_summary["_avg_pred"].max()) if pd.notna(reg_summary["_avg_pred"].max()) and reg_summary["_avg_pred"].max() > 0 else 1000.0
                fig_reg = style_chart(fig_reg, h=320)
                fig_reg.update_layout(margin=dict(l=12, r=16, t=28, b=16), showlegend=False)
                fig_reg.update_xaxes(title_text="Region")
                fig_reg.update_yaxes(
                    title_text=f"Avg Predicted Spend ({CURRENCY})",
                    tickprefix=CURRENCY,
                    tickformat="~s",
                    gridcolor="#e2e8f0",
                    range=[0, max_ry * 1.28],
                )
                st.plotly_chart(fig_reg, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with r_tbl_col:
                with st.container(border=True):
                    st.markdown('<div class="ct">📋 Regional Spending Potential Summary</div>', unsafe_allow_html=True)
                    reg_tbl = pd.DataFrame({
                        "Region": reg_summary["Region"],
                        "Customers": reg_summary["Customer Count"].map(lambda x: f"{x:,}"),
                        "Avg Predicted": reg_summary["_avg_pred"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
                        "Median Predicted": reg_summary["_med_pred"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
                        "Total Predicted": reg_summary["_tot_pred"].map(format_inr),
                        "High-Potential": reg_summary["_hp_cnt"].map(lambda x: f"{x:,}"),
                        "High-Potential %": reg_summary["_hp_pct"].map(lambda v: f"{v:.1f}%"),
                    })
                    st.dataframe(reg_tbl, hide_index=True, use_container_width=True)

        st.write("")

    # 8. Product Category Spending Potential
    cat_summary = pd.DataFrame()
    if "ProductCategory" in filtered_df.columns and filtered_df["ProductCategory"].notna().any():
        st.markdown('<div class="sec" style="margin-bottom:8px;">Product Category Spending Potential</div>', unsafe_allow_html=True)
        cat_summary = _build_grouped_potential_table(filtered_df, "ProductCategory", "Product Category")

        if not cat_summary.empty:
            c_chart_col, c_tbl_col = st.columns([1.15, 1.25])
            with c_chart_col:
                st.markdown('<div class="card"><h4>Average Predicted Annual Spending by Product Category</h4>', unsafe_allow_html=True)
                fig_cat = go.Figure(go.Bar(
                    x=cat_summary["Product Category"],
                    y=cat_summary["_avg_pred"],
                    marker_color=CAT_COLORS[: len(cat_summary)],
                    text=[money(v) if pd.notna(v) else "N/A" for v in cat_summary["_avg_pred"]],
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=11, color="#172033"),
                    hovertemplate="<b>%{x}</b><br>Avg Predicted Spend: %{text}<extra></extra>",
                ))
                max_cy = float(cat_summary["_avg_pred"].max()) if pd.notna(cat_summary["_avg_pred"].max()) and cat_summary["_avg_pred"].max() > 0 else 1000.0
                fig_cat = style_chart(fig_cat, h=330)
                fig_cat.update_layout(margin=dict(l=12, r=16, t=28, b=26), showlegend=False)
                fig_cat.update_xaxes(
                    title_text="Product Category",
                    tickangle=-18 if len(cat_summary) > 5 else 0,
                )
                fig_cat.update_yaxes(
                    title_text=f"Avg Predicted Spend ({CURRENCY})",
                    tickprefix=CURRENCY,
                    tickformat="~s",
                    gridcolor="#e2e8f0",
                    range=[0, max_cy * 1.28],
                )
                st.plotly_chart(fig_cat, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c_tbl_col:
                with st.container(border=True):
                    st.markdown('<div class="ct">📋 Product Category Spending Potential Summary</div>', unsafe_allow_html=True)
                    cat_tbl = pd.DataFrame({
                        "Product Category": cat_summary["Product Category"],
                        "Customers": cat_summary["Customer Count"].map(lambda x: f"{x:,}"),
                        "Avg Predicted": cat_summary["_avg_pred"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
                        "Median Predicted": cat_summary["_med_pred"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
                        "Total Predicted": cat_summary["_tot_pred"].map(format_inr),
                        "High-Potential": cat_summary["_hp_cnt"].map(lambda x: f"{x:,}"),
                        "High-Potential %": cat_summary["_hp_pct"].map(lambda v: f"{v:.1f}%"),
                    })
                    st.dataframe(cat_tbl, hide_index=True, use_container_width=True)

        st.write("")

    # 9. Customer Profile Analysis
    st.markdown('<div class="sec" style="margin-bottom:4px;">Customer Profile Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#475569;font-size:0.82rem;margin-bottom:8px;">'
        'ℹ️ Descriptive demographic and behavioral associations observed across predicted spending-potential tiers.</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown('<div class="ct">🧬 Profile Comparison by Spending Potential Group</div>', unsafe_allow_html=True)
        prof_rows = []
        for g in pot_order:
            sub = filtered_df[filtered_df["SpendingPotentialGroup"] == g]
            cnt = int(len(sub))

            def _mean_fmt(col_name: str, fmt_type: str) -> str:
                if cnt == 0 or col_name not in sub.columns:
                    return "N/A"
                s = pd.to_numeric(sub[col_name], errors="coerce").dropna()
                if s.empty:
                    return "N/A"
                val = float(s.mean())
                if fmt_type == "money":
                    return money(val)
                if fmt_type == "age":
                    return f"{val:.1f} yrs"
                if fmt_type == "tenure":
                    return f"{val:.2f} yrs"
                if fmt_type == "freq":
                    return f"{val:.1f}"
                if fmt_type == "pct":
                    return f"{val * 100:.1f}%"
                return f"{val:.2f}"

            prof_rows.append({
                "Spending Potential Group": g,
                "Customer Count": f"{cnt:,}",
                "Average Age": _mean_fmt("Age", "age"),
                "Average Annual Income": _mean_fmt("AnnualIncome", "money"),
                "Average Purchase Frequency": _mean_fmt("PurchaseFrequency", "freq"),
                "Average Customer Tenure": _mean_fmt("CustomerTenure", "tenure"),
                "Average Return Rate": _mean_fmt("ReturnRate", "pct"),
                "Average Predicted Annual Spending": _mean_fmt("PredictedAnnualPurchaseAmount", "money"),
            })
        st.dataframe(pd.DataFrame(prof_rows), hide_index=True, use_container_width=True)

    st.write("")

    # 10. Top Predicted Spending Customers
    st.markdown('<div class="sec" style="margin-bottom:8px;">Top Predicted Spending Customers</div>', unsafe_allow_html=True)
    with st.container(border=True):
        top_hdr1, top_hdr2 = st.columns([3, 1])
        top_hdr1.markdown('<div class="ct">🏆 Highest Predicted Spending Accounts (Prioritized Outreach)</div>', unsafe_allow_html=True)
        top_n = top_hdr2.radio(
            "Top N",
            options=[10, 20],
            horizontal=True,
            key="sp_top_n_customers",
            label_visibility="collapsed",
        )
        candidate_cols = [
            "CustomerID",
            "Region",
            "ProductCategory",
            "Age",
            "PurchaseFrequency",
            "AnnualIncome",
            "PredictedAnnualPurchaseAmount",
            "SpendingPotentialGroup",
        ]
        avail_top_cols = [col for col in candidate_cols if col in filtered_df.columns]
        top_cust_df = (
            filtered_df.sort_values(by="PredictedAnnualPurchaseAmount", ascending=False)
            .head(top_n)[avail_top_cols]
            .copy()
        )
        if "SpendingPotentialGroup" in top_cust_df.columns:
            top_cust_df = top_cust_df.rename(columns={"SpendingPotentialGroup": "Spending Potential Group"})

        top_col_cfg = {
            "AnnualIncome": st.column_config.NumberColumn(f"Annual Income ({CURRENCY})", format=f"{CURRENCY} %,.0f"),
            "PredictedAnnualPurchaseAmount": st.column_config.NumberColumn(
                f"Predicted Annual Spending ({CURRENCY})", format=f"{CURRENCY} %,.0f"
            ),
        }
        st.dataframe(top_cust_df, hide_index=True, use_container_width=True, column_config=top_col_cfg)

    st.write("")

    # 11. Dynamic Business Insights
    st.markdown('<div class="sec" style="margin-bottom:8px;">Business Insights</div>', unsafe_allow_html=True)
    insights = []

    # High-potential cohort size & share
    insights.append((
        "🌟", "#fdf3df", "#fb923c",
        f"{high_cnt:,} High-Potential Customers ({high_pct:.1f}%)",
        f"In the current selection, {high_cnt:,} customers exceed the {money(p75_full)} predicted spending threshold "
        f"and represent priority candidates for personalized VIP retention.",
    ))

    # Largest spending-potential group
    if not pot_counts.empty:
        dom_grp = str(pot_counts.index[0])
        dom_cnt = int(pot_counts.iloc[0])
        dom_pct = (dom_cnt / total_cust * 100.0) if total_cust > 0 else 0.0
        insights.append((
            "📊", "#e3f6ec", "#22c277",
            f"Largest Tier: {dom_grp}",
            f"{dom_grp} accounts for the largest share of the filtered cohort with {dom_cnt:,} customers ({dom_pct:.1f}%).",
        ))

    # Highest average predicted spending region
    if not reg_summary.empty and reg_summary["_avg_pred"].notna().any():
        top_reg = reg_summary.dropna(subset=["_avg_pred"]).iloc[0]
        insights.append((
            "🌍", "#e8f1fd", "#3b82f6",
            f"Highest Regional Potential: {top_reg['Region']}",
            f"{top_reg['Region']} records the highest average predicted annual spending at {money(top_reg['_avg_pred'])}. "
            f"Consider investigating this region for targeted retention and premium engagement campaigns.",
        ))

        top_hp_reg = reg_summary.sort_values(by=["_hp_pct", "_hp_cnt"], ascending=[False, False]).iloc[0]
        insights.append((
            "📍", "#f0eafd", "#8b5cf6",
            f"Highest High-Potential Concentration: {top_hp_reg['Region']}",
            f"{top_hp_reg['Region']} leads all regions in high-potential customer share at {top_hp_reg['_hp_pct']:.1f}% "
            f"({int(top_hp_reg['_hp_cnt']):,} of {int(top_hp_reg['Customer Count']):,} customers).",
        ))

    # Highest average predicted spending category & high-potential share
    if not cat_summary.empty and cat_summary["_avg_pred"].notna().any():
        top_cat = cat_summary.dropna(subset=["_avg_pred"]).iloc[0]
        insights.append((
            "🛍️", "#e0f2fe", "#0ea5e9",
            f"Top Category Potential: {top_cat['Product Category']}",
            f"Customers associated with {top_cat['Product Category']} show the highest average predicted annual spending at "
            f"{money(top_cat['_avg_pred'])} ({int(top_cat['Customer Count']):,} customers).",
        ))

        top_hp_cat = cat_summary.sort_values(by=["_hp_pct", "_hp_cnt"], ascending=[False, False]).iloc[0]
        insights.append((
            "🎯", "#fce7f3", "#f43f5e",
            f"Category with Highest High-Potential Share: {top_hp_cat['Product Category']}",
            f"{top_hp_cat['Product Category']} has the highest concentration of high-potential buyers at "
            f"{top_hp_cat['_hp_pct']:.1f}% ({int(top_hp_cat['_hp_cnt']):,} customers).",
        ))

    # Descriptive profile association for High vs Low Spending Potential
    hp_sub = filtered_df[filtered_df["SpendingPotentialGroup"] == "High Spending Potential"]
    lp_sub = filtered_df[filtered_df["SpendingPotentialGroup"] == "Low Spending Potential"]
    if (
        not hp_sub.empty
        and not lp_sub.empty
        and "AnnualIncome" in filtered_df.columns
        and "PurchaseFrequency" in filtered_df.columns
    ):
        hp_inc = pd.to_numeric(hp_sub["AnnualIncome"], errors="coerce").mean()
        lp_inc = pd.to_numeric(lp_sub["AnnualIncome"], errors="coerce").mean()
        hp_frq = pd.to_numeric(hp_sub["PurchaseFrequency"], errors="coerce").mean()
        lp_frq = pd.to_numeric(lp_sub["PurchaseFrequency"], errors="coerce").mean()
        if pd.notna(hp_inc) and pd.notna(lp_inc) and pd.notna(hp_frq) and pd.notna(lp_frq):
            insights.append((
                "🧬", "#eef2ff", "#4f63e8",
                "High-Potential Customer Profile Association",
                f"High Spending Potential customers average {money(hp_inc)} in annual income and {hp_frq:.1f} purchases/yr, "
                f"compared with {money(lp_inc)} and {lp_frq:.1f} purchases/yr in the Low Spending Potential tier.",
            ))

    if not insights:
        st.info("Insufficient data in the current filter selection to generate business insights.")
    else:
        for row_start in range(0, len(insights), 3):
            batch = insights[row_start : row_start + 3]
            cols = st.columns(len(batch))
            for idx, (icon, bg, accent, title, desc) in enumerate(batch):
                with cols[idx]:
                    st.markdown(
                        f"""
                        <div class="card" style="border-top:4px solid {accent};height:100%;">
                            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                                <div style="width:36px;height:36px;border-radius:50%;background:{bg};
                                            display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;">{icon}</div>
                                <div style="font-size:0.9rem;font-weight:800;color:#111a3b;line-height:1.25;">{title}</div>
                            </div>
                            <div style="font-size:0.78rem;color:#475569;line-height:1.45;">{desc}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# def render_spending_prediction():
#     """Renders the redesigned Spending Prediction page with cohort filters, regional/category comparisons, accuracy metrics, and insights."""
#     st.markdown(
#         '<div class="welcome">Spending <span style="color:#4f63e8">Prediction</span></div>',
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         '<div class="tagline">Actual vs. predicted annual spending across regions and product categories '
#         '&nbsp;•&nbsp; Powered by pre-trained HistGradientBoosting regression</div>',
#         unsafe_allow_html=True,
#     )

#     raw, source_name = get_active_raw()
#     if raw is None or raw.empty:
#         st.markdown(
#             """<div class="vacant"><div style="font-size:2.5rem">📂</div>
#             <h3>No dataset available</h3>
#             <p>Ensure <b>Data/Main_Clean.csv</b> exists or upload a CSV/Excel file in the left sidebar.</p></div>""",
#             unsafe_allow_html=True,
#         )
#         return

#     st.markdown(
#         f'<div style="color:#334155;font-size:0.84rem;font-weight:600;margin:-4px 0 12px 0;">'
#         f'📁 Active Dataset: <span style="color:#111a3b;">{source_name}</span> ({len(raw):,} total records)</div>',
#         unsafe_allow_html=True,
#     )

#     reg_model, model_err = load_spending_model()
#     if model_err:
#         st.error(f"❌ {model_err}")
#         return

#     expected_features = (
#         list(reg_model.feature_names_in_)
#         if hasattr(reg_model, "feature_names_in_")
#         else list(SPENDING_FEATURES)
#     )

#     work_df = _resolve_recency_column(raw.copy())

#     # Convert CustomerTenureMonths -> CustomerTenure (in years) if CustomerTenure is absent
#     if "CustomerTenure" not in work_df.columns and "CustomerTenureMonths" in work_df.columns:
#         work_df["CustomerTenure"] = pd.to_numeric(work_df["CustomerTenureMonths"], errors="coerce") / 12.0

#     # Case/space-insensitive column matching for remaining features
#     norm_col_map = {c.lower().replace(" ", "").replace("_", ""): c for c in work_df.columns}
#     missing_cols = []
#     feat_df = pd.DataFrame(index=work_df.index)

#     for feat in expected_features:
#         if feat in work_df.columns:
#             feat_df[feat] = pd.to_numeric(work_df[feat], errors="coerce")
#         else:
#             matched = norm_col_map.get(feat.lower().replace(" ", "").replace("_", ""))
#             if matched is not None:
#                 feat_df[feat] = pd.to_numeric(work_df[matched], errors="coerce")
#             else:
#                 missing_cols.append(feat)
#                 feat_df[feat] = np.nan

#     if len(missing_cols) == len(expected_features) or feat_df.isna().all().all():
#         st.warning(
#             f"⚠️ Cannot run Spending Prediction because none of the required model features "
#             f"(`{expected_features}`) are available in **{source_name}**."
#         )
#         return

#     if missing_cols:
#         st.info(
#             f"ℹ️ **{source_name}** is missing **{', '.join(missing_cols)}**. "
#             f"Using the saved regression pipeline's built-in median imputer for those feature(s)."
#         )

#     if feat_df.isna().all().all():
#         st.warning("⚠️ None of the six required model features contain valid numeric values in the active dataset.")
#         return

#     work_df["PredictedAnnualPurchaseAmount"] = reg_model.predict(feat_df[expected_features])
#     work_df["_actual_spend"] = (
#         pd.to_numeric(work_df["AnnualPurchaseAmount"], errors="coerce")
#         if "AnnualPurchaseAmount" in work_df.columns
#         else np.nan
#     )

#     q = st.session_state.get("search", "").strip().lower()
#     if q:
#         search_cols = [c for c in ["CustomerID", "Region", "Gender", "ProductCategory"] if c in work_df.columns]
#         if search_cols:
#             mask = work_df[search_cols].astype(str).apply(lambda s: s.str.lower().str.contains(q)).any(axis=1)
#             work_df = work_df.loc[mask].copy()
#             if work_df.empty:
#                 st.warning(f"No customers match “{q}”.")
#                 return

#     # ---------------- 3. COHORT FILTERS (Region, Age Group, Product Category) ----------------
#     filtered_df = _apply_spending_filters(work_df)
#     if filtered_df.empty:
#         st.warning("⚠️ No customers match the selected filters. Clear one or more filters to view spending analysis.")
#         return

#     st.write("")

#     # ---------------- 4. FOUR KPI CARDS ----------------
#     total_cust = int(len(filtered_df))
#     valid_act = filtered_df["_actual_spend"].dropna()
#     valid_pred = filtered_df["PredictedAnnualPurchaseAmount"].dropna()
#     valid_both = filtered_df.dropna(subset=["_actual_spend", "PredictedAnnualPurchaseAmount"])

#     avg_actual_txt = money(float(valid_act.mean())) if not valid_act.empty else "N/A"
#     avg_actual_sub = f"Across {len(valid_act):,} observed records" if not valid_act.empty else "No actual spend values"

#     avg_pred_txt = money(float(valid_pred.mean())) if not valid_pred.empty else "N/A"
#     avg_pred_sub = (
#         f"Median {money(float(valid_pred.median()))} (model output)"
#         if not valid_pred.empty
#         else "No predictions available"
#     )

#     if not valid_both.empty:
#         abs_errors = (valid_both["_actual_spend"] - valid_both["PredictedAnnualPurchaseAmount"]).abs()
#         mae_kpi_txt = money(float(abs_errors.mean()))
#         mae_kpi_sub = f"Avg |Actual − Predicted| ({len(valid_both):,} rows)"
#     else:
#         mae_kpi_txt = "N/A"
#         mae_kpi_sub = "Requires actual & predicted values"

#     c = st.columns(4)
#     kpi(c[0], "👥", "Total Customers", f"{total_cust:,}", "Filtered cohort size", "#e8f1fd")
#     kpi(c[1], "🧾", "Avg. Actual Annual Spending", avg_actual_txt, avg_actual_sub, "#e3f6ec")
#     kpi(c[2], "📈", "Avg. Predicted Annual Spending", avg_pred_txt, avg_pred_sub, "#f0eafd")
#     kpi(c[3], "🎯", "Mean Absolute Prediction Error", mae_kpi_txt, mae_kpi_sub, "#fdf3df")

#     st.write("")

#     # ---------------- 5. REGIONAL SPENDING ANALYSIS ----------------
#     reg_summary = pd.DataFrame()
#     if "Region" in filtered_df.columns and filtered_df["Region"].notna().any():
#         st.markdown('<div class="sec" style="margin-bottom:8px;">Regional Spending Analysis</div>', unsafe_allow_html=True)
#         reg_summary = _build_grouped_spending_table(filtered_df, "Region", "Region")

#         if not reg_summary.empty:
#             reg_col_chart, reg_col_tbl = st.columns([1.25, 1])

#             with reg_col_chart:
#                 st.markdown('<div class="card"><h4>Actual vs Predicted Annual Spending by Region</h4>', unsafe_allow_html=True)
#                 fig_reg = go.Figure()
#                 fig_reg.add_trace(
#                     go.Bar(
#                         name="Average Actual Spending",
#                         x=reg_summary["Region"],
#                         y=reg_summary["_avg_actual"],
#                         marker_color="#22c277",
#                         text=[money(v) if pd.notna(v) else "N/A" for v in reg_summary["_avg_actual"]],
#                         textposition="outside",
#                         cliponaxis=False,
#                         textfont=dict(size=10, color="#172033"),
#                         hovertemplate="<b>%{x}</b><br>Avg Actual: %{text}<extra></extra>",
#                     )
#                 )
#                 fig_reg.add_trace(
#                     go.Bar(
#                         name="Average Predicted Spending",
#                         x=reg_summary["Region"],
#                         y=reg_summary["_avg_pred"],
#                         marker_color="#3b82f6",
#                         text=[money(v) if pd.notna(v) else "N/A" for v in reg_summary["_avg_pred"]],
#                         textposition="outside",
#                         cliponaxis=False,
#                         textfont=dict(size=10, color="#172033"),
#                         hovertemplate="<b>%{x}</b><br>Avg Predicted: %{text}<extra></extra>",
#                     )
#                 )
#                 max_reg_y = float(
#                     np.nanmax([reg_summary["_avg_actual"].max(), reg_summary["_avg_pred"].max()])
#                 )
#                 max_reg_y = max_reg_y if pd.notna(max_reg_y) and max_reg_y > 0 else 1000.0
#                 fig_reg = style_chart(fig_reg, h=330)
#                 fig_reg.update_layout(
#                     barmode="group",
#                     bargap=0.22,
#                     bargroupgap=0.08,
#                     margin=dict(l=12, r=16, t=44, b=16),
#                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
#                 )
#                 fig_reg.update_xaxes(title_text="Region")
#                 fig_reg.update_yaxes(
#                     title_text=f"Average Annual Spending ({CURRENCY})",
#                     tickprefix=CURRENCY,
#                     tickformat="~s",
#                     gridcolor="#e2e8f0",
#                     range=[0, max_reg_y * 1.28],
#                 )
#                 st.plotly_chart(fig_reg, use_container_width=True)
#                 st.markdown("</div>", unsafe_allow_html=True)

#             with reg_col_tbl:
#                 with st.container(border=True):
#                     st.markdown('<div class="ct">📋 Regional Summary</div>', unsafe_allow_html=True)
#                     reg_display = pd.DataFrame({
#                         "Region": reg_summary["Region"],
#                         "Customer Count": reg_summary["Customer Count"].map(lambda x: f"{x:,}"),
#                         "Avg Actual Spending": reg_summary["_avg_actual"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
#                         "Avg Predicted Spending": reg_summary["_avg_pred"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
#                         "Mean Absolute Error": reg_summary["_mae"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
#                     })
#                     st.dataframe(reg_display, hide_index=True, use_container_width=True)

#         st.write("")

#     # ---------------- 6. PRODUCT CATEGORY SPENDING ANALYSIS ----------------
#     cat_summary = pd.DataFrame()
#     if "ProductCategory" in filtered_df.columns and filtered_df["ProductCategory"].notna().any():
#         st.markdown('<div class="sec" style="margin-bottom:8px;">Product Category Spending Analysis</div>', unsafe_allow_html=True)
#         cat_summary = _build_grouped_spending_table(filtered_df, "ProductCategory", "Product Category")

#         if not cat_summary.empty:
#             cat_col_chart, cat_col_tbl = st.columns([1.25, 1])

#             with cat_col_chart:
#                 st.markdown(
#                     '<div class="card"><h4>Actual vs Predicted Annual Spending by Product Category</h4>',
#                     unsafe_allow_html=True,
#                 )
#                 fig_cat = go.Figure()
#                 fig_cat.add_trace(
#                     go.Bar(
#                         name="Average Actual Spending",
#                         x=cat_summary["Product Category"],
#                         y=cat_summary["_avg_actual"],
#                         marker_color="#22c277",
#                         text=[money(v) if pd.notna(v) else "N/A" for v in cat_summary["_avg_actual"]],
#                         textposition="outside",
#                         cliponaxis=False,
#                         textfont=dict(size=10, color="#172033"),
#                         hovertemplate="<b>%{x}</b><br>Avg Actual: %{text}<extra></extra>",
#                     )
#                 )
#                 fig_cat.add_trace(
#                     go.Bar(
#                         name="Average Predicted Spending",
#                         x=cat_summary["Product Category"],
#                         y=cat_summary["_avg_pred"],
#                         marker_color="#4f63e8",
#                         text=[money(v) if pd.notna(v) else "N/A" for v in cat_summary["_avg_pred"]],
#                         textposition="outside",
#                         cliponaxis=False,
#                         textfont=dict(size=10, color="#172033"),
#                         hovertemplate="<b>%{x}</b><br>Avg Predicted: %{text}<extra></extra>",
#                     )
#                 )
#                 max_cat_y = float(
#                     np.nanmax([cat_summary["_avg_actual"].max(), cat_summary["_avg_pred"].max()])
#                 )
#                 max_cat_y = max_cat_y if pd.notna(max_cat_y) and max_cat_y > 0 else 1000.0
#                 fig_cat = style_chart(fig_cat, h=340)
#                 fig_cat.update_layout(
#                     barmode="group",
#                     bargap=0.22,
#                     bargroupgap=0.08,
#                     margin=dict(l=12, r=16, t=44, b=28),
#                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
#                 )
#                 fig_cat.update_xaxes(
#                     title_text="Product Category",
#                     tickangle=-18 if len(cat_summary) > 5 else 0,
#                 )
#                 fig_cat.update_yaxes(
#                     title_text=f"Average Annual Spending ({CURRENCY})",
#                     tickprefix=CURRENCY,
#                     tickformat="~s",
#                     gridcolor="#e2e8f0",
#                     range=[0, max_cat_y * 1.28],
#                 )
#                 st.plotly_chart(fig_cat, use_container_width=True)
#                 st.markdown("</div>", unsafe_allow_html=True)

#             with cat_col_tbl:
#                 with st.container(border=True):
#                     st.markdown('<div class="ct">📋 Category Summary</div>', unsafe_allow_html=True)
#                     cat_display = pd.DataFrame({
#                         "Product Category": cat_summary["Product Category"],
#                         "Customer Count": cat_summary["Customer Count"].map(lambda x: f"{x:,}"),
#                         "Avg Actual Spending": cat_summary["_avg_actual"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
#                         "Avg Predicted Spending": cat_summary["_avg_pred"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
#                         "Mean Absolute Error": cat_summary["_mae"].map(lambda v: money(v) if pd.notna(v) else "N/A"),
#                     })
#                     st.dataframe(cat_display, hide_index=True, use_container_width=True)

#         st.write("")

#     # ---------------- 7. PREDICTION ACCURACY ANALYSIS ----------------
#     st.markdown('<div class="sec" style="margin-bottom:4px;">Prediction Accuracy Analysis</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div style="color:#475569;font-size:0.82rem;margin-bottom:10px;">'
#         'ℹ️ These evaluation metrics measure <b>regression model accuracy</b> on the filtered cohort '
#         '(comparing predicted vs. observed <code>AnnualPurchaseAmount</code>), not customer revenue.</div>',
#         unsafe_allow_html=True,
#     )

#     if valid_both.empty:
#         st.info("Actual `AnnualPurchaseAmount` values are unavailable in this dataset, so error metrics cannot be computed.")
#     else:
#         y_true = valid_both["_actual_spend"].to_numpy(dtype=float)
#         y_hat = valid_both["PredictedAnnualPurchaseAmount"].to_numpy(dtype=float)
#         err_diff = y_true - y_hat

#         mae_score = float(np.mean(np.abs(err_diff)))
#         rmse_score = float(np.sqrt(np.mean(err_diff ** 2)))

#         if len(y_true) >= 2 and float(np.var(y_true)) > 1e-12:
#             ss_res = float(np.sum(err_diff ** 2))
#             ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
#             r2_txt = f"{1.0 - (ss_res / ss_tot):.4f}"
#             r2_sub = f"Variance explained ({len(y_true):,} samples)"
#         else:
#             r2_txt = "N/A"
#             r2_sub = "Requires ≥ 2 samples with spend variance"

#         acc_left, acc_right = st.columns([1, 1.35])
#         with acc_left:
#             with st.container(border=True):
#                 st.markdown('<div class="ct">📐 Model Evaluation Metrics</div>', unsafe_allow_html=True)
#                 st.markdown(
#                     f"""
#                     <div style="display:grid;grid-template-columns:1fr;gap:10px;margin-top:6px;">
#                         <div style="background:#f8f9fc;border-left:4px solid #3b82f6;border-radius:8px;padding:10px 14px;">
#                             <div style="font-size:0.72rem;color:#475569;font-weight:700;">MAE (MEAN ABSOLUTE ERROR)</div>
#                             <div style="font-size:1.35rem;font-weight:800;color:#111a3b;">{money(mae_score)}</div>
#                             <div style="font-size:0.72rem;color:#64748b;">Average absolute rupee deviation per customer</div>
#                         </div>
#                         <div style="background:#f8f9fc;border-left:4px solid #8b5cf6;border-radius:8px;padding:10px 14px;">
#                             <div style="font-size:0.72rem;color:#475569;font-weight:700;">RMSE (ROOT MEAN SQUARED ERROR)</div>
#                             <div style="font-size:1.35rem;font-weight:800;color:#111a3b;">{money(rmse_score)}</div>
#                             <div style="font-size:0.72rem;color:#64748b;">Penalizes large outlier spending errors</div>
#                         </div>
#                         <div style="background:#f8f9fc;border-left:4px solid #22c277;border-radius:8px;padding:10px 14px;">
#                             <div style="font-size:0.72rem;color:#475569;font-weight:700;">R² SCORE (COEFFICIENT OF DETERMINATION)</div>
#                             <div style="font-size:1.35rem;font-weight:800;color:#111a3b;">{r2_txt}</div>
#                             <div style="font-size:0.72rem;color:#64748b;">{r2_sub}</div>
#                         </div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#         with acc_right:
#             err_df = cat_summary if not cat_summary.empty else reg_summary
#             err_dim = "Product Category" if not cat_summary.empty else "Region"
#             if not err_df.empty and err_df["_mae"].notna().any():
#                 err_sorted = err_df.dropna(subset=["_mae"]).sort_values("_mae", ascending=False)
#                 st.markdown(
#                     f'<div class="card"><h4>Mean Absolute Prediction Error by {err_dim}</h4>',
#                     unsafe_allow_html=True,
#                 )
#                 fig_err = go.Figure(
#                     go.Bar(
#                         x=err_sorted[err_dim],
#                         y=err_sorted["_mae"],
#                         marker_color="#fb923c",
#                         text=[money(v) for v in err_sorted["_mae"]],
#                         textposition="outside",
#                         cliponaxis=False,
#                         textfont=dict(size=11, color="#172033"),
#                         hovertemplate="<b>%{x}</b><br>MAE: %{text}<extra></extra>",
#                     )
#                 )
#                 max_err_y = float(err_sorted["_mae"].max()) if err_sorted["_mae"].max() > 0 else 1000.0
#                 fig_err = style_chart(fig_err, h=305)
#                 fig_err.update_layout(margin=dict(l=12, r=16, t=28, b=24), showlegend=False)
#                 fig_err.update_xaxes(
#                     title_text=err_dim,
#                     tickangle=-18 if len(err_sorted) > 5 else 0,
#                 )
#                 fig_err.update_yaxes(
#                     title_text=f"Mean Absolute Error ({CURRENCY})",
#                     tickprefix=CURRENCY,
#                     tickformat="~s",
#                     gridcolor="#e2e8f0",
#                     range=[0, max_err_y * 1.28],
#                 )
#                 st.plotly_chart(fig_err, use_container_width=True)
#                 st.markdown("</div>", unsafe_allow_html=True)
#             else:
#                 st.info("Insufficient regional or category breakdown to plot prediction error.")

#     st.write("")

#     # ---------------- 8. DYNAMIC BUSINESS INSIGHTS ----------------
#     st.markdown('<div class="sec" style="margin-bottom:8px;">Business Insights</div>', unsafe_allow_html=True)
#     insight_cards = []

#     if not reg_summary.empty and reg_summary["_avg_actual"].notna().any():
#         top_reg_row = reg_summary.dropna(subset=["_avg_actual"]).iloc[0]
#         insight_cards.append((
#             "🌍", "#e8f1fd", "#3b82f6",
#             f"Highest-Spending Region: {top_reg_row['Region']}",
#             f"Customers in {top_reg_row['Region']} record the highest average actual annual spend at "
#             f"{money(top_reg_row['_avg_actual'])} (vs. {money(top_reg_row['_avg_pred'])} predicted).",
#         ))

#     if not reg_summary.empty and reg_summary["_mae"].notna().any():
#         worst_reg_row = reg_summary.dropna(subset=["_mae"]).sort_values("_mae", ascending=False).iloc[0]
#         insight_cards.append((
#             "⚠️", "#fdf3df", "#fb923c",
#             f"Largest Regional Prediction Error: {worst_reg_row['Region']}",
#             f"{worst_reg_row['Region']} shows the highest MAE at {money(worst_reg_row['_mae'])} across "
#             f"{int(worst_reg_row['Customer Count']):,} customers, indicating higher spending variance.",
#         ))

#     if not cat_summary.empty and cat_summary["_avg_actual"].notna().any():
#         top_cat_row = cat_summary.dropna(subset=["_avg_actual"]).iloc[0]
#         insight_cards.append((
#             "🛍️", "#e3f6ec", "#22c277",
#             f"Top-Spending Category: {top_cat_row['Product Category']}",
#             f"{top_cat_row['Product Category']} leads all categories with {money(top_cat_row['_avg_actual'])} "
#             f"in average actual annual spending ({int(top_cat_row['Customer Count']):,} customers).",
#         ))

#     if not cat_summary.empty and cat_summary["_bias"].notna().any():
#         valid_bias = cat_summary.dropna(subset=["_bias"])
#         over_cats = valid_bias[valid_bias["_bias"] > 0].sort_values("_bias", ascending=False)
#         if not over_cats.empty:
#             r_over = over_cats.iloc[0]
#             insight_cards.append((
#                 "📈", "#f0eafd", "#8b5cf6",
#                 f"Predicted Above Actual: {r_over['Product Category']}",
#                 f"Model predictions for {r_over['Product Category']} ({money(r_over['_avg_pred'])}) exceed "
#                 f"actual spending ({money(r_over['_avg_actual'])}) by {money(r_over['_bias'])} on average.",
#             ))
#         under_cats = valid_bias[valid_bias["_bias"] < 0].sort_values("_bias", ascending=True)
#         if not under_cats.empty:
#             r_under = under_cats.iloc[0]
#             insight_cards.append((
#                 "📉", "#fce7f3", "#f43f5e",
#                 f"Predicted Below Actual: {r_under['Product Category']}",
#                 f"Actual spending in {r_under['Product Category']} ({money(r_under['_avg_actual'])}) outpaces "
#                 f"model predictions ({money(r_under['_avg_pred'])}) by {money(abs(r_under['_bias']))} on average.",
#             ))

#     if not insight_cards:
#         st.info("Insufficient regional, category, or actual spending data in the current filter selection to generate insights.")
#     else:
#         for row_start in range(0, len(insight_cards), 3):
#             batch = insight_cards[row_start : row_start + 3]
#             cols = st.columns(len(batch))
#             for idx, (icon, bg, accent, title, desc) in enumerate(batch):
#                 with cols[idx]:
#                     st.markdown(
#                         f"""
#                         <div class="card" style="border-top:4px solid {accent};height:100%;">
#                             <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
#                                 <div style="width:36px;height:36px;border-radius:50%;background:{bg};
#                                             display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;">{icon}</div>
#                                 <div style="font-size:0.9rem;font-weight:800;color:#111a3b;line-height:1.25;">{title}</div>
#                             </div>
#                             <div style="font-size:0.78rem;color:#475569;line-height:1.45;">{desc}</div>
#                         </div>
#                         """,
#                         unsafe_allow_html=True,
#                     )            

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