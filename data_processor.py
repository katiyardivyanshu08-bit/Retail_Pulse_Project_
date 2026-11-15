"""
RetailPulse – Data Processing & Telemetry Helpers
Provides CSV loading, column detection, KPI computation, correlation analysis,
and dynamic insight generation while preserving missing values in Data/Main_Clean.csv.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def load_csv_data(source: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Loads a CSV from a file path or Streamlit UploadedFile without dropping rows
    or altering missing values. Returns (df, error_message).
    """
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source)
        if df.empty:
            return None, "The CSV file contains no records."
        return df, None
    except Exception as exc:
        return None, str(exc)


def identify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorizes DataFrame columns into numeric, non-ID feature_numeric, and categorical.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    id_tokens = ("id", "index", "uuid", "code")
    feature_numeric = [
        col for col in numeric_cols
        if not any(tok == col.lower() or col.lower().endswith(f"_{tok}") or col.lower().endswith(tok) for tok in id_tokens)
    ]
    if not feature_numeric:
        feature_numeric = list(numeric_cols)

    return {
        "numeric": numeric_cols,
        "feature_numeric": feature_numeric,
        "categorical": categorical_cols,
    }


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes structural and data-quality KPIs across the full dataset,
    accounting accurately for null/missing cells.
    """
    total_records = int(len(df))
    total_features = int(df.shape[1])
    col_info = identify_columns(df)
    numeric_count = int(len(col_info["numeric"]))
    categorical_count = int(len(col_info["categorical"]))
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 ** 2))

    total_cells = total_records * total_features
    missing_cells = int(df.isna().sum().sum())
    missing_pct = float((missing_cells / total_cells) * 100) if total_cells > 0 else 0.0

    return {
        "total_records": total_records,
        "total_features": total_features,
        "numeric_count": numeric_count,
        "categorical_count": categorical_count,
        "memory_mb": memory_mb,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
    }


def compute_correlation_matrix(
    df: pd.DataFrame, max_features: int = 8
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Computes a Pearson correlation matrix for up to `max_features` non-ID,
    non-constant numeric columns while safely handling missing values pairwise.
    """
    col_info = identify_columns(df)
    candidates = col_info["feature_numeric"]

    valid_cols: List[str] = []
    for col in candidates:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.nunique() > 1:
            valid_cols.append(col)
        if len(valid_cols) >= max_features:
            break

    if len(valid_cols) < 2:
        return None, valid_cols

    numeric_df = df[valid_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric_df.corr(method="pearson")
    return corr_matrix, valid_cols


def compute_quick_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generates up to 4 dynamic statistical insights from numerical and categorical
    distributions, ignoring missing values per metric without dropping rows.
    """
    if df is None or df.empty:
        return []

    insights: List[Dict[str, str]] = []
    col_info = identify_columns(df)

    # 1. Spending / primary revenue distribution insight
    spend_candidates = [c for c in col_info["feature_numeric"] if any(k in c.lower() for k in ("spend", "amount", "revenue", "sales"))]
    target_num = spend_candidates[0] if spend_candidates else (col_info["feature_numeric"][0] if col_info["feature_numeric"] else None)

    if target_num:
        s = pd.to_numeric(df[target_num], errors="coerce").dropna()
        if not s.empty:
            insights.append({
                "icon": "💰",
                "bg": "#EFF6FF",
                "title": f"{target_num} averages {s.mean():,.1f}",
                "desc": f"Median sits at {s.median():,.1f} across {len(s):,} valid non-null records (max {s.max():,.1f}).",
            })

    # 2. Dominant categorical segment insight
    if col_info["categorical"]:
        cat_col = col_info["categorical"][0]
        vc = df[cat_col].dropna().astype(str).value_counts()
        if not vc.empty:
            top_val = vc.index[0]
            share = (vc.iloc[0] / vc.sum()) * 100
            insights.append({
                "icon": "🏷️",
                "bg": "#ECFDF5",
                "title": f"Top {cat_col}: {top_val} ({share:.1f}%)",
                "desc": f"Accounts for {vc.iloc[0]:,} of {vc.sum():,} non-null entries across {vc.shape[0]} distinct categories.",
            })

    # 3. Strongest feature correlation insight
    corr_matrix, valid_cols = compute_correlation_matrix(df, max_features=8)
    if corr_matrix is not None and len(valid_cols) >= 2:
        corr_copy = corr_matrix.to_numpy(copy=True)
        np.fill_diagonal(corr_copy, 0.0)
        max_idx = np.unravel_index(np.argmax(np.abs(corr_copy)), corr_copy.shape)
        r_val = float(corr_copy[max_idx])
        c1, c2 = valid_cols[max_idx[0]], valid_cols[max_idx[1]]
        direction = "positive" if r_val >= 0 else "negative"
        insights.append({
            "icon": "🔗",
            "bg": "#EEF2FF",
            "title": f"Strongest {direction} link: {c1} & {c2}",
            "desc": f"Pairwise Pearson correlation r = {r_val:.2f} computed on available overlapping values.",
        })

    # 4. Secondary numerical spread insight
    other_nums = [c for c in col_info["feature_numeric"] if c != target_num]
    if other_nums:
        sec_col = other_nums[0]
        s2 = pd.to_numeric(df[sec_col], errors="coerce").dropna()
        if not s2.empty:
            insights.append({
                "icon": "📊",
                "bg": "#FEF3C7",
                "title": f"{sec_col} spans {s2.min():,.1f} to {s2.max():,.1f}",
                "desc": f"Mean value is {s2.mean():,.1f} with standard deviation {s2.std():,.1f}.",
            })

    return insights[:4]


def compute_bi_highlights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Generates Business Intelligence & Data Health highlights without mutating or
    filtering out null rows from the source DataFrame.
    """
    if df is None or df.empty:
        return []

    kpis = compute_kpis(df)
    col_info = identify_columns(df)
    highlights: List[Dict[str, str]] = []

    # 1. Data Completeness / Missing Value Health
    if kpis["missing_cells"] == 0:
        highlights.append({
            "icon": "✅",
            "badge": "Data Health",
            "badge_bg": "#ECFDF5",
            "badge_color": "#047857",
            "title": "100% Complete Dataset",
            "desc": f"All {kpis['total_records']:,} records across {kpis['total_features']} features contain zero missing cells.",
        })
    else:
        null_cols = df.columns[df.isna().any()].tolist()
        highlights.append({
            "icon": "🛡️",
            "badge": "Imputation Ready",
            "badge_bg": "#FEF3C7",
            "badge_color": "#B45309",
            "title": f"{kpis['missing_cells']:,} Missing Cells Preserved ({kpis['missing_pct']:.2f}%)",
            "desc": f"Nulls detected in {len(null_cols)} column(s) ({', '.join(null_cols[:3])}{'…' if len(null_cols) > 3 else ''}); handled safely during analytics & ML pipelines.",
        })

    # 2. Feature Schema Balance
    highlights.append({
        "icon": "🧬",
        "badge": "Schema Profile",
        "badge_bg": "#EFF6FF",
        "badge_color": "#1D4ED8",
        "title": f"{kpis['numeric_count']} Quantitative & {kpis['categorical_count']} Categorical Features",
        "desc": f"Dataset occupies ~{kpis['memory_mb']:.2f} MB in memory with {len(col_info['feature_numeric'])} modeling-ready numeric attributes.",
    })

    # 3. Duplicate Record Check
    dup_count = int(df.duplicated().sum())
    highlights.append({
        "icon": "🔍",
        "badge": "Integrity",
        "badge_bg": "#EEF2FF",
        "badge_color": "#4338CA",
        "title": f"{dup_count:,} Exact Duplicate Rows Detected",
        "desc": f"{kpis['total_records'] - dup_count:,} unique customer records available for segmentation and modeling.",
    })

    return highlights