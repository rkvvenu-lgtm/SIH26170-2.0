import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

RESULT_FILE = RESULTS_DIR / "final_screening_results.csv"
METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
DATA_FILE = DATA_DIR / "burn_in_measurements.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Assisted Adaptive Component Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    .status-pass {
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        background: #dff5e3;
    }

    .status-investigate {
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        background: #fff1cc;
    }

    .status-reject {
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 700;
        text-align: center;
        background: #ffdede;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_results():
    if not RESULT_FILE.exists():
        return None

    return pd.read_csv(RESULT_FILE)


@st.cache_data
def load_metrics():
    if not METRICS_FILE.exists():
        return None

    return pd.read_csv(METRICS_FILE)


@st.cache_data
def load_raw_data():
    if not DATA_FILE.exists():
        return None

    return pd.read_csv(DATA_FILE)


results = load_results()
metrics = load_metrics()
raw_data = load_raw_data()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 AI-Assisted Adaptive Component Screening</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Predictive burn-in analysis using AI anomaly detection,
    multi-parameter drift prediction, engineering specifications,
    and risk-based screening.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE CHECK
# ============================================================

if results is None:
    st.error(
        "Screening results were not found. "
        "Run the V2 screening pipeline first."
    )

    st.code(
        "python -c "
        "\"from main import ScreeningPipeline; "
        "ScreeningPipeline().run('data/burn_in_measurements.csv')\"",
        language="powershell",
    )

    st.stop()


if metrics is None:
    st.warning(
        "Model metrics file was not found. "
        "Some AI model comparison sections will be unavailable."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Overview",
        "Component Screening",
        "AI Anomaly Analysis",
        "168h Prediction",
        "Model Comparison",
        "Engineering Specification",
    ],
)

st.sidebar.markdown("---")

st.sidebar.write("### Data Source")
st.sidebar.write("Burn-in Measurement Dataset")

st.sidebar.write("### Records")
st.sidebar.write(f"{len(results):,} components")


# ============================================================
# COMMON COUNTS
# ============================================================

decision_counts = (
    results["Decision"]
    .value_counts()
    if "Decision" in results.columns
    else pd.Series(dtype=int)
)

pass_count = int(decision_counts.get("PASS", 0))
investigate_count = int(decision_counts.get("INVESTIGATE", 0))
reject_count = int(decision_counts.get("REJECT", 0))

anomaly_count = (
    int((results["Anomaly_Status"] == "ANOMALY").sum())
    if "Anomaly_Status" in results.columns
    else 0
)

normal_count = len(results) - anomaly_count


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Screening Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Components",
        f"{len(results):,}",
    )

    col2.metric(
        "PASS",
        f"{pass_count:,}",
    )

    col3.metric(
        "INVESTIGATE",
        f"{investigate_count:,}",
    )

    col4.metric(
        "REJECT",
        f"{reject_count:,}",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # DECISION DISTRIBUTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Final Decision Distribution</div>',
        unsafe_allow_html=True,
    )

    decision_df = pd.DataFrame(
        {
            "Decision": ["PASS", "INVESTIGATE", "REJECT"],
            "Components": [
                pass_count,
                investigate_count,
                reject_count,
            ],
        }
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.bar_chart(
            decision_df.set_index("Decision")
        )

    with chart_col2:

        st.dataframe(
            decision_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # AI STATUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">AI Screening Status</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Normal Behaviour",
        f"{normal_count:,}",
    )

    c2.metric(
        "Anomalous Behaviour",
        f"{anomaly_count:,}",
    )

    if "Drift_Status" in results.columns:
        drift_normal = int(
            (results["Drift_Status"] == "NORMAL").sum()
        )
    else:
        drift_normal = 0

    c3.metric(
        "Normal Drift",
        f"{drift_normal:,}",
    )

    # --------------------------------------------------------
    # RISK SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Risk Score Summary</div>',
        unsafe_allow_html=True,
    )

    if "Risk_Score" in results.columns:

        risk = pd.to_numeric(
            results["Risk_Score"],
            errors="coerce",
        )

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Average Risk",
            f"{risk.mean():.3f}",
        )

        r2.metric(
            "Minimum",
            f"{risk.min():.3f}",
        )

        r3.metric(
            "Median",
            f"{risk.median():.3f}",
        )

        r4.metric(
            "Maximum",
            f"{risk.max():.3f}",
        )

        st.line_chart(
            risk.reset_index(drop=True)
        )


# ============================================================
# COMPONENT SCREENING
# ============================================================

elif page == "Component Screening":

    st.markdown(
        '<div class="section-title">Component-Level Screening</div>',
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Search Component ID",
        placeholder="Example: IC00514",
    )

    filtered = results.copy()

    if search.strip():

        filtered = filtered[
            filtered["Component_ID"]
            .astype(str)
            .str.contains(
                search.strip(),
                case=False,
                na=False,
            )
        ]

    decision_filter = st.multiselect(
        "Filter by Final Decision",
        ["PASS", "INVESTIGATE", "REJECT"],
        default=["PASS", "INVESTIGATE", "REJECT"],
    )

    if "Decision" in filtered.columns:

        filtered = filtered[
            filtered["Decision"].isin(decision_filter)
        ]

    st.write(
        f"Showing **{len(filtered):,}** components"
    )

    display_columns = [
        "Component_ID",
        "Lot_ID",
        "Component_Type",
        "Temperature_C",
        "Anomaly_Status",
        "Anomaly_Risk",
        "Drift_Status",
        "Drift_Risk",
        "Predicted_Specification_Status",
        "Risk_Score",
        "Risk_Percentage",
        "Decision",
    ]

    available_columns = [
        c for c in display_columns
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.download_button(
        label="Download Screening Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="final_screening_results.csv",
        mime="text/csv",
    )


# ============================================================
# AI ANOMALY ANALYSIS
# ============================================================

elif page == "AI Anomaly Analysis":

    st.markdown(
        '<div class="section-title">AI Anomaly Detection</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Normal",
        f"{normal_count:,}",
    )

    c2.metric(
        "Anomalous",
        f"{anomaly_count:,}",
    )

    if "Anomaly_Risk" in results.columns:

        anomaly_risk = pd.to_numeric(
            results["Anomaly_Risk"],
            errors="coerce",
        )

        c3.metric(
            "Maximum Anomaly Risk",
            f"{anomaly_risk.max():.3f}",
        )

    st.markdown("---")

    if "Anomaly_Status" in results.columns:

        anomaly_distribution = (
            results["Anomaly_Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Components")
        )

        st.bar_chart(
            anomaly_distribution.set_index("Status")
        )

    # --------------------------------------------------------
    # ANOMALY COMPONENTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Detected Anomalies</div>',
        unsafe_allow_html=True,
    )

    anomaly_columns = [
        "Component_ID",
        "Lot_ID",
        "Anomaly_Status",
        "Isolation_Status",
        "Isolation_Score",
        "Max_Z_Score",
        "Statistical_Status",
        "Statistical_Score",
        "Anomaly_Risk",
        "Decision",
    ]

    available = [
        c for c in anomaly_columns
        if c in results.columns
    ]

    anomalies = results[
        results["Anomaly_Status"] == "ANOMALY"
    ]

    st.dataframe(
        anomalies[available],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 168h PREDICTION
# ============================================================

elif page == "168h Prediction":

    st.markdown(
        '<div class="section-title">Predicted 168h Component Behaviour</div>',
        unsafe_allow_html=True,
    )

    component_ids = (
        results["Component_ID"]
        .astype(str)
        .tolist()
    )

    selected_id = st.selectbox(
        "Select Component",
        component_ids,
    )

    component = results[
        results["Component_ID"].astype(str)
        == selected_id
    ]

    if component.empty:
        st.warning("Component not found.")
        st.stop()

    row = component.iloc[0]

    # --------------------------------------------------------
    # CURRENT VS PREDICTED
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Parameter Prediction</div>',
        unsafe_allow_html=True,
    )

    parameters = [
        (
            "IDDQ",
            "Iddq_24h_uA",
            "Iddq_96h_uA",
            "Predicted_Iddq_168h_uA",
            "Iddq_Max_Limit_uA",
        ),
        (
            "Leakage",
            "Leakage_24h_uA",
            "Leakage_96h_uA",
            "Predicted_Leakage_168h_uA",
            "Leakage_Max_Limit_uA",
        ),
        (
            "Delay",
            "Delay_24h_ns",
            "Delay_96h_ns",
            "Predicted_Delay_168h_ns",
            "Delay_Max_Limit_ns",
        ),
    ]

    for name, col24, col96, pred_col, limit_col in parameters:

        st.write(f"### {name}")

        a, b, c, d = st.columns(4)

        if col24 in row.index:
            a.metric(
                "24h",
                f"{float(row[col24]):.3f}",
            )

        if col96 in row.index:
            b.metric(
                "96h",
                f"{float(row[col96]):.3f}",
            )

        if pred_col in row.index:
            c.metric(
                "Predicted 168h",
                f"{float(row[pred_col]):.3f}",
            )

        if limit_col in row.index:
            d.metric(
                "Maximum Limit",
                f"{float(row[limit_col]):.3f}",
            )

        graph_columns = [
            c for c in [col24, col96, pred_col]
            if c in row.index
        ]

        if graph_columns:

            graph_data = pd.DataFrame(
                {
                    "Stage": [
                        "24h",
                        "96h",
                        "Predicted 168h",
                    ][:len(graph_columns)],
                    "Value": [
                        float(row[c])
                        for c in graph_columns
                    ],
                }
            )

            st.line_chart(
                graph_data.set_index("Stage")
            )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Screening Result</div>',
        unsafe_allow_html=True,
    )

    status = row.get(
        "Decision",
        "UNKNOWN",
    )

    risk = row.get(
        "Risk_Percentage",
        row.get("Risk_Score", 0),
    )

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Final Decision",
        str(status),
    )

    s2.metric(
        "Risk",
        f"{float(risk):.2f}"
        + ("%" if "Risk_Percentage" in row.index else ""),
    )

    s3.metric(
        "Predicted Specification",
        str(
            row.get(
                "Predicted_Specification_Status",
                "UNKNOWN",
            )
        ),
    )

    if "Explanation" in row.index:
        st.info(
            str(row["Explanation"])
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.markdown(
        '<div class="section-title">AI Model Comparison</div>',
        unsafe_allow_html=True,
    )

    if metrics is None:
        st.warning(
            "Model metrics are unavailable."
        )
        st.stop()

    st.write(
        """
        Three regression models are evaluated for each
        monitored parameter. The model with the lowest
        validation MAE is selected.
        """
    )

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Selected Models</div>',
        unsafe_allow_html=True,
    )

    best_rows = []

    for parameter in metrics["Parameter"].unique():

        subset = metrics[
            metrics["Parameter"] == parameter
        ]

        best = subset.loc[
            subset["MAE"].idxmin()
        ]

        best_rows.append(
            {
                "Parameter": parameter,
                "Selected Model": best["Model"],
                "MAE": best["MAE"],
                "RMSE": best["RMSE"],
                "R2": best["R2"],
            }
        )

    best_df = pd.DataFrame(best_rows)

    st.dataframe(
        best_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # MODEL CHART
    # --------------------------------------------------------

    selected_parameter = st.selectbox(
        "Select Parameter",
        metrics["Parameter"].unique(),
    )

    parameter_metrics = metrics[
        metrics["Parameter"] == selected_parameter
    ].copy()

    st.markdown(
        "### MAE Comparison"
    )

    st.bar_chart(
        parameter_metrics.set_index("Model")["MAE"]
    )

    st.markdown(
        "### R² Comparison"
    )

    st.bar_chart(
        parameter_metrics.set_index("Model")["R2"]
    )


# ============================================================
# ENGINEERING SPECIFICATION
# ============================================================

elif page == "Engineering Specification":

    st.markdown(
        '<div class="section-title">Predicted Engineering Specification</div>',
        unsafe_allow_html=True,
    )

    if "Predicted_Specification_Status" in results.columns:

        specification_counts = (
            results[
                "Predicted_Specification_Status"
            ]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Components")
        )

        st.bar_chart(
            specification_counts.set_index("Status")
        )

        st.dataframe(
            specification_counts,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    st.markdown(
        """
        ### Screening Logic

        **PASS**  
        Predicted 168h behaviour remains within the
        engineering specification and AI risk is acceptable.

        **INVESTIGATE**  
        The component shows elevated AI risk or approaches
        a warning boundary and requires further inspection.

        **REJECT**  
        Predicted 168h behaviour exceeds the defined
        engineering specification.
        """
    )

    # --------------------------------------------------------
    # REJECTED COMPONENTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Predicted Specification Violations</div>',
        unsafe_allow_html=True,
    )

    rejected = results[
        results["Predicted_Specification_Status"]
        == "REJECT"
    ]

    reject_columns = [
        "Component_ID",
        "Predicted_Iddq_168h_uA",
        "Predicted_Leakage_168h_uA",
        "Predicted_Delay_168h_ns",
        "Iddq_Max_Limit_uA",
        "Leakage_Max_Limit_uA",
        "Delay_Max_Limit_ns",
        "Predicted_Specification_Status",
        "Decision",
    ]

    available_reject_columns = [
        c for c in reject_columns
        if c in rejected.columns
    ]

    st.dataframe(
        rejected[available_reject_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Assisted Adaptive Component Screening | "
    "V2 Predictive Burn-In Analysis"
)