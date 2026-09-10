from pathlib import Path
from io import BytesIO
import re

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


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
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "uploaded_data": None,
    "screening_result": None,
    "model_metrics": None,
    "analysis_mode": None,
    "domain": None,
    "domain_confidence": None,
    "domain_reason": None,
    "component_column": None,
    "parameter_columns": [],
    "screening_done": False,
    "uploaded_filename": None,
    "domain_selection": "Auto Detect",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_number(value, default=0.0):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return default


def percentage(value):
    return f"{safe_number(value) * 100:.1f}%"


def normalize_columns(data):
    """
    Standardize column names without destroying their meaning.
    """
    data = data.copy()

    cleaned = []

    for column in data.columns:
        name = str(column).strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")

        cleaned.append(name)

    data.columns = cleaned
    return data


def is_identifier_column(column_name):
    """
    Identify columns that are probably identifiers rather than measurements.
    """
    name = str(column_name).strip().lower()

    exact_identifier_names = {
        "id",
        "component_id",
        "device_id",
        "unit_id",
        "asset_id",
        "equipment_id",
        "machine_id",
        "part_id",
        "sample_id",
        "serial_number",
        "serial_no",
        "lot_id",
        "batch_id",
        "record_id",
    }

    if name in exact_identifier_names:
        return True

    identifier_patterns = [
        r"^id$",
        r".*_id$",
        r"^id_.*",
        r".*_number$",
        r".*_no$",
        r"^serial.*",
        r"^lot.*",
        r"^batch.*",
    ]

    return any(re.match(pattern, name) for pattern in identifier_patterns)


def is_non_measurement_column(column_name):
    """
    Identify common non-measurement columns.
    """
    name = str(column_name).strip().lower()

    if is_identifier_column(name):
        return True

    non_measurement_keywords = [
        "name",
        "type",
        "category",
        "class",
        "label",
        "status",
        "decision",
        "description",
        "comment",
        "remarks",
        "date",
        "datetime",
        "timestamp",
        "time",
    ]

    return any(keyword == name or name.startswith(keyword + "_")
               for keyword in non_measurement_keywords)


def find_component_column(data):
    """
    Automatically find the most likely component/device identifier.
    """
    preferred = [
        "Component_ID",
        "Device_ID",
        "Unit_ID",
        "Asset_ID",
        "Equipment_ID",
        "Machine_ID",
        "Part_ID",
        "Sample_ID",
        "Serial_Number",
        "Serial_No",
        "Lot_ID",
        "Batch_ID",
    ]

    for column in preferred:
        if column in data.columns:
            return column

    for column in data.columns:
        if is_identifier_column(column):
            return column

    return None


def find_numeric_parameters(data):
    """
    Find numeric measurement columns while excluding obvious identifiers.
    """
    parameters = []

    for column in data.columns:
        if is_non_measurement_column(column):
            continue

        numeric_values = pd.to_numeric(data[column], errors="coerce")

        if numeric_values.notna().sum() < max(5, int(len(data) * 0.05)):
            continue

        unique_count = numeric_values.nunique(dropna=True)

        if unique_count <= 1:
            continue

        parameters.append(column)

    return parameters


# ============================================================
# ELECTRONICS SCHEMA DETECTION
# ============================================================

def detect_electronics_v2_schema(data):
    required_columns = {
        "Component_ID",
        "Lot_ID",
        "Component_Type",
        "Temperature_C",
        "Iddq_0h_uA",
        "Iddq_24h_uA",
        "Iddq_96h_uA",
        "Iddq_168h_uA",
        "Leakage_0h_uA",
        "Leakage_24h_uA",
        "Leakage_96h_uA",
        "Leakage_168h_uA",
        "Delay_0h_ns",
        "Delay_24h_ns",
        "Delay_96h_ns",
        "Delay_168h_ns",
        "Iddq_Max_Limit_uA",
        "Leakage_Max_Limit_uA",
        "Delay_Max_Limit_ns",
    }

    return required_columns.issubset(set(data.columns))


# ============================================================
# DOMAIN DETECTION
# ============================================================

DOMAIN_SIGNATURES = {
    "Electronics": [
        "iddq",
        "leakage",
        "delay",
        "voltage",
        "current",
        "frequency",
        "power",
        "capacitance",
        "resistance",
        "temperature",
        "clock",
        "jitter",
        "rise_time",
        "fall_time",
        "transistor",
        "gate",
        "signal",
    ],

    "Mechanical": [
        "vibration",
        "rpm",
        "torque",
        "pressure",
        "force",
        "stress",
        "strain",
        "displacement",
        "velocity",
        "acceleration",
        "bearing",
        "shaft",
        "load",
        "wear",
        "friction",
    ],

    "Automotive": [
        "engine",
        "rpm",
        "torque",
        "brake",
        "vehicle",
        "speed",
        "fuel",
        "coolant",
        "oil",
        "throttle",
        "gear",
        "wheel",
        "tire",
        "battery",
        "motor",
        "temperature",
    ],

    "Manufacturing": [
        "production",
        "cycle",
        "machine",
        "tool",
        "defect",
        "yield",
        "scrap",
        "quality",
        "process",
        "line",
        "batch",
        "pressure",
        "temperature",
        "flow",
        "speed",
    ],

    "Energy": [
        "energy",
        "power",
        "voltage",
        "current",
        "frequency",
        "grid",
        "load",
        "transformer",
        "generator",
        "solar",
        "wind",
        "battery",
        "soc",
        "state_of_charge",
    ],

    "Aerospace": [
        "altitude",
        "airspeed",
        "flight",
        "engine",
        "fuel",
        "pressure",
        "temperature",
        "vibration",
        "acceleration",
        "pitch",
        "roll",
        "yaw",
        "thrust",
    ],

    "Medical Equipment": [
        "spo2",
        "ecg",
        "heart_rate",
        "blood_pressure",
        "respiratory",
        "oxygen",
        "pulse",
        "infusion",
        "pump",
        "flow",
        "pressure",
        "temperature",
    ],
}


def calculate_domain_scores(data):
    """
    Calculate domain evidence from column names.

    This is intentionally transparent:
    domain detection is based on measurable column-name evidence,
    not an unsupported black-box claim.
    """

    normalized_columns = [
        str(column).lower().replace("-", "_").replace(" ", "_")
        for column in data.columns
    ]

    scores = {}

    for domain, keywords in DOMAIN_SIGNATURES.items():
        score = 0
        matched = []

        for column in normalized_columns:
            for keyword in keywords:
                keyword = keyword.lower()

                if keyword in column:
                    score += 1
                    matched.append(column)
                    break

        scores[domain] = {
            "score": score,
            "matched_columns": sorted(set(matched)),
        }

    return scores


def detect_domain(data):
    """
    Automatically detect the most likely application domain.
    """

    # Strong deterministic detection for the current specialized pipeline.
    if detect_electronics_v2_schema(data):
        return (
            "Electronics",
            "High",
            "Detected the complete electronics burn-in measurement structure "
            "including IDDQ, Leakage, Delay and engineering limits."
        )

    scores = calculate_domain_scores(data)

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    if not ranked:
        return (
            "General / Unknown",
            "Low",
            "No measurable domain-specific evidence was identified."
        )

    best_domain, best_info = ranked[0]

    second_score = ranked[1][1]["score"] if len(ranked) > 1 else 0
    best_score = best_info["score"]

    if best_score == 0:
        return (
            "General / Unknown",
            "Low",
            "No strong domain-specific measurement names were detected."
        )

    if best_score >= 4 and best_score >= second_score + 2:
        confidence = "High"
    elif best_score >= 2 and best_score > second_score:
        confidence = "Medium"
    else:
        confidence = "Low"

    matched_columns = best_info["matched_columns"][:8]

    reason = (
        f"Detected {best_score} domain-related measurement indicators: "
        + ", ".join(matched_columns)
        + "."
    )

    if confidence == "Low":
        return (
            "General / Unknown",
            "Low",
            "The available column names do not provide enough evidence "
            "for a reliable domain-specific classification."
        )

    return best_domain, confidence, reason


# ============================================================
# DECISION HELPERS
# ============================================================

def decision_message(decision):
    messages = {
        "PASS": "No significant abnormality detected.",
        "INVESTIGATE": "Potential risk detected. Further engineering investigation is recommended.",
        "REJECT": "High-risk behaviour detected. The component requires rejection or immediate engineering review.",
    }

    return messages.get(
        str(decision).upper(),
        "Decision generated from the available AI and engineering evidence."
    )


# ============================================================
# UNIVERSAL AI SCREENING
# ============================================================

def run_universal_screening(data):
    """
    Domain-independent AI screening engine.

    Used when:
    - domain is not Electronics
    - domain is unknown
    - specialized domain pipeline is not available

    It detects abnormal behaviour from available numeric measurements.
    """

    df = data.copy()

    parameter_columns = find_numeric_parameters(df)

    if not parameter_columns:
        raise ValueError(
            "No usable numeric measurement parameters were found in the uploaded data."
        )

    feature_data = df[parameter_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    feature_data = feature_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_data = feature_data.fillna(
        feature_data.median(numeric_only=True)
    )

    feature_data = feature_data.fillna(0)

    if len(feature_data) < 5:
        raise ValueError(
            "At least 5 valid records are required for AI screening."
        )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)

    contamination = min(
        max(0.05, 0.15),
        0.35,
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(scaled_features)
    decision_scores = model.decision_function(scaled_features)

    isolation_risk = np.clip(
        0.5 - decision_scores,
        0,
        1,
    )

    z_scores = np.abs(
        (
            feature_data
            - feature_data.mean()
        )
        /
        feature_data.std(ddof=0).replace(
            0,
            np.nan,
        )
    ).fillna(0)

    max_z_score = z_scores.max(axis=1)

    statistical_risk = np.clip(
        max_z_score / 5,
        0,
        1,
    )

    anomaly_risk = (
        0.5 * isolation_risk
        + 0.5 * statistical_risk
    )

    anomaly_status = np.where(
        (predictions == -1) | (max_z_score >= 3),
        "ANOMALY",
        "NORMAL",
    )

    risk_score = anomaly_risk

    decisions = np.select(
        [
            risk_score >= 0.70,
            risk_score >= 0.25,
        ],
        [
            "REJECT",
            "INVESTIGATE",
        ],
        default="PASS",
    )

    result = df.copy()

    result["Isolation_Prediction"] = predictions
    result["Isolation_Status"] = np.where(
        predictions == -1,
        "ANOMALY",
        "NORMAL",
    )
    result["Isolation_Score"] = isolation_risk
    result["Max_Z_Score"] = max_z_score
    result["Statistical_Score"] = statistical_risk
    result["Anomaly_Risk"] = anomaly_risk
    result["Anomaly_Status"] = anomaly_status
    result["Drift_Risk"] = 0.0
    result["Drift_Status"] = "NOT_AVAILABLE"
    result["Risk_Score"] = risk_score
    result["Risk_Percentage"] = risk_score * 100
    result["Final_Decision"] = decisions

    explanations = []

    for index in result.index:
        status = result.loc[index, "Anomaly_Status"]
        risk = safe_number(result.loc[index, "Anomaly_Risk"])

        if status == "ANOMALY":
            explanation = (
                f"AI detected abnormal behaviour across the uploaded "
                f"measurement profile. Anomaly risk = {risk:.2f}."
            )
        else:
            explanation = (
                f"No significant abnormal behaviour detected in the "
                f"available measurements. Risk = {risk:.2f}."
            )

        explanations.append(explanation)

    result["AI_Explanation"] = explanations

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Records Processed",
                "Measurement Parameters",
                "Detected Anomalies",
                "Normal Records",
                "Mean Risk",
                "Maximum Risk",
            ],
            "Value": [
                len(result),
                len(parameter_columns),
                int((result["Anomaly_Status"] == "ANOMALY").sum()),
                int((result["Anomaly_Status"] == "NORMAL").sum()),
                round(result["Anomaly_Risk"].mean(), 4),
                round(result["Anomaly_Risk"].max(), 4),
            ],
        }
    )

    return result, metrics, parameter_columns


# ============================================================
# ELECTRONICS SPECIALIZED SCREENING
# ============================================================

def run_electronics_screening(data):
    """
    Run the existing specialized Electronics AI pipeline.
    """

    upload_path = Path("data/uploaded_electronics_data.csv")
    upload_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        upload_path,
        index=False,
    )

    try:
        from main import ScreeningPipeline
    except Exception as error:
        raise RuntimeError(
            "The Electronics AI Pipeline could not be loaded. "
            f"Details: {error}"
        )

    pipeline = ScreeningPipeline()

    pipeline_result = pipeline.run(
        file_path=str(upload_path)
    )

    if isinstance(pipeline_result, tuple):
        result = pipeline_result[0]

        metrics = (
            pipeline_result[1]
            if len(pipeline_result) > 1
            else None
        )
    else:
        result = pipeline_result
        metrics = None

    result = result.copy()

    # --------------------------------------------------------
    # Use predicted 168h specification for early screening
    # when available.
    # --------------------------------------------------------

    if "Predicted_Specification_Status" in result.columns:

        predicted_status = (
            result["Predicted_Specification_Status"]
            .fillna("PASS")
            .astype(str)
            .str.upper()
        )

        result["Final_Decision"] = np.select(
            [
                predicted_status == "REJECT",
                predicted_status == "INVESTIGATE",
            ],
            [
                "REJECT",
                "INVESTIGATE",
            ],
            default="PASS",
        )

    elif "Decision" in result.columns:

        result["Final_Decision"] = (
            result["Decision"]
            .fillna("PASS")
            .astype(str)
            .str.upper()
        )

    else:
        result["Final_Decision"] = "PASS"

    # --------------------------------------------------------
    # User-friendly explanation
    # --------------------------------------------------------

    explanations = []

    for index in result.index:

        decision = str(
            result.loc[index, "Final_Decision"]
        ).upper()

        anomaly_status = str(
            result.loc[index, "Anomaly_Status"]
        ).upper() if "Anomaly_Status" in result.columns else "NORMAL"

        anomaly_risk = safe_number(
            result.loc[index, "Anomaly_Risk"]
        ) if "Anomaly_Risk" in result.columns else 0

        predicted_spec = str(
            result.loc[index, "Predicted_Specification_Status"]
        ).upper() if "Predicted_Specification_Status" in result.columns else "PASS"

        reasons = []

        if predicted_spec == "REJECT":
            reasons.append(
                "Predicted future value exceeds the engineering specification."
            )

        elif predicted_spec == "INVESTIGATE":
            reasons.append(
                "Predicted future value is approaching the engineering limit."
            )

        if anomaly_status == "ANOMALY":
            reasons.append(
                f"Abnormal behaviour detected by AI "
                f"(anomaly risk {anomaly_risk:.2f})."
            )

        if not reasons:
            reasons.append(
                "Behaviour remains within the evaluated engineering and AI risk boundaries."
            )

        explanation = " ".join(reasons)

        explanations.append(explanation)

    result["AI_Explanation"] = explanations

    return result, metrics


# ============================================================
# SCREENING ROUTER
# ============================================================

def execute_screening(data, selected_domain):
    """
    Decide which AI engine should process the uploaded data.
    """

    # --------------------------------------------------------
    # Auto Detect
    # --------------------------------------------------------

    if selected_domain == "Auto Detect":

        detected_domain, confidence, reason = detect_domain(data)

        st.session_state.domain = detected_domain
        st.session_state.domain_confidence = confidence
        st.session_state.domain_reason = reason

        # Specialized Electronics route
        if detected_domain == "Electronics" and detect_electronics_v2_schema(data):

            result, metrics = run_electronics_screening(data)

            return (
                result,
                metrics,
                "Electronics AI Pipeline",
                detected_domain,
                confidence,
                reason,
            )

        # Generic universal route
        result, metrics, _ = run_universal_screening(data)

        return (
            result,
            metrics,
            "Universal AI Screening Engine",
            detected_domain,
            confidence,
            reason,
        )

    # --------------------------------------------------------
    # Manual Electronics
    # --------------------------------------------------------

    if selected_domain == "Electronics":

        if not detect_electronics_v2_schema(data):

            st.warning(
                "The uploaded data does not match the specialized "
                "Electronics burn-in structure. The Universal AI Screening "
                "Engine will be used instead to avoid applying an incorrect "
                "electronics model."
            )

            result, metrics, _ = run_universal_screening(data)

            return (
                result,
                metrics,
                "Universal AI Screening Engine",
                "Electronics",
                "Low",
                "Electronics was manually selected, but the specialized "
                "burn-in schema was not available. Universal screening was used safely.",
            )

        result, metrics = run_electronics_screening(data)

        return (
            result,
            metrics,
            "Electronics AI Pipeline",
            "Electronics",
            "High",
            "User selected Electronics and the specialized burn-in structure was detected.",
        )

    # --------------------------------------------------------
    # Other domains
    # --------------------------------------------------------

    result, metrics, _ = run_universal_screening(data)

    return (
        result,
        metrics,
        "Universal AI Screening Engine",
        selected_domain,
        "User Selected",
        f"{selected_domain} was selected manually. "
        "The universal domain-independent AI screening engine was used.",
    )


# ============================================================
# COMPONENT HISTORY
# ============================================================

def build_component_history(row):
    """
    Build a time-series style history for component inspection.
    """

    history = []

    electronics_groups = {
        "IDDQ (uA)": [
            ("0h", "Iddq_0h_uA"),
            ("24h", "Iddq_24h_uA"),
            ("96h", "Iddq_96h_uA"),
            ("168h Actual", "Iddq_168h_uA"),
            ("168h Predicted", "Predicted_Iddq_168h_uA"),
        ],
        "Leakage (uA)": [
            ("0h", "Leakage_0h_uA"),
            ("24h", "Leakage_24h_uA"),
            ("96h", "Leakage_96h_uA"),
            ("168h Actual", "Leakage_168h_uA"),
            ("168h Predicted", "Predicted_Leakage_168h_uA"),
        ],
        "Delay (ns)": [
            ("0h", "Delay_0h_ns"),
            ("24h", "Delay_24h_ns"),
            ("96h", "Delay_96h_ns"),
            ("168h Actual", "Delay_168h_ns"),
            ("168h Predicted", "Predicted_Delay_168h_ns"),
        ],
    }

    found_electronics = False

    for parameter, points in electronics_groups.items():

        values = []

        for label, column in points:

            if column in row.index:

                value = pd.to_numeric(
                    pd.Series([row[column]]),
                    errors="coerce",
                ).iloc[0]

                if pd.notna(value):
                    values.append(
                        {
                            "Parameter": parameter,
                            "Time": label,
                            "Value": float(value),
                        }
                    )

        if values:
            found_electronics = True
            history.extend(values)

    if found_electronics:
        return pd.DataFrame(history)

    # Generic fallback
    generic_values = []

    for column, value in row.items():

        numeric_value = pd.to_numeric(
            pd.Series([value]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(numeric_value):

            if not is_non_measurement_column(column):

                generic_values.append(
                    {
                        "Parameter": column,
                        "Time": "Current",
                        "Value": float(numeric_value),
                    }
                )

    return pd.DataFrame(generic_values)


# ============================================================
# HOME PAGE
# ============================================================

def render_home():

    st.title("🔬 AI-Assisted Adaptive Component Screening")

    st.markdown(
        """
        ### Intelligent screening for component and equipment health

        This platform analyzes uploaded measurement data using an
        **adaptive AI screening architecture**.

        **Upload your own data → detect the domain → analyze abnormal behaviour
        → estimate risk → inspect individual components.**
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "AI Architecture",
            "Adaptive",
        )

    with col2:
        st.metric(
            "Domain Detection",
            "Automatic",
        )

    with col3:
        st.metric(
            "Screening",
            "AI Assisted",
        )

    with col4:
        st.metric(
            "Decision",
            "Risk Based",
        )

    st.divider()

    st.subheader("How the platform works")

    steps = [
        ("01", "Upload Data", "Upload CSV or Excel measurement data."),
        ("02", "Domain Detection", "Select a domain or allow AI to identify it automatically."),
        ("03", "Data Validation", "Check the structure and measurement quality."),
        ("04", "AI Analysis", "Detect abnormal behaviour and calculate risk."),
        ("05", "Future Screening", "Use a specialized pipeline when a validated domain model exists."),
        ("06", "Decision", "PASS, INVESTIGATE or REJECT."),
        ("07", "Component Intelligence", "Inspect individual component behaviour."),
    ]

    for number, title, description in steps:

        with st.container(border=True):

            c1, c2 = st.columns([1, 5])

            with c1:
                st.markdown(f"### {number}")

            with c2:
                st.markdown(f"**{title}**")
                st.caption(description)

    st.info(
        "💡 If you do not know the domain, simply keep **Auto Detect** selected. "
        "The application will inspect the uploaded measurement structure and "
        "identify the most likely domain."
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

def render_upload():

    st.title("📂 Upload Measurement Data")

    st.write(
        "Upload your own CSV or Excel dataset. The platform will analyze "
        "the structure before selecting the appropriate AI screening route."
    )

    uploaded_file = st.file_uploader(
        "Choose a measurement dataset",
        type=["csv", "xlsx", "xls"],
        help="CSV, XLSX and XLS files are supported.",
    )

    st.subheader("Domain")

    domain_options = [
        "Auto Detect",
        "Electronics",
        "Mechanical",
        "Automotive",
        "Manufacturing",
        "Energy",
        "Aerospace",
        "Medical Equipment",
        "Custom",
    ]

    selected_domain = st.selectbox(
        "Select application domain",
        domain_options,
        index=domain_options.index(
            st.session_state.get(
                "domain_selection",
                "Auto Detect",
            )
        ),
        help=(
            "Choose a domain manually or keep Auto Detect to let the "
            "application identify the most likely domain from your dataset."
        ),
    )

    st.session_state.domain_selection = selected_domain

    if selected_domain == "Auto Detect":

        st.info(
            "🤖 Auto Detect is enabled. The application will inspect the "
            "measurement columns and determine the most likely domain."
        )

    else:

        st.info(
            f"📌 Manual domain selection: **{selected_domain}**"
        )

    if uploaded_file is None:
        return

    try:

        if uploaded_file.name.lower().endswith(".csv"):
            data = pd.read_csv(uploaded_file)

        else:
            data = pd.read_excel(uploaded_file)

        data = data.dropna(
            axis=0,
            how="all",
        )

        data = data.dropna(
            axis=1,
            how="all",
        )

        data = normalize_columns(data)

    except Exception as error:

        st.error(
            f"Unable to read the uploaded file: {error}"
        )
        return

    if data.empty:

        st.error(
            "The uploaded file does not contain usable records."
        )
        return

    st.session_state.uploaded_data = data
    st.session_state.uploaded_filename = uploaded_file.name
    st.session_state.screening_done = False
    st.session_state.screening_result = None

    st.success(
        f"Successfully loaded **{len(data):,} records** from "
        f"**{uploaded_file.name}**."
    )

    st.divider()

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Records",
            f"{len(data):,}",
        )

    with c2:
        st.metric(
            "Columns",
            f"{len(data.columns):,}",
        )

    with c3:
        st.metric(
            "Numeric Columns",
            f"{len(data.select_dtypes(include=np.number).columns):,}",
        )

    with c4:
        component_column = find_component_column(data)

        st.metric(
            "Component ID",
            component_column if component_column else "Not detected",
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Preview automatic domain detection
    # --------------------------------------------------------

    detected_domain, confidence, reason = detect_domain(data)

    st.subheader("🤖 Domain Intelligence")

    d1, d2 = st.columns(2)

    with d1:

        st.markdown(
            f"**Detected Domain:** `{detected_domain}`"
        )

        st.markdown(
            f"**Confidence:** `{confidence}`"
        )

    with d2:

        st.caption(
            reason
        )

    # --------------------------------------------------------
    # Run button
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🚀 Start AI Screening",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.analysis_mode = None

        with st.spinner(
            "Analyzing dataset and running AI screening..."
        ):

            try:

                (
                    result,
                    metrics,
                    analysis_mode,
                    final_domain,
                    final_confidence,
                    final_reason,
                ) = execute_screening(
                    data,
                    selected_domain,
                )

                st.session_state.screening_result = result
                st.session_state.model_metrics = metrics
                st.session_state.analysis_mode = analysis_mode
                st.session_state.domain = final_domain
                st.session_state.domain_confidence = final_confidence
                st.session_state.domain_reason = final_reason
                st.session_state.component_column = find_component_column(
                    result
                )
                st.session_state.parameter_columns = find_numeric_parameters(
                    data
                )
                st.session_state.screening_done = True

                st.success(
                    "AI screening completed successfully."
                )

            except Exception as error:

                st.error(
                    f"Screening failed: {error}"
                )

                st.exception(error)


# ============================================================
# SCREENING PAGE
# ============================================================

def render_screening():

    st.title("🧠 AI Screening")

    result = st.session_state.screening_result

    if result is None:

        st.info(
            "Upload a dataset and start AI screening first."
        )
        return

    analysis_mode = st.session_state.analysis_mode
    domain = st.session_state.domain
    confidence = st.session_state.domain_confidence
    reason = st.session_state.domain_reason

    st.subheader("AI Routing")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Detected / Selected Domain",
            domain or "Unknown",
        )

    with c2:
        st.metric(
            "Confidence",
            confidence or "N/A",
        )

    with c3:
        st.metric(
            "AI Engine",
            analysis_mode or "N/A",
        )

    st.caption(
        reason or ""
    )

    st.divider()

    # --------------------------------------------------------
    # Decision metrics
    # --------------------------------------------------------

    decisions = (
        result["Final_Decision"]
        if "Final_Decision" in result.columns
        else pd.Series(dtype=str)
    )

    pass_count = int(
        (decisions == "PASS").sum()
    )

    investigate_count = int(
        (decisions == "INVESTIGATE").sum()
    )

    reject_count = int(
        (decisions == "REJECT").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total",
            f"{len(result):,}",
        )

    with c2:
        st.metric(
            "PASS",
            f"{pass_count:,}",
        )

    with c3:
        st.metric(
            "INVESTIGATE",
            f"{investigate_count:,}",
        )

    with c4:
        st.metric(
            "REJECT",
            f"{reject_count:,}",
        )

    st.divider()

    st.subheader("Screening Results")

    display_columns = []

    preferred_columns = [
        "Component_ID",
        "Device_ID",
        "Unit_ID",
        "Component_Type",
        "Anomaly_Status",
        "Anomaly_Risk",
        "Drift_Status",
        "Drift_Risk",
        "Predicted_Specification_Status",
        "Risk_Score",
        "Risk_Percentage",
        "Final_Decision",
        "AI_Explanation",
    ]

    for column in preferred_columns:

        if column in result.columns:
            display_columns.append(column)

    if not display_columns:
        display_columns = list(result.columns)

    st.dataframe(
        result[display_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RESULTS PAGE
# ============================================================

def render_results():

    st.title("📊 Results Dashboard")

    result = st.session_state.screening_result

    if result is None:

        st.info(
            "No screening results available yet."
        )
        return

    st.subheader("Final Decision Distribution")

    decisions = (
        result["Final_Decision"]
        .value_counts()
        .reindex(
            [
                "PASS",
                "INVESTIGATE",
                "REJECT",
            ],
            fill_value=0,
        )
    )

    st.bar_chart(
        decisions
    )

    st.divider()

    if "Risk_Score" in result.columns:

        st.subheader("Risk Analysis")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Mean Risk",
                f"{result['Risk_Score'].mean():.3f}",
            )

        with c2:
            st.metric(
                "Maximum Risk",
                f"{result['Risk_Score'].max():.3f}",
            )

        with c3:
            st.metric(
                "High Risk Records",
                f"{(result['Risk_Score'] >= 0.70).sum():,}",
            )

        st.line_chart(
            result["Risk_Score"].reset_index(
                drop=True
            )
        )

    st.divider()

    if "Anomaly_Status" in result.columns:

        st.subheader("AI Anomaly Detection")

        anomaly_counts = (
            result["Anomaly_Status"]
            .value_counts()
        )

        st.bar_chart(
            anomaly_counts
        )


# ============================================================
# COMPONENT INTELLIGENCE
# ============================================================

def render_component_inspection():

    st.title("🔎 Component Intelligence")

    result = st.session_state.screening_result

    if result is None:

        st.info(
            "Run AI screening before opening Component Intelligence."
        )
        return

    component_column = st.session_state.component_column

    if component_column is None:

        st.warning(
            "A unique component/device identifier was not detected. "
            "Individual component selection is unavailable for this dataset."
        )

        return

    component_values = (
        result[component_column]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )

    if not component_values:
        st.warning(
            "No component identifiers were found."
        )
        return

    selected_component = st.selectbox(
        "Select Component / Device",
        component_values,
    )

    row_data = result[
        result[component_column].astype(str)
        == selected_component
    ]

    if row_data.empty:
        st.warning(
            "Selected component was not found."
        )
        return

    row = row_data.iloc[0]

    decision = str(
        row.get(
            "Final_Decision",
            "N/A",
        )
    )

    risk = safe_number(
        row.get(
            "Risk_Score",
            row.get(
                "Anomaly_Risk",
                0,
            ),
        )
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Final Decision",
            decision,
        )

    with c2:

        st.metric(
            "Risk Score",
            f"{risk:.3f}",
        )

    with c3:

        st.metric(
            "Risk",
            f"{risk * 100:.1f}%",
        )

    st.info(
        decision_message(decision)
    )

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    st.subheader("Component Identity")

    identity_columns = [
        component_column,
        "Component_Type",
        "Lot_ID",
        "Device_Type",
        "Equipment_Type",
        "Machine_Type",
    ]

    identity = {}

    for column in identity_columns:

        if column in row.index:

            identity[column] = row[column]

    if identity:

        identity_df = pd.DataFrame(
            [
                {
                    "Property": key,
                    "Value": value,
                }
                for key, value in identity.items()
            ]
        )

        st.dataframe(
            identity_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Measurement details
    # --------------------------------------------------------

    st.subheader("Measurement Details")

    measurement_columns = []

    for column in row.index:

        if column in identity_columns:
            continue

        numeric_value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(numeric_value):
            measurement_columns.append(column)

    if measurement_columns:

        measurement_df = pd.DataFrame(
            {
                "Parameter": measurement_columns,
                "Value": [
                    row[column]
                    for column in measurement_columns
                ],
            }
        )

        st.dataframe(
            measurement_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_columns = [
        column
        for column in row.index
        if "Predicted" in str(column)
    ]

    if prediction_columns:

        st.subheader("🔮 AI Predicted Values")

        prediction_df = pd.DataFrame(
            {
                "Parameter": prediction_columns,
                "Predicted Value": [
                    row[column]
                    for column in prediction_columns
                ],
            }
        )

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Engineering specifications
    # --------------------------------------------------------

    specification_columns = [
        column
        for column in row.index
        if (
            "Limit" in str(column)
            or "Specification" in str(column)
        )
    ]

    if specification_columns:

        st.subheader("📏 Engineering Specifications")

        specification_df = pd.DataFrame(
            {
                "Parameter": specification_columns,
                "Value": [
                    row[column]
                    for column in specification_columns
                ],
            }
        )

        st.dataframe(
            specification_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # AI anomaly analysis
    # --------------------------------------------------------

    st.subheader("🤖 AI Anomaly Analysis")

    if "Anomaly_Status" in row.index:

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Anomaly Status",
                str(row["Anomaly_Status"]),
            )

        with c2:

            st.metric(
                "Anomaly Risk",
                f"{safe_number(row.get('Anomaly_Risk', 0)):.3f}",
            )

        with c3:

            st.metric(
                "Max Z-Score",
                f"{safe_number(row.get('Max_Z_Score', 0)):.2f}",
            )

    # --------------------------------------------------------
    # Drift analysis
    # --------------------------------------------------------

    drift_columns = [
        column
        for column in row.index
        if "Drift" in str(column)
    ]

    if drift_columns:

        st.subheader("📈 Drift Analysis")

        drift_df = pd.DataFrame(
            {
                "Metric": drift_columns,
                "Value": [
                    row[column]
                    for column in drift_columns
                ],
            }
        )

        st.dataframe(
            drift_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Trend analysis
    # --------------------------------------------------------

    history = build_component_history(row)

    if not history.empty:

        st.subheader("📊 Component Behaviour Trend")

        parameters = history["Parameter"].unique()

        for parameter in parameters:

            parameter_history = history[
                history["Parameter"] == parameter
            ].copy()

            if len(parameter_history) >= 2:

                chart_data = parameter_history[
                    ["Time", "Value"]
                ].set_index("Time")

                st.markdown(
                    f"**{parameter}**"
                )

                st.line_chart(
                    chart_data
                )

    # --------------------------------------------------------
    # AI explanation
    # --------------------------------------------------------

    st.subheader("💡 AI Explanation")

    explanation = row.get(
        "AI_Explanation",
        "No explanation was generated.",
    )

    st.info(
        str(explanation)
    )

    # --------------------------------------------------------
    # Complete record
    # --------------------------------------------------------

    with st.expander(
        "View Complete Component Record"
    ):

        st.dataframe(
            pd.DataFrame(
                [row]
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Individual report
    # --------------------------------------------------------

    component_report = pd.DataFrame(
        [row]
    )

    report_bytes = component_report.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Component Report",
        data=report_bytes,
        file_name=f"{selected_component}_screening_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# AI INTELLIGENCE PAGE
# ============================================================

def render_ai_intelligence():

    st.title("🧠 AI Intelligence")

    result = st.session_state.screening_result

    if result is None:

        st.info(
            "Run AI screening first."
        )
        return

    domain = st.session_state.domain
    confidence = st.session_state.domain_confidence
    reason = st.session_state.domain_reason

    st.subheader("Domain Intelligence")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Domain",
            domain or "General / Unknown",
        )

        st.metric(
            "Confidence",
            confidence or "N/A",
        )

    with c2:

        st.write(
            "**Detection reasoning**"
        )

        st.info(
            reason or "No additional reasoning available."
        )

    st.divider()

    st.subheader("AI Engine")

    st.write(
        f"**{st.session_state.analysis_mode}**"
    )

    if st.session_state.analysis_mode == "Electronics AI Pipeline":

        st.success(
            "The uploaded dataset matched the validated Electronics "
            "burn-in structure. The specialized Electronics AI pipeline "
            "was selected."
        )

    else:

        st.info(
            "A domain-independent AI screening engine was used. "
            "The system intentionally avoids applying a specialized "
            "domain model when the required validated structure is not available."
        )

    # --------------------------------------------------------
    # Anomaly intelligence
    # --------------------------------------------------------

    if "Anomaly_Risk" in result.columns:

        st.subheader("Anomaly Intelligence")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Mean Anomaly Risk",
                f"{result['Anomaly_Risk'].mean():.3f}",
            )

        with c2:

            st.metric(
                "Maximum Anomaly Risk",
                f"{result['Anomaly_Risk'].max():.3f}",
            )

        with c3:

            st.metric(
                "Anomalies",
                f"{(result['Anomaly_Status'] == 'ANOMALY').sum():,}"
                if "Anomaly_Status" in result.columns
                else "N/A",
            )

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

    metrics = st.session_state.model_metrics

    if metrics is not None:

        st.subheader("Model / Screening Metrics")

        if isinstance(metrics, pd.DataFrame):

            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True,
            )

        elif isinstance(metrics, dict):

            st.json(
                metrics
            )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    if "Risk_Score" in result.columns:

        st.subheader("Risk Distribution")

        st.line_chart(
            result["Risk_Score"].reset_index(
                drop=True
            )
        )


# ============================================================
# REPORTS PAGE
# ============================================================

def render_reports():

    st.title("📑 Reports")

    result = st.session_state.screening_result

    if result is None:

        st.info(
            "Run AI screening before generating reports."
        )
        return

    st.subheader("Complete Screening Report")

    csv_data = result.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Complete Screening Results",
        data=csv_data,
        file_name="ai_screening_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

    metrics = st.session_state.model_metrics

    if isinstance(metrics, pd.DataFrame):

        metrics_csv = metrics.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download AI Metrics",
            data=metrics_csv,
            file_name="ai_model_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.subheader("Report Summary")

    decisions = (
        result["Final_Decision"]
        .value_counts()
        .reindex(
            [
                "PASS",
                "INVESTIGATE",
                "REJECT",
            ],
            fill_value=0,
        )
    )

    summary = pd.DataFrame(
        {
            "Decision": decisions.index,
            "Count": decisions.values,
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔬 AI Screening")

    st.caption(
        "Adaptive Component Intelligence Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Upload Data",
            "AI Screening",
            "Results",
            "Component Intelligence",
            "AI Intelligence",
            "Reports",
        ],
    )

    st.divider()

    if st.session_state.uploaded_data is not None:

        st.markdown("### Current Dataset")

        st.caption(
            st.session_state.uploaded_filename
            or "Uploaded dataset"
        )

        st.caption(
            f"{len(st.session_state.uploaded_data):,} records"
        )

        if st.session_state.domain:

            st.caption(
                f"Domain: {st.session_state.domain}"
            )

        if st.session_state.analysis_mode:

            st.caption(
                f"Engine: {st.session_state.analysis_mode}"
            )

    else:

        st.caption(
            "No dataset uploaded."
        )


# ============================================================
# PAGE ROUTER
# ============================================================

if page == "Home":

    render_home()

elif page == "Upload Data":

    render_upload()

elif page == "AI Screening":

    render_screening()

elif page == "Results":

    render_results()

elif page == "Component Intelligence":

    render_component_inspection()

elif page == "AI Intelligence":

    render_ai_intelligence()

elif page == "Reports":

    render_reports()