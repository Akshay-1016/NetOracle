"""
NetOracle - Real Network Attack Forecasting Dashboard
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import NetOracleInference
from src.mitre_mapper import MITREMapper
from src.explainer import WorldModelExplainer


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="NetOracle",
    page_icon="🛡️",
    layout="wide",
)


st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
}

.main-title {
    font-size: 2.6rem;
    font-weight: 800;
}

.subtitle {
    color: #9ca3af;
    margin-bottom: 1.5rem;
}

.critical-box {
    border-left: 5px solid #ef4444;
    padding: 16px;
    background: rgba(239,68,68,0.08);
    border-radius: 8px;
}

.safe-box {
    border-left: 5px solid #22c55e;
    padding: 16px;
    background: rgba(34,197,94,0.08);
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_engine():

    return NetOracleInference(
        model_path="models/best_world_model.pth",
        config_path="configs/config.yaml",
    )


try:
    engine = load_engine()
    model_ready = True

except Exception as error:
    engine = None
    model_ready = False

    st.error(
        f"Failed to load trained model: {error}"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ NetOracle</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="subtitle">
AI World Model for predictive cyber defence —
learning network-state dynamics and forecasting attack
progression before compromise.
</div>
""",
    unsafe_allow_html=True,
)


if model_ready:

    st.caption(
        f"Model: Loaded | "
        f"Device: {str(engine.device).upper()} | "
        f"Context: 200 seconds | "
        f"Forecast: 50 seconds | "
        f"Alert threshold: {engine.threshold:.2f}"
    )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "🔮 Real Traffic Forecast",
        "📊 Benchmark",
        "🔍 Explainability",
        "🗺️ MITRE ATT&CK",
        "🧠 Architecture",
    ]
)


# ============================================================
# TAB 1 — REAL TRAFFIC
# ============================================================

with tabs[0]:

    st.subheader(
        "Real Network Traffic Forecast"
    )

    st.write(
        "Upload a CIC-IDS-2018 flow CSV. "
        "NetOracle converts flows into real 10-second "
        "network states and repeatedly simulates the "
        "next 50 seconds of network evolution."
    )

    col_controls, col_info = st.columns(
        [2, 1]
    )

    with col_controls:

        uploaded_file = st.file_uploader(
            "Upload CIC-IDS-2018 CSV",
            type=["csv"],
        )

        demo_mode = st.checkbox(
            "Use local held-out infiltration demo",
            value=False,
        )

        stride = st.slider(
            "Forecast interval",
            min_value=1,
            max_value=20,
            value=5,
            help=(
                "Each state is 10 seconds. "
                "Stride 5 generates a new forecast "
                "every 50 seconds."
            ),
        )

    with col_info:

        st.info(
            "Observation window: 200 sec\n\n"
            "Forecast horizon: 50 sec\n\n"
            "State dimension: 85\n\n"
            "Execution: Offline"
        )

    run_analysis = st.button(
        "🚀 Analyze Traffic",
        type="primary",
    )

    if run_analysis:

        if not model_ready:

            st.error(
                "The trained World Model is unavailable."
            )

        elif (
            uploaded_file is None
            and not demo_mode
        ):

            st.warning(
                "Upload a CSV or enable the "
                "local held-out demo."
            )

        else:

            temp_path = None

            try:

                # --------------------------------------------
                # DATA SOURCE
                # --------------------------------------------

                if demo_mode:

                    csv_path = (
                        "data/raw/"
                        "Wednesday-28-02-2018_"
                        "TrafficForML_CICFlowMeter.csv"
                    )

                    start_time = (
                        "2018-02-28 10:00:00"
                    )

                    end_time = (
                        "2018-02-28 13:00:00"
                    )

                    source_name = (
                        "Held-out Wednesday-28 "
                        "late infiltration episode"
                    )

                else:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".csv",
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = temp_file.name

                    csv_path = temp_path
                    start_time = None
                    end_time = None
                    source_name = uploaded_file.name

                # --------------------------------------------
                # REAL INFERENCE
                # --------------------------------------------

                with st.spinner(
                    "Extracting network states and "
                    "running recursive World Model simulation..."
                ):

                    timeline = engine.predict_timeline(
                        csv_path,
                        start_time=start_time,
                        end_time=end_time,
                        stride=stride,
                    )

                risks = timeline[
                    "risk_timeline"
                ]

                stages = timeline[
                    "stage_timeline"
                ]

                observed = timeline[
                    "observed_timeline"
                ]

                threshold = float(
                    timeline["threshold"]
                )

                if len(risks) == 0:

                    raise RuntimeError(
                        "No forecasts were generated. "
                        "The uploaded file may contain too "
                        "little traffic."
                    )

                # --------------------------------------------
                # SUMMARY
                # --------------------------------------------

                max_risk = float(
                    np.max(risks)
                )

                mean_risk = float(
                    np.mean(risks)
                )

                alert_count = int(
                    np.sum(
                        risks >= threshold
                    )
                )

                peak_index = int(
                    np.argmax(risks)
                )

                peak_stage = int(
                    stages[peak_index]
                )

                peak_info = (
                    MITREMapper.get_stage_info(
                        peak_stage
                    )
                )

                st.success(
                    f"Analyzed: {source_name}"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Maximum Forecast Risk",
                    f"{max_risk:.1%}",
                )

                c2.metric(
                    "Mean Forecast Risk",
                    f"{mean_risk:.1%}",
                )

                c3.metric(
                    "Forecast Alerts",
                    alert_count,
                )

                c4.metric(
                    "Highest-Risk Stage",
                    peak_info["name"],
                )

                # --------------------------------------------
                # RISK TIMELINE
                # --------------------------------------------

                st.subheader(
                    "Forecast Risk Timeline"
                )

                x = np.arange(
                    len(risks)
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=risks,
                        mode="lines",
                        name="Predicted Future Risk",
                        line=dict(
                            color="#ef4444",
                            width=2,
                        ),
                        fill="tozeroy",
                        fillcolor=(
                            "rgba(239,68,68,0.12)"
                        ),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.full(
                            len(x),
                            threshold,
                        ),
                        mode="lines",
                        name="Alert Threshold",
                        line=dict(
                            color="#f59e0b",
                            width=2,
                            dash="dash",
                        ),
                    )
                )

                malicious_points = np.where(
                    observed == 1
                )[0]

                if len(malicious_points):

                    fig.add_trace(
                        go.Scatter(
                            x=malicious_points,
                            y=np.ones(
                                len(
                                    malicious_points
                                )
                            ),
                            mode="markers",
                            name=(
                                "Observed Malicious State"
                            ),
                            marker=dict(
                                color="#a855f7",
                                size=6,
                            ),
                        )
                    )

                fig.update_layout(
                    template="plotly_dark",
                    height=430,
                    xaxis_title=(
                        "Chronological Forecast Step"
                    ),
                    yaxis_title=(
                        "Attack Probability"
                    ),
                    yaxis=dict(
                        range=[0, 1]
                    ),
                    legend=dict(
                        orientation="h"
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                # --------------------------------------------
                # MITRE TIMELINE
                # --------------------------------------------

                st.subheader(
                    "Predicted MITRE ATT&CK Timeline"
                )

                stage_names = [

                    MITREMapper.get_stage_name(
                        int(stage)
                    )

                    for stage in stages
                ]

                stage_table = pd.DataFrame(
                    {
                        "Forecast Step":
                            x,

                        "Forecast Risk":
                            risks,

                        "MITRE Stage":
                            stage_names,

                        "Alert":
                            risks >= threshold,
                    }
                )

                st.dataframe(
                    stage_table,
                    use_container_width=True,
                    height=300,
                )

                # --------------------------------------------
                # DECISION SUPPORT
                # --------------------------------------------

                if max_risk >= threshold:

                    st.markdown(
                        f"""
<div class="critical-box">

<h3>⚠️ Predictive Security Alert</h3>

<b>Peak predicted risk:</b>
{max_risk:.1%}

<br>

<b>Predicted attack stage:</b>
{peak_info['name']}

<br>

<b>MITRE tactic:</b>
{peak_info['mitre_id']}

<br><br>

<b>Recommended defensive action:</b><br>
{peak_info['recommended_action']}

</div>
""",
                        unsafe_allow_html=True,
                    )

                else:

                    st.markdown(
                        """
<div class="safe-box">

<h3>✅ No Forecast Threshold Exceeded</h3>

No future attack trajectory exceeded the
configured alert threshold.

</div>
""",
                        unsafe_allow_html=True,
                    )

            except Exception as error:

                st.exception(error)

            finally:

                if (
                    temp_path
                    and os.path.exists(
                        temp_path
                    )
                ):

                    try:
                        os.remove(
                            temp_path
                        )

                    except OSError:
                        pass


# ============================================================
# TAB 2 — BENCHMARK
# ============================================================

with tabs[1]:

    st.subheader(
        "World Model vs Static Classifier"
    )

    st.write(
        "Both systems are evaluated on the same "
        "chronologically held-out network traffic."
    )

    benchmark_path = (
        "models/final_benchmark.json"
    )

    if os.path.exists(
        benchmark_path
    ):

        try:

            with open(
                benchmark_path,
                "r",
            ) as file:

                benchmark = json.load(
                    file
                )

            world = benchmark[
                "world_model_T+1"
            ]

            baseline = benchmark[
                "logistic_regression_T+1"
            ]

            comparison = pd.DataFrame(
                {
                    "Metric": [
                        "F1 Score",
                        "AUC-ROC",
                        "Precision",
                        "Recall",
                        "False Positive Rate",
                    ],

                    "World Model": [
                        world[
                            "f1_score"
                        ],
                        world[
                            "auc_roc"
                        ],
                        world[
                            "precision"
                        ],
                        world[
                            "recall"
                        ],
                        world[
                            "false_positive_rate"
                        ],
                    ],

                    "Logistic Regression": [
                        baseline[
                            "f1_score"
                        ],
                        baseline[
                            "auc_roc"
                        ],
                        baseline[
                            "precision"
                        ],
                        baseline[
                            "recall"
                        ],
                        baseline[
                            "false_positive_rate"
                        ],
                    ],
                }
            )

            # --------------------------------------------
            # KEY METRICS
            # --------------------------------------------

            f1_improvement = (
                (
                    world["f1_score"]
                    - baseline["f1_score"]
                )
                / baseline["f1_score"]
                * 100
            )

            auc_improvement = (
                (
                    world["auc_roc"]
                    - baseline["auc_roc"]
                )
                / baseline["auc_roc"]
                * 100
            )

            fpr_reduction = (
                (
                    baseline[
                        "false_positive_rate"
                    ]
                    - world[
                        "false_positive_rate"
                    ]
                )
                / baseline[
                    "false_positive_rate"
                ]
                * 100
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Relative F1 Improvement",
                f"+{f1_improvement:.1f}%",
            )

            c2.metric(
                "Relative AUC Improvement",
                f"+{auc_improvement:.1f}%",
            )

            c3.metric(
                "Relative FPR Reduction",
                f"{fpr_reduction:.1f}%",
            )

            st.dataframe(
                comparison,
                use_container_width=True,
            )

            # --------------------------------------------
            # GRAPH
            # --------------------------------------------

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    name="World Model",
                    x=comparison["Metric"],
                    y=comparison[
                        "World Model"
                    ],
                    marker_color="#22c55e",
                )
            )

            fig.add_trace(
                go.Bar(
                    name=(
                        "Logistic Regression"
                    ),
                    x=comparison["Metric"],
                    y=comparison[
                        "Logistic Regression"
                    ],
                    marker_color="#64748b",
                )
            )

            fig.update_layout(
                barmode="group",
                template="plotly_dark",
                height=430,
                yaxis_title="Score",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.info(
                "The temporal World Model achieves "
                "higher F1, AUC and recall than the "
                "static Logistic Regression baseline "
                "while reducing its false-positive rate."
            )

            st.caption(
                "Note: The current prototype still has "
                "a non-trivial absolute false-positive "
                "rate. Further calibration is required "
                "for production deployment."
            )

        except Exception as error:

            st.error(
                f"Could not read benchmark: {error}"
            )

    else:

        st.warning(
            "models/final_benchmark.json "
            "was not found."
        )


# ============================================================
# TAB 3 — EXPLAINABILITY
# ============================================================

with tabs[2]:

    st.subheader(
        "Explainable Forecasting"
    )

    st.write(
        "Input-gradient attribution estimates which "
        "traffic-state features most strongly influence "
        "the model's future attack prediction."
    )

    if not model_ready:

        st.warning(
            "Model unavailable."
        )

    else:

        demo_file = (
            "data/raw/"
            "Wednesday-28-02-2018_"
            "TrafficForML_CICFlowMeter.csv"
        )

        if st.button(
            "🔍 Generate Real Traffic Explanation"
        ):

            if not os.path.exists(
                demo_file
            ):

                st.warning(
                    "Local CIC-IDS demo dataset "
                    "is not available."
                )

            else:

                try:

                    with st.spinner(
                        "Computing feature attribution..."
                    ):

                        prepared = (
                            engine.prepare_csv(

                                demo_file,

                                start_time=
                                    "2018-02-28 10:00:00",

                                end_time=
                                    "2018-02-28 12:00:00",
                            )
                        )

                        explainer = (
                            WorldModelExplainer(

                                engine.model,

                                engine.feature_names,
                            )
                        )

                        features = (
                            explainer
                            .get_top_features(

                                prepared[
                                    "input_tensor"
                                ],

                                top_k=10,
                            )
                        )

                    names = [
                        item[0]
                        for item in features
                    ]

                    scores = [
                        item[1]
                        for item in features
                    ]

                    fig = go.Figure(
                        go.Bar(
                            x=scores,
                            y=names,
                            orientation="h",
                            marker=dict(
                                color=scores,
                                colorscale="Reds",
                            ),
                        )
                    )

                    fig.update_layout(
                        template="plotly_dark",
                        height=450,
                        title=(
                            "Features Driving "
                            "the Attack Forecast"
                        ),
                        xaxis_title=(
                            "Attribution Magnitude"
                        ),
                        yaxis=dict(
                            autorange="reversed"
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                    st.info(
                        "Higher attribution means the "
                        "feature had a stronger influence "
                        "on the model's predicted risk. "
                        "Attribution indicates model "
                        "influence, not proof of causality."
                    )

                except Exception as error:

                    st.exception(error)


# ============================================================
# TAB 4 — MITRE ATT&CK
# ============================================================

with tabs[3]:

    st.subheader(
        "MITRE ATT&CK Decision Support"
    )

    st.caption(
        "CIC-IDS attack labels are mapped to "
        "prototype defensive ATT&CK stages. "
        "These are semantic mappings, not native "
        "ground-truth MITRE annotations."
    )

    for stage_id in range(7):

        info = (
            MITREMapper.get_stage_info(
                stage_id
            )
        )

        with st.expander(
            f"{info['name']} "
            f"({info['mitre_id']}) "
            f"— Severity {info['severity']}/10"
        ):

            st.write(
                info["description"]
            )

            st.write(
                "**Recommended Action:**"
            )

            st.write(
                info[
                    "recommended_action"
                ]
            )

            if (
                "techniques"
                in info
            ):

                st.write(
                    "**Representative Techniques:**"
                )

                for technique in info[
                    "techniques"
                ]:

                    st.write(
                        f"- {technique}"
                    )


# ============================================================
# TAB 5 — ARCHITECTURE
# ============================================================

with tabs[4]:

    st.subheader(
        "NetOracle World Model"
    )

    st.code(
        """
Real CIC-IDS-2018 Flow Telemetry
               |
               v
       10-second States
               |
               v
          State Encoder
               |
               v
   Causal Temporal Transformer
               |
               v
       Latent State Z(t)
               |
               v
  Learned Transition Dynamics
               |
          +----+----+
          |         |
          v         v
      Z(t+1)   Future Network State
          |
          +----> Attack Probability
          |
          +----> MITRE Stage
          |
          v
      Z(t+2)
          |
         ...
          |
          v
      Z(t+5)

Historical context : 20 states = 200 seconds
Forecast horizon   : 5 states  = 50 seconds
Network state      : 85 dimensions
        """,
        language="text",
    )

    st.markdown(
        """
Key properties:

- Real timestamp-based network states
- 200-second temporal observation context
- Recursive five-step future simulation
- Causal Transformer representation
- Learned latent state-transition dynamics
- Multi-task future-state, risk and stage prediction
- Training-only normalization
- Chronological evaluation
- Explainable feature attribution
- Offline execution
        """
    )

    st.subheader(
        "Why this is different from a classifier"
    )

    st.write(
        "A static classifier asks: "
        "'Is the current flow malicious?' "
        "NetOracle instead asks: "
        "'Given how the network has evolved over the "
        "last 200 seconds, what states are likely to "
        "occur during the next 50 seconds?'"
    )

    st.warning(
        "Prototype limitation: the current model "
        "primarily uses CICFlowMeter flow-level "
        "telemetry. Packet-level PCAP feature support "
        "is being developed as a separate ingestion "
        "pipeline."
    )