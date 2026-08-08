import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from prophet import Prophet

from llm import generate_insights

# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------

if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

LIGHT_MODE = st.session_state.light_mode

# ---------------------------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="iADAS | Intelligent Anomaly Detection and Alerting System",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_ROOT = Path(__file__).parent
CYAN = "#8FCBFF"
VIOLET = "#B8B2FF"


# A small, intentionally scoped layer for the branded app shell. The actual
# widget, table, chart, and semantic colors live in .streamlit/config.toml.
st.html(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% -8%, rgba(101, 116, 255, .2), transparent 36rem),
            radial-gradient(circle at -10% 36%, rgba(61, 154, 222, .16), transparent 34rem),
            radial-gradient(circle at 62% 86%, rgba(40, 163, 151, .08), transparent 30rem),
            linear-gradient(145deg, #07101D 0%, #090D18 48%, #07121A 100%);
    }

    .stApp::before {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        content: "";
        opacity: .75;
        background:
            radial-gradient(ellipse at 18% 12%, rgba(91, 173, 255, .1), transparent 30%),
            radial-gradient(ellipse at 78% 36%, rgba(132, 112, 255, .09), transparent 28%);
    }

    .stApp::after {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        content: "";
        opacity: .22;
        background-image:
            linear-gradient(rgba(255, 255, 255, .025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, .025) 1px, transparent 1px);
        background-size: 72px 72px;
        mask-image: linear-gradient(to bottom, black, transparent 78%);
    }

    .main .block-container {
        position: relative;
        z-index: 1;
        max-width: 1480px;
        padding: 1rem clamp(1.1rem, 3.2vw, 3.25rem) 5rem;
        perspective: 1600px;
        perspective-origin: 50% 30%;
    }

    html, body, [class*="st-"] {
        font-feature-settings: "cv02", "cv03", "cv04", "cv11";
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
    }

    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -.035em !important;
    }

    h1 {
        max-width: 1120px;
        font-size: clamp(2.15rem, 4vw, 3.25rem) !important;
        line-height: 1.04 !important;
        text-wrap: balance;
    }

    p, h1, h2, h3, h4 {
        overflow-wrap: anywhere;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .st-key-top_navigation {
        position: sticky;
        z-index: 999;
        top: .5rem;
        padding: .72rem .85rem;
        border: 1px solid rgba(255, 255, 255, .13);
        border-radius: 22px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, .075), rgba(255, 255, 255, .025));
        box-shadow:
            0 24px 70px rgba(0, 0, 0, .23),
            inset 0 1px 0 rgba(255, 255, 255, .11),
            inset 0 -1px 0 rgba(255, 255, 255, .025);
        backdrop-filter: blur(34px) saturate(150%);
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: .7rem;
        min-width: 185px;
    }

    .brand-mark {
        display: grid;
        width: 38px;
        height: 38px;
        place-items: center;
        border: 1px solid rgba(255, 255, 255, .24);
        border-radius: 13px;
        color: #F7FAFC;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, .19), rgba(111, 180, 255, .1));
        box-shadow:
            0 8px 28px rgba(18, 83, 146, .24),
            inset 0 1px 0 rgba(255, 255, 255, .24);
    }

    .brand-wave {
        display: block;
        margin-top: -.15rem;
        color: #DCEEFF;
        font-family: "Plus Jakarta Sans", sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1;
        filter: drop-shadow(0 3px 8px rgba(111, 187, 255, .28));
    }

    .brand-name {
        color: #F8FAFC;
        font-size: 1rem;
        font-weight: 720;
        letter-spacing: -.035em;
    }

    .brand-name span { color: #9FCBF4; }

    .brand-sub {
        margin-top: .08rem;
        color: #94A3B8;
        font-size: .62rem;
        font-weight: 600;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    .st-key-main_navigation [data-testid="stSegmentedControl"] {
        width: 100%;
    }

    .st-key-main_navigation button {
        min-height: 42px;
        border-radius: 10px;
        font-weight: 650;
        border-color: rgba(255, 255, 255, .07) !important;
        background: rgba(5, 10, 18, .25) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, .025);
        backdrop-filter: blur(12px);
    }

    .st-key-main_navigation button[aria-checked="true"],
    .st-key-main_navigation button[aria-pressed="true"] {
        border-color: rgba(149, 202, 255, .34) !important;
        background: rgba(128, 180, 236, .13) !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, .1),
            0 8px 24px rgba(21, 70, 118, .12);
    }

    .st-key-page_status {
        display: flex;
        justify-content: flex-end;
    }

    .live-status {
        display: inline-flex;
        align-items: center;
        gap: .48rem;
        padding: .48rem .7rem;
        border: 1px solid rgba(255, 255, 255, .11);
        border-radius: 999px;
        color: #C8D1DC;
        background: rgba(255, 255, 255, .035);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, .06);
        backdrop-filter: blur(18px);
        font-size: .73rem;
        font-weight: 650;
        white-space: nowrap;
    }

    .live-status-dot {
        width: .46rem;
        height: .46rem;
        border-radius: 999px;
        background: #7EC7A3;
        box-shadow: 0 0 0 0 rgba(126, 199, 163, .36);
        animation: status-pulse 2.4s ease-out infinite;
    }

    @keyframes status-pulse {
        0% { box-shadow: 0 0 0 0 rgba(126, 199, 163, .34); }
        65%, 100% { box-shadow: 0 0 0 .48rem rgba(126, 199, 163, 0); }
    }

    [data-testid="stMetric"] {
        min-height: 142px;
        border-color: rgba(255, 255, 255, .11);
        background:
            linear-gradient(145deg, rgba(255, 255, 255, .045), rgba(255, 255, 255, .012));
        box-shadow:
            0 14px 42px rgba(0, 0, 0, .13),
            inset 0 1px 0 rgba(255, 255, 255, .075);
        backdrop-filter: blur(26px) saturate(145%);
        transform-style: preserve-3d;
        transition:
            transform .26s cubic-bezier(.2, .8, .2, 1),
            border-color .2s ease,
            box-shadow .26s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translate3d(0, -4px, 16px) rotateX(1deg);
        border-color: rgba(160, 201, 240, .25);
        box-shadow:
            0 24px 58px rgba(0, 0, 0, .21),
            inset 0 1px 0 rgba(255, 255, 255, .1);
    }

    [class*="st-key-metric_tile_"] {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, .12);
        border-radius: 20px;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, .045), rgba(255, 255, 255, .012));
        box-shadow:
            0 18px 48px rgba(0, 0, 0, .15),
            inset 0 1px 0 rgba(255, 255, 255, .085);
        backdrop-filter: blur(30px) saturate(150%);
        transform-style: preserve-3d;
        will-change: transform;
        transition:
            transform .32s cubic-bezier(.2, .8, .2, 1),
            border-color .24s ease,
            background .24s ease,
            box-shadow .32s ease;
    }

    [class*="st-key-metric_tile_"]::before {
        position: absolute;
        inset: 0;
        pointer-events: none;
        content: "";
        background:
            radial-gradient(circle at 90% -10%, rgba(143, 203, 255, .13), transparent 42%),
            linear-gradient(110deg, rgba(255, 255, 255, .045), transparent 42%);
    }

    [class*="st-key-metric_tile_"]:hover {
        transform: translate3d(0, -7px, 24px) rotateX(1.4deg) rotateY(-.7deg);
        border-color: rgba(159, 203, 244, .38);
        background:
            linear-gradient(145deg, rgba(255, 255, 255, .065), rgba(255, 255, 255, .018));
        box-shadow:
            0 30px 74px rgba(0, 0, 0, .27),
            0 12px 34px rgba(43, 104, 163, .1),
            inset 0 1px 0 rgba(255, 255, 255, .12);
    }

    [class*="st-key-metric_tile_"] [data-testid="stMetric"] {
        min-height: 126px;
        border: 0;
        background: transparent;
        box-shadow: none;
        backdrop-filter: none;
    }

    [class*="st-key-metric_tile_"] [data-testid="stMetric"]:hover {
        transform: none;
    }

    [class*="st-key-metric_tile_"] button {
        border-color: transparent;
        color: #A9C9EA;
        background: rgba(255, 255, 255, .025);
    }

    [class*="st-key-metric_tile_"] button:hover {
        border-color: rgba(159, 203, 244, .2);
        color: #D9ECFF;
        background: rgba(143, 203, 255, .1);
    }

    .st-key-hero_panel {
        position: relative;
        overflow: hidden;
        padding: clamp(.75rem, 1.25vw, 1.15rem);
        border-color: rgba(170, 211, 255, .16);
        background:
            radial-gradient(circle at 88% 25%, rgba(143, 203, 255, .1), transparent 20rem),
            linear-gradient(145deg, rgba(255, 255, 255, .04), rgba(255, 255, 255, .01));
        box-shadow:
            0 30px 76px rgba(0, 0, 0, .23),
            0 10px 30px rgba(38, 91, 145, .08),
            inset 0 1px 0 rgba(255, 255, 255, .08);
        backdrop-filter: blur(30px) saturate(150%);
        transform-style: preserve-3d;
        transition:
            transform .34s cubic-bezier(.2, .8, .2, 1),
            border-color .24s ease,
            box-shadow .34s ease;
    }

    .st-key-hero_panel::before {
        position: absolute;
        inset: 0;
        pointer-events: none;
        content: "";
        background:
            linear-gradient(120deg, rgba(255, 255, 255, .07), transparent 24%),
            radial-gradient(circle at 96% 0%, rgba(143, 203, 255, .12), transparent 32%);
    }

    .st-key-hero_panel:hover {
        transform: translate3d(0, -5px, 20px) rotateX(.65deg);
        border-color: rgba(170, 211, 255, .25);
        box-shadow:
            0 38px 90px rgba(0, 0, 0, .28),
            0 14px 38px rgba(38, 91, 145, .11),
            inset 0 1px 0 rgba(255, 255, 255, .11);
    }

    .st-key-forecast_explorer,
    .st-key-forecast_chart_panel,
    .st-key-register_panel,
    .st-key-ai_panel {
        border-color: rgba(255, 255, 255, .1);
        background:
            linear-gradient(145deg, rgba(255, 255, 255, .035), rgba(255, 255, 255, .008));
        box-shadow:
            0 18px 50px rgba(0, 0, 0, .15),
            inset 0 1px 0 rgba(255, 255, 255, .055);
        backdrop-filter: blur(30px) saturate(150%);
    }

    .st-key-chart_controls_dock {
        margin-top: .45rem;
        padding: .25rem .35rem .35rem;
        border-color: rgba(159, 203, 244, .14);
        background:
            linear-gradient(135deg, rgba(124, 180, 237, .07), rgba(255, 255, 255, .012));
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, .065),
            0 12px 34px rgba(0, 0, 0, .12);
        backdrop-filter: blur(22px) saturate(140%);
    }

    [class*="st-key-pipeline_stage_"] {
        background: linear-gradient(145deg, rgba(255, 255, 255, .032), rgba(255, 255, 255, .008));
    }

    [class*="st-key-pipeline_stage_"],
    [class*="st-key-value_card_"],
    .st-key-forecast_explorer,
    .st-key-forecast_chart_panel,
    .st-key-ai_panel,
    .st-key-register_panel {
        transform-style: preserve-3d;
        transition:
            transform .3s cubic-bezier(.2, .8, .2, 1),
            border-color .22s ease,
            box-shadow .3s ease;
    }

    [class*="st-key-pipeline_stage_"]:hover,
    [class*="st-key-value_card_"]:hover {
        transform: translate3d(0, -6px, 20px) rotateX(1.4deg) rotateY(-.6deg);
        border-color: rgba(159, 203, 244, .28);
        box-shadow:
            0 26px 62px rgba(0, 0, 0, .23),
            inset 0 1px 0 rgba(255, 255, 255, .1);
    }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border-radius: 16px;
        box-shadow: 0 14px 36px rgba(0, 0, 0, .14);
    }

    [data-testid="stDialog"] [role="dialog"] {
        border: 1px solid rgba(255, 255, 255, .14);
        border-radius: 26px;
        background: rgba(27, 32, 41, .9);
        box-shadow:
            0 34px 100px rgba(0, 0, 0, .48),
            inset 0 1px 0 rgba(255, 255, 255, .08);
        backdrop-filter: blur(36px) saturate(130%);
    }

    .st-key-forecast_layers [data-testid="stPills"] {
        width: 100%;
    }

    .st-key-forecast_window button,
    .st-key-forecast_chart_view button,
    .st-key-forecast_layers button,
    .st-key-anomaly_selector button {
        border-color: rgba(255, 255, 255, .09) !important;
        background: rgba(5, 10, 18, .23) !important;
        backdrop-filter: blur(14px);
    }

    .st-key-forecast_window button[aria-checked="true"],
    .st-key-forecast_chart_view button[aria-checked="true"],
    .st-key-forecast_layers button[aria-pressed="true"],
    .st-key-anomaly_selector button[aria-checked="true"] {
        border-color: rgba(149, 202, 255, .36) !important;
        color: #DCEEFF !important;
        background: rgba(122, 177, 235, .14) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, .08);
    }

    .anomaly-split {
        display: flex;
        gap: .7rem;
        align-items: center;
        flex-wrap: wrap;
        margin-top: .65rem;
        font-size: .82rem;
        font-weight: 650;
    }

    .anomaly-positive {
        color: #91D3B0;
        background: rgba(145, 211, 176, .08);
        border: 1px solid rgba(145, 211, 176, .16);
        padding: .28rem .55rem;
        border-radius: 999px;
    }

    .anomaly-negative {
        color: #F0A0A0;
        background: rgba(240, 160, 160, .08);
        border: 1px solid rgba(240, 160, 160, .16);
        padding: .28rem .55rem;
        border-radius: 999px;
    }

    @keyframes viewport-reveal {
        from {
            opacity: .08;
            translate: 0 52px;
            rotate: x 5deg;
            scale: .975;
            filter: blur(7px);
        }
        to {
            opacity: 1;
            translate: 0 0;
            rotate: x 0deg;
            scale: 1;
            filter: blur(0);
        }
    }

    @supports (animation-timeline: view()) {
        .st-key-hero_panel,
        [class*="st-key-metric_tile_"],
        [class*="st-key-pipeline_stage_"],
        [class*="st-key-value_card_"],
        .st-key-forecast_explorer,
        .st-key-forecast_chart_panel,
        .st-key-ai_panel,
        .st-key-register_panel {
            animation-name: viewport-reveal;
            animation-duration: 1ms;
            animation-fill-mode: both;
            animation-timing-function: cubic-bezier(.2, .8, .2, 1);
            animation-timeline: view(block);
            animation-range: entry 0% entry 34%;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .live-status-dot { animation: none; }
        [data-testid="stMetric"],
        [class*="st-key-metric_tile_"],
        .st-key-hero_panel,
        [class*="st-key-pipeline_stage_"],
        [class*="st-key-value_card_"] {
            animation: none !important;
            transition: none;
            translate: 0;
            rotate: none;
            scale: 1;
            filter: none;
        }
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding: .65rem 1rem 3.5rem;
        }

        .st-key-top_navigation { position: relative; top: 0; }

        .main [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: .9rem !important;
        }

        .main [data-testid="stColumn"] {
            flex: 1 1 280px !important;
            width: auto !important;
            min-width: min(100%, 280px) !important;
        }

        .st-key-top_navigation [data-testid="stColumn"]:nth-child(1) {
            flex: 1 1 55% !important;
            min-width: 180px !important;
        }

        .st-key-top_navigation [data-testid="stColumn"]:nth-child(2) {
            order: 3;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        .st-key-top_navigation [data-testid="stColumn"]:nth-child(3) {
            flex: 0 1 auto !important;
            min-width: auto !important;
        }

        .st-key-hero_panel [data-testid="stHorizontalBlock"],
        .st-key-forecast_explorer [data-testid="stHorizontalBlock"],
        .st-key-chart_controls_dock [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }

        .st-key-hero_panel [data-testid="stColumn"],
        .st-key-forecast_explorer [data-testid="stColumn"],
        .st-key-chart_controls_dock [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }

        .st-key-hero_panel {
            padding: .6rem;
        }

        .st-key-hero_panel h3 {
            font-size: clamp(1.15rem, 5vw, 1.45rem) !important;
            line-height: 1.24 !important;
        }

        .brand-lockup { min-width: auto; }
        .brand-sub { display: none; }
    }

    @media (max-width: 560px) {
        .main .block-container {
            padding: .4rem .75rem 3rem;
        }

        h1 {
            font-size: clamp(1.95rem, 9vw, 2.4rem) !important;
            line-height: 1.08 !important;
        }

        .st-key-top_navigation {
            padding: .6rem;
            border-radius: 18px;
        }

        .st-key-top_navigation [data-testid="stColumn"]:nth-child(1) {
            flex-basis: 60% !important;
        }

        .live-status {
            padding: .4rem .55rem;
            font-size: .67rem;
        }

        .main [data-testid="stColumn"] {
            flex-basis: 100% !important;
            min-width: 100% !important;
        }

        [class*="st-key-metric_tile_"] {
            border-radius: 18px;
        }
    }
    </style>
    """
)


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dashboard_data():
    with open(APP_ROOT / "dashboard_data.pkl", "rb") as file:
        return pickle.load(file)


data = load_dashboard_data()
results = data["results"].copy()
anomalies = data["anomalies"].copy()
reports_df = data["reports_df"].copy()
mae = float(data["mae"])
rmse = float(data["rmse"])

results["ds"] = pd.to_datetime(results["ds"])
anomalies["ds"] = pd.to_datetime(anomalies["ds"])


@st.cache_data(show_spinner=False)
def load_weekly_rca():
    """Load transaction-derived RCA lookup tables built in Colab."""
    rca_path = APP_ROOT / "weekly_rca.pkl"
    if not rca_path.exists():
        return None
    with open(rca_path, "rb") as file:
        rca = pickle.load(file)
    for key in ("product_rca", "customer_rca", "country_rca"):
        rca[key] = rca[key].copy()
        rca[key]["Week"] = pd.to_datetime(rca[key]["Week"]).dt.normalize()
    return rca


weekly_rca = load_weekly_rca()


total_sales_all = float(results["y"].sum())
forecast_sales_all = float(results["yhat"].sum())
accuracy_all = (
    100
    - abs(results["y"] - results["yhat"]).mean()
    / results["y"].mean()
    * 100
)
high_count = int((anomalies["AnomalyType"] == "High Sales").sum())
low_count = int((anomalies["AnomalyType"] == "Low Sales").sum())
largest = reports_df.iloc[reports_df["Deviation %"].abs().argmax()]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def section_header(title: str, caption: str, icon: str | None = None):
    heading = f"{icon} {title}" if icon else title
    st.subheader(heading)
    st.caption(caption)


def page_header(label: str, title: str, description: str, badge: str):
    st.badge(label, icon=":material/auto_awesome:", color="violet")
    st.title(title)
    st.markdown(description)
    st.caption(badge)


@st.cache_data(show_spinner=False)
def generate_dynamic_forecast(base_results: pd.DataFrame, confidence_level: float):
    """Re-fit the Prophet model with the selected prediction-interval width."""
    training_data = base_results[["ds", "y"]].copy()
    training_data["ds"] = pd.to_datetime(training_data["ds"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=confidence_level,
    )
    model.fit(training_data)

    forecast = model.predict(training_data[["ds"]])

    dynamic_results = training_data.merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="left",
    )

    dynamic_results["Anomaly"] = (
        (dynamic_results["y"] > dynamic_results["yhat_upper"])
        | (dynamic_results["y"] < dynamic_results["yhat_lower"])
    )
    dynamic_results["AnomalyType"] = "Normal"
    dynamic_results.loc[
        dynamic_results["y"] > dynamic_results["yhat_upper"],
        "AnomalyType",
    ] = "High Sales"
    dynamic_results.loc[
        dynamic_results["y"] < dynamic_results["yhat_lower"],
        "AnomalyType",
    ] = "Low Sales"

    dynamic_anomalies = dynamic_results[
        dynamic_results["Anomaly"]
    ].copy()

    return dynamic_results, dynamic_anomalies


def get_dynamic_rca(week):
    """Return transaction-derived RCA for any detected anomaly week."""
    if weekly_rca is None:
        return None

    week = pd.Timestamp(week).normalize()
    product_rca = weekly_rca["product_rca"]
    customer_rca = weekly_rca["customer_rca"]
    country_rca = weekly_rca["country_rca"]

    result = {
        "week": week,
        "previous_week": None,
        "product": None,
        "customer": None,
        "country": None,
    }

    # Find the immediately preceding RCA week that exists in the data.
    all_weeks = sorted(
        set(product_rca["Week"])
        | set(customer_rca["Week"])
        | set(country_rca["Week"])
    )
    previous_weeks = [w for w in all_weeks if w < week]
    previous_week = previous_weeks[-1] if previous_weeks else None
    result["previous_week"] = previous_week

    def top_record(frame, name_col, previous_frame=None):
        current = frame[frame["Week"] == week].copy()
        if current.empty:
            return None
        current = current.sort_values("TotalSales", ascending=False)
        top = current.iloc[0]
        total = float(current["TotalSales"].sum())
        current_sales = float(top["TotalSales"])

        previous_sales = 0.0
        if previous_frame is not None and previous_week is not None:
            previous = previous_frame[previous_frame["Week"] == previous_week]
            if not previous.empty:
                match = previous[previous[name_col] == top[name_col]]
                if not match.empty:
                    previous_sales = float(match.iloc[0]["TotalSales"])

        change_pct = None
        if previous_sales > 0:
            change_pct = (current_sales - previous_sales) / previous_sales * 100

        return {
            "name": str(top[name_col]).strip(),
            "sales": current_sales,
            "contribution_pct": (current_sales / total * 100) if total > 0 else 0.0,
            "previous_week_sales": previous_sales,
            "week_over_week_change_pct": change_pct,
        }

    product = top_record(product_rca, "Description", product_rca)
    customer = top_record(customer_rca, "Customer ID", customer_rca)
    country = top_record(country_rca, "Country", country_rca)

    if customer is not None:
        customer["name"] = customer["name"]
        customer["id"] = customer["name"]
        customer.pop("name", None)

    result["product"] = product
    result["customer"] = customer
    result["country"] = country
    return result


def build_dynamic_anomaly_report(anomaly_row):
    """Combine dynamic forecast output with transaction-level RCA."""
    week = pd.Timestamp(anomaly_row["ds"]).normalize()
    rca = get_dynamic_rca(week)

    if rca is None:
        return pd.Series({
            "Week": week,
            "Type": anomaly_row["AnomalyType"],
            "Actual Sales": float(anomaly_row["y"]),
            "Forecast Sales": float(anomaly_row["yhat"]),
            "Deviation %": ((float(anomaly_row["y"]) - float(anomaly_row["yhat"])) / float(anomaly_row["yhat"]) * 100) if float(anomaly_row["yhat"]) else 0.0,
            "Top Product": "RCA unavailable",
            "Top Customer": "RCA unavailable",
            "Top Country": "RCA unavailable",
            "Root Cause Driver": "RCA unavailable",
            "Driver Increase": 0.0,
            "Driver Change %": 0.0,
            "Product Contribution %": 0.0,
            "Customer Contribution %": 0.0,
            "Country Contribution %": 0.0,
        })

    product = rca["product"]
    customer = rca["customer"]
    country = rca["country"]

    drivers = []
    for label, item in (("Product", product), ("Customer", customer), ("Country", country)):
        if item and item.get("week_over_week_change_pct") is not None:
            amount_change = float(item["sales"]) - float(item.get("previous_week_sales", 0.0))
            drivers.append((
                label,
                abs(item["week_over_week_change_pct"]),
                item["week_over_week_change_pct"],
                amount_change,
            ))

    driver = max(drivers, key=lambda x: x[1]) if drivers else None
    driver_name = driver[0] if driver else "Transaction concentration"
    driver_change_pct = driver[2] if driver else 0.0
    driver_change_amount = driver[3] if driver else 0.0

    actual = float(anomaly_row["y"])
    forecast = float(anomaly_row["yhat"])
    deviation = ((actual - forecast) / forecast * 100) if forecast else 0.0

    anomaly_type = anomaly_row["AnomalyType"]
    recommendation = (
        "Review inventory levels, validate whether the increase is seasonal or a bulk customer order, and prepare replenishment if demand is expected to continue."
        if anomaly_type == "High Sales"
        else "Investigate inventory availability, operational issues, pricing, promotions, or changes in customer demand."
    )

    return pd.Series({
        "Week": week,
        "Type": anomaly_type,
        "Actual Sales": actual,
        "Forecast Sales": forecast,
        "Deviation %": deviation,
        "Top Product": product["name"] if product else "RCA unavailable",
        "Top Customer": customer["id"] if customer else "RCA unavailable",
        "Top Country": country["name"] if country else "RCA unavailable",
        "Root Cause Driver": driver_name,
        "Driver Increase": driver_change_amount,
        "Driver Change %": driver_change_pct,
        "Product Contribution %": product["contribution_pct"] if product else 0.0,
        "Customer Contribution %": customer["contribution_pct"] if customer else 0.0,
        "Country Contribution %": country["contribution_pct"] if country else 0.0,
        "Recommendation": recommendation,
    })


@st.cache_data(show_spinner=False)
def dynamic_reports_from_anomalies(dynamic_anomalies):
    """Build the anomaly register directly from the current sensitivity setting."""
    if dynamic_anomalies.empty:
        return pd.DataFrame(columns=[
            "Week", "Type", "Actual Sales", "Forecast Sales", "Deviation %",
            "Top Product", "Top Customer", "Top Country", "Root Cause Driver",
            "Driver Increase", "Driver Change %", "Product Contribution %",
            "Customer Contribution %", "Country Contribution %"
        ])
    reports = [build_dynamic_anomaly_report(row) for _, row in dynamic_anomalies.iterrows()]
    return pd.DataFrame(reports).sort_values("Week").reset_index(drop=True)


def resolve_window(option: str):
    end = results["ds"].max()
    if option == "Last 26 weeks":
        return end - pd.Timedelta(weeks=26), end
    if option == "Last 52 weeks":
        return end - pd.Timedelta(weeks=52), end
    return results["ds"].min(), end


# Persistent sensitivity state shared across all workspaces.
# IMPORTANT: do NOT use the slider widget key for this value because Streamlit
# removes widget state when that widget is not rendered on another page.
if "active_confidence_pct" not in st.session_state:
    st.session_state["active_confidence_pct"] = 95

def persist_confidence():
    st.session_state["active_confidence_pct"] = int(
        st.session_state["confidence_slider"]
    )

active_confidence_pct = int(st.session_state["active_confidence_pct"])


@st.dialog(
    "Metric intelligence",
    width="large",
    icon=":material/monitoring:",
    on_dismiss="rerun",
)
def show_metric_details(
    metric_key: str,
    frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
):
    """Render one reusable, selection-aware metric detail surface."""
    detail_frame = frame.copy()
    signal_frame = signal_frame.copy()
    detail_frame["Accuracy"] = (
        100
        - abs(detail_frame["y"] - detail_frame["yhat"])
        / detail_frame["y"].replace(0, pd.NA)
        * 100
    ).fillna(0)
    detail_frame["Variance"] = detail_frame["y"] - detail_frame["yhat"]

    title_map = {
        "actual": (
            "Actual sales",
            "Observed demand across the active analysis window.",
        ),
        "forecast": (
            "AI forecast",
            "Expected demand and its distance from the observed result.",
        ),
        "accuracy": (
            "Forecast accuracy",
            "Week-level model fit across the active analysis window.",
        ),
        "anomalies": (
            "Anomaly signals",
            "Statistically meaningful demand events in the active window.",
        ),
    }
    title, caption = title_map[metric_key]
    st.badge(
        f"{detail_frame['ds'].min():%d %b %Y} — "
        f"{detail_frame['ds'].max():%d %b %Y}",
        icon=":material/date_range:",
        color="blue",
    )
    st.subheader(title)
    st.caption(caption)

    overview_columns = st.columns(3)
    if metric_key == "actual":
        overview_columns[0].metric(
            "Total sales",
            f"£{detail_frame['y'].sum() / 1_000_000:.2f}M",
            border=True,
        )
        overview_columns[1].metric(
            "Weekly average",
            f"£{detail_frame['y'].mean():,.0f}",
            border=True,
        )
        peak_row = detail_frame.loc[detail_frame["y"].idxmax()]
        overview_columns[2].metric(
            "Peak week",
            f"£{peak_row['y']:,.0f}",
            f"{peak_row['ds']:%d %b %Y}",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        chart_columns = ["y"]
        chart_colors = [CYAN]
        chart_y_label = "Weekly sales (£)"
    elif metric_key == "forecast":
        total_variance = detail_frame["Variance"].sum()
        overview_columns[0].metric(
            "Forecast total",
            f"£{detail_frame['yhat'].sum() / 1_000_000:.2f}M",
            border=True,
        )
        overview_columns[1].metric(
            "Net variance",
            f"£{total_variance:+,.0f}",
            border=True,
        )
        overview_columns[2].metric(
            "Mean absolute error",
            f"£{detail_frame['Variance'].abs().mean():,.0f}",
            border=True,
        )
        chart_columns = ["y", "yhat"]
        chart_colors = [CYAN, VIOLET]
        chart_y_label = "Weekly sales (£)"
    elif metric_key == "accuracy":
        overview_columns[0].metric(
            "Mean accuracy",
            f"{detail_frame['Accuracy'].mean():.1f}%",
            border=True,
        )
        overview_columns[1].metric(
            "Best week",
            f"{detail_frame['Accuracy'].max():.1f}%",
            border=True,
        )
        overview_columns[2].metric(
            "Weeks above 90%",
            int((detail_frame["Accuracy"] >= 90).sum()),
            border=True,
        )
        chart_columns = ["Accuracy"]
        chart_colors = ["#91D3B0"]
        chart_y_label = "Accuracy (%)"
    else:
        high_signals = int(
            (signal_frame["AnomalyType"] == "High Sales").sum()
        )
        low_signals = int(
            (signal_frame["AnomalyType"] == "Low Sales").sum()
        )
        overview_columns[0].metric(
            "Signals",
            len(signal_frame),
            border=True,
        )
        overview_columns[1].metric(
            "Demand spikes",
            high_signals,
            border=True,
        )
        overview_columns[2].metric(
            "Demand drops",
            low_signals,
            border=True,
        )
        chart_columns = ["Variance"]
        chart_colors = [VIOLET]
        chart_y_label = "Variance to forecast (£)"

    trend_tab, signals_tab, data_tab = st.tabs(
        [
            ":material/show_chart: Trend",
            ":material/priority_high: Signals",
            ":material/table_view: Data",
        ]
    )
    with trend_tab:
        st.line_chart(
            detail_frame,
            x="ds",
            y=chart_columns,
            x_label="Week",
            y_label=chart_y_label,
            color=chart_colors,
            height=390,
            width="stretch",
        )
    with signals_tab:
        if signal_frame.empty:
            st.info(
                "No statistically significant events in this window.",
                icon=":material/check_circle:",
            )
        else:
            dialog_events = signal_frame[
                ["ds", "AnomalyType", "y"]
            ].rename(
                columns={
                    "ds": "Week",
                    "AnomalyType": "Signal",
                    "y": "Actual sales",
                }
            )
            st.dataframe(
                dialog_events,
                hide_index=True,
                width="stretch",
                column_config={
                    "Week": st.column_config.DateColumn(
                        "Week",
                        format="DD MMM YYYY",
                    ),
                    "Actual sales": st.column_config.NumberColumn(
                        "Actual sales (£)",
                        format="localized",
                    ),
                },
            )
    with data_tab:
        table_columns = [
            "ds",
            "y",
            "yhat",
            "Variance",
            "Accuracy",
        ]
        st.dataframe(
            detail_frame[table_columns].rename(
                columns={
                    "ds": "Week",
                    "y": "Actual sales",
                    "yhat": "AI forecast",
                }
            ),
            hide_index=True,
            width="stretch",
            height=360,
            column_config={
                "Week": st.column_config.DateColumn(
                    "Week",
                    format="DD MMM YYYY",
                ),
                "Actual sales": st.column_config.NumberColumn(
                    "Actual sales (£)",
                    format="localized",
                ),
                "AI forecast": st.column_config.NumberColumn(
                    "AI forecast (£)",
                    format="localized",
                ),
                "Variance": st.column_config.NumberColumn(
                    "Variance (£)",
                    format="localized",
                ),
                "Accuracy": st.column_config.NumberColumn(
                    "Accuracy",
                    format="%.1f%%",
                ),
            },
        )


def interactive_metric_tile(
    column,
    *,
    tile_key: str,
    metric_key: str,
    label: str,
    value,
    delta,
    frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    chart_data,
    chart_type: str,
    delta_color: str = "normal",
    delta_arrow: str = "auto",
):
    """Render a glass KPI tile with a large selection-aware detail dialog."""
    with column:
        with st.container(border=True, key=f"metric_tile_{tile_key}"):
            st.metric(
                label,
                value,
                delta,
                delta_color=delta_color,
                delta_arrow=delta_arrow,
                border=False,
                chart_data=chart_data,
                chart_type=chart_type,
            )
            if st.button(
                "Explore metric",
                icon=":material/arrow_outward:",
                key=f"open_metric_{tile_key}",
                type="tertiary",
                width="stretch",
                help=f"Explore the {label.lower()} behind this tile",
            ):
                show_metric_details(metric_key, frame, signal_frame)


# ---------------------------------------------------------------------------
# TOP WORKSPACE NAVIGATION
# ---------------------------------------------------------------------------

navigation_labels = {
    "Executive pulse": ":material/dashboard_customize: Executive pulse",
    "Forecast studio": ":material/finance_mode: Forecast studio",
    "Anomaly intelligence": ":material/crisis_alert: Anomaly intelligence",
}

with st.container(key="top_navigation"):
    brand_column, navigation_column, status_column = st.columns(
        [1.15, 3.1, 1],
        vertical_alignment="center",
    )
    with brand_column:
        st.html(
            """
            <div class="brand-lockup">
                <div class="brand-mark" aria-label="iADAS waveform">
                    <span class="brand-wave" aria-hidden="true">∿</span>
                </div>
                <div>
                    <div class="brand-name">iADAS</div>
                    <div class="brand-sub">Intelligent Anomaly Detection &amp; Alerting System</div>
                </div>
            </div>
            """
        )
    with navigation_column:
        page = st.segmented_control(
            "Workspace navigation",
            options=list(navigation_labels),
            default="Executive pulse",
            required=True,
            format_func=lambda option: navigation_labels[option],
            label_visibility="collapsed",
            width="stretch",
            key="main_navigation",
            bind="query-params",
        )
    with status_column:
        with st.container(key="page_status", horizontal_alignment="right"):
            st.html(
                f"""
                <div class="live-status" title="Latest model observation">
                    <span class="live-status-dot"></span>
                    Monitoring · {datetime.now():%d %b %Y}
                </div>
                """
            )

st.space("medium")


# ---------------------------------------------------------------------------
# EXECUTIVE PULSE
# ---------------------------------------------------------------------------

if page == "Executive pulse":
    # Always use the same sensitivity selected in Forecast Studio.
    executive_confidence_pct = int(st.session_state["active_confidence_pct"])
    executive_results, executive_anomalies = generate_dynamic_forecast(
        results,
        executive_confidence_pct / 100.0,
    )

    exec_high = int((executive_anomalies["AnomalyType"] == "High Sales").sum())
    exec_low = int((executive_anomalies["AnomalyType"] == "Low Sales").sum())
    exec_accuracy = (
        100
        - abs(executive_results["y"] - executive_results["yhat"]).mean()
        / executive_results["y"].mean()
        * 100
    )

    page_header(
        "Executive pulse",
        "See the signal. Understand the exception.",
        "A compact view of the forecast and the exceptions worth investigating.",
        f"{len(executive_anomalies)} anomalies · {executive_confidence_pct}% prediction interval · Updated {datetime.now():%d %b %Y}",
    )

    with st.container(border=True, key="hero_panel"):
        hero_left, hero_right = st.columns([2.2, 1], vertical_alignment="center")
        with hero_left:
            st.subheader(f"{len(executive_anomalies)} anomalies across {len(executive_results)} weeks")
            st.caption("The forecast defines expected demand; exceptions outside the selected interval become signals.")
            st.markdown(
                f'<div class="anomaly-split"><span class="anomaly-positive">+{exec_high} above forecast</span>'
                f'<span class="anomaly-negative">-{exec_low} below forecast</span></div>',
                unsafe_allow_html=True,
            )
        with hero_right:
            if not executive_anomalies.empty:
                largest_exec = executive_anomalies.loc[
                    executive_anomalies["y"].sub(executive_anomalies["yhat"]).abs().idxmax()
                ]
                deviation = (
                    (largest_exec["y"] - largest_exec["yhat"]) / largest_exec["yhat"] * 100
                    if largest_exec["yhat"] else 0
                )
                st.metric(
                    "Largest deviation",
                    f"{deviation:+.1f}%",
                    f"{largest_exec['ds']:%d %b %Y}",
                    delta_color="normal",
                    delta_arrow="off",
                    border=True,
                )
            else:
                st.metric("Largest deviation", "—", "No anomaly detected", border=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("Anomalies", len(executive_anomalies), f"{exec_high} above · {exec_low} below", delta_color="off", border=True)
    k2.metric("Forecast accuracy", f"{exec_accuracy:.1f}%", border=True)
    k3.metric("Analysis window", f"{len(executive_results)} weeks", f"{executive_confidence_pct}% interval", border=True)

    section_header(
        "Demand signal",
        "Actual sales versus the AI expectation.",
        ":material/show_chart:",
    )

    chart_data = executive_results.rename(columns={"y": "Actual sales", "yhat": "AI forecast"})
    with st.container(border=True, key="executive_chart"):
        st.line_chart(
            chart_data,
            x="ds",
            y=["Actual sales", "AI forecast"],
            x_label="Week",
            y_label="Weekly sales (£)",
            color=[CYAN, VIOLET],
            height=430,
            width="stretch",
        )

# ---------------------------------------------------------------------------
# FORECAST STUDIO
# ---------------------------------------------------------------------------

elif page == "Forecast studio":
    page_header(
        "Interactive model view",
        "Explore the forecast. Find the exception.",
        "Zoom into weekly demand, compare actuals with the AI expectation, and switch instantly between trajectory and forecast variance.",
        "Tip: hover for exact values · use window presets to focus the analysis",
    )

    confidence_pct = st.slider(
        "Anomaly Detection Sensitivity",
        min_value=80,
        max_value=99,
        value=int(st.session_state["active_confidence_pct"]),
        step=1,
        format="%d%%",
        key="confidence_slider",
        on_change=persist_confidence,
        help=(
            "Lower confidence creates a narrower prediction interval and can "
            "surface more anomalies. Higher confidence is more conservative."
        ),
    )

    # Keep the persistent value synchronized even on the initial render.
    st.session_state["active_confidence_pct"] = int(confidence_pct)
    confidence_level = confidence_pct / 100.0
    st.caption(
        f"Prediction interval: {confidence_pct}% · "
        "95% is the baseline model configuration"
    )

    studio_results, studio_anomalies = generate_dynamic_forecast(
        results,
        confidence_level,
    )

    chart_view = st.session_state.get("forecast_chart_view", "Trajectory")
    if chart_view not in {"Trajectory", "Variance"}:
        chart_view = "Trajectory"

    default_layers = [
        ":material/blur_on: Confidence",
        ":material/crisis_alert: Signals",
    ]
    active_layers = st.session_state.get("forecast_layers", default_layers)
    show_confidence = (
        ":material/blur_on: Confidence" in active_layers
        and chart_view == "Trajectory"
    )
    show_anomalies = (
        ":material/crisis_alert: Signals" in active_layers
        and chart_view == "Trajectory"
    )

    with st.container(border=True, key="forecast_explorer"):
        window = st.segmented_control(
            "Analysis window",
            ["Full history", "Last 52 weeks", "Last 26 weeks", "Custom"],
            default="Full history",
            required=True,
            width="stretch",
            key="forecast_window",
        )

        if window == "Custom":
            custom_dates = st.date_input(
                "Custom date range",
                value=(results["ds"].min().date(), results["ds"].max().date()),
                min_value=results["ds"].min().date(),
                max_value=results["ds"].max().date(),
                key="custom_forecast_range",
            )
            if isinstance(custom_dates, (tuple, list)) and len(custom_dates) == 2:
                start_date = pd.Timestamp(custom_dates[0])
                end_date = pd.Timestamp(custom_dates[1])
            else:
                start_date, end_date = results["ds"].min(), results["ds"].max()
        else:
            start_date, end_date = resolve_window(window)

    filtered = studio_results[
        (studio_results["ds"] >= start_date) & (studio_results["ds"] <= end_date)
    ].copy()
    filtered_anomalies = studio_anomalies[
        (studio_anomalies["ds"] >= start_date)
        & (studio_anomalies["ds"] <= end_date)
    ].copy()

    if filtered.empty:
        st.warning(
            "No observations exist in the selected period.",
            icon=":material/calendar_month:",
        )
        st.stop()

    total_sales = float(filtered["y"].sum())
    forecast_sales = float(filtered["yhat"].sum())
    variance = total_sales - forecast_sales
    accuracy = (
        100
        - abs(filtered["y"] - filtered["yhat"]).mean()
        / filtered["y"].mean()
        * 100
    )

    high_filtered = int(
        (filtered_anomalies["AnomalyType"] == "High Sales").sum()
    )
    low_filtered = int(
        (filtered_anomalies["AnomalyType"] == "Low Sales").sum()
    )

    # Compact KPI row — no drill-down dialogs or duplicate diagnostics.
    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Actual sales",
        f"£{total_sales / 1_000_000:.2f}M",
        f"{len(filtered)} weeks",
        border=True,
    )

    metric_columns[1].metric(
        "AI forecast",
        f"£{forecast_sales / 1_000_000:.2f}M",
        f"£{variance:+,.0f} variance",
        border=True,
    )

    metric_columns[2].metric(
        "Forecast accuracy",
        f"{accuracy:.1f}%",
        border=True,
    )

    with metric_columns[3]:
        st.metric(
            "Anomalies",
            len(filtered_anomalies),
            border=True,
        )
        st.markdown(
            f'<div class="anomaly-split"><span class="anomaly-positive">+{high_filtered} above</span>'
            f'<span class="anomaly-negative">-{low_filtered} below</span></div>',
            unsafe_allow_html=True,
        )

    section_header(
        "Forecast trajectory",
        "Actual sales, AI forecast, confidence envelope, and detected exceptions.",
        ":material/show_chart:",
    )

    with st.container(border=True, key="forecast_chart_panel"):
        trajectory_data = filtered.rename(
            columns={
                "y": "Actual sales",
                "yhat": "AI forecast",
                "yhat_lower": "Lower confidence",
                "yhat_upper": "Upper confidence",
            }
        )

        series = ["Actual sales", "AI forecast", "Lower confidence", "Upper confidence"]

        st.line_chart(
            trajectory_data,
            x="ds",
            y=series,
            x_label="Week",
            y_label="Weekly sales (£)",
            color=[CYAN, VIOLET, "#64748B66", "#64748B66"],
            height=480,
            width="stretch",
        )

    if not filtered_anomalies.empty:
        st.caption(
            f"{len(filtered_anomalies)} detected events in this window · "
            f":green-badge[+{high_filtered} above] "
            f":red-badge[-{low_filtered} below]"
        )


# ---------------------------------------------------------------------------
# ANOMALY INTELLIGENCE
# ---------------------------------------------------------------------------

else:
    # Re-run the same Prophet interval logic used by Forecast Studio so this
    # page always reflects the currently selected sensitivity.
    confidence_pct = int(st.session_state["active_confidence_pct"])
    confidence_level = confidence_pct / 100.0
    intelligence_results, intelligence_anomalies = generate_dynamic_forecast(
        results,
        confidence_level,
    )
    dynamic_reports = dynamic_reports_from_anomalies(intelligence_anomalies)

    page_header(
        "Root-cause workspace",
        "Move from signal to explanation.",
        "Select a detected event to inspect financial impact, commercial drivers, and an AI-generated management response.",
        f"{len(dynamic_reports)} events at {confidence_pct}% sensitivity · transaction-level RCA",
    )

    high_dynamic = int((intelligence_anomalies["AnomalyType"] == "High Sales").sum())
    low_dynamic = int((intelligence_anomalies["AnomalyType"] == "Low Sales").sum())

    st.caption(
        f"Active sensitivity: **{confidence_pct}%** · "
        f":green-badge[+{high_dynamic} above forecast] "
        f":red-badge[-{low_dynamic} below forecast]"
    )

    if dynamic_reports.empty:
        st.info(
            "No anomalies were detected at the current sensitivity level.",
            icon=":material/check_circle:",
        )
        st.stop()

    selector_labels = {
        index: f"{row['Week']:%d %b %Y} · {row['Type']} · {row['Deviation %']:+.1f}%"
        for index, row in dynamic_reports.iterrows()
    }

    selected_index = st.segmented_control(
        "Select a detected event",
        options=list(selector_labels),
        default=list(selector_labels)[0],
        required=True,
        format_func=lambda index: selector_labels[index],
        width="stretch",
        key="anomaly_selector",
    )

    row = dynamic_reports.loc[selected_index]
    impact = abs(float(row["Actual Sales"]) - float(row["Forecast Sales"]))

    metric_columns = st.columns(4)
    with metric_columns[0]:
        st.metric(
            "Sales impact",
            f"£{impact:,.0f}",
            "Absolute variance",
            delta_color="orange",
            delta_arrow="off",
            border=True,
        )
    with metric_columns[1]:
        st.metric(
            "Actual sales",
            f"£{row['Actual Sales']:,.0f}",
            "Observed result",
            delta_color="green",
            delta_arrow="off",
            border=True,
        )
    with metric_columns[2]:
        st.metric(
            "AI forecast",
            f"£{row['Forecast Sales']:,.0f}",
            "Expected result",
            delta_color="violet",
            delta_arrow="off",
            border=True,
        )
    with metric_columns[3]:
        st.metric(
            "Deviation",
            f"{row['Deviation %']:+.1f}%",
            str(row["Type"]),
            delta_color="orange",
            border=True,
        )

    section_header(
        "Root-cause lens",
        "Transaction-derived context for the selected anomaly.",
        ":material/hub:",
    )

    root_cause_column, evidence_column = st.columns([1.15, 1], gap="medium")
    with root_cause_column:
        with st.container(border=True, key="root_cause_card"):
            st.badge(
                str(row["Type"]),
                icon=":material/priority_high:",
                color="orange" if row["Type"] == "Low Sales" else "violet",
            )
            st.subheader(str(row["Top Product"]))
            st.caption("Leading product · transaction-derived RCA")

            a, b = st.columns(2)
            with a:
                st.caption("Top customer")
                st.markdown(f"**{row['Top Customer']}**")
                st.caption(f"{row['Customer Contribution %']:.1f}% of weekly sales")
            with b:
                st.caption("Market")
                st.markdown(f"**{row['Top Country']}**")
                st.caption(f"{row['Country Contribution %']:.1f}% of weekly sales")

            st.divider()
            st.caption(
                f"Primary driver · **{row['Root Cause Driver']}** · "
                f"Product contribution {row['Product Contribution %']:.1f}%"
            )
            if float(row.get("Driver Change %", 0)) != 0:
                change = float(row["Driver Change %"])
                tone = "anomaly-positive" if change > 0 else "anomaly-negative"
                st.markdown(
                    f'<span class="{tone}">Driver change {change:+.1f}% vs prior available week</span>',
                    unsafe_allow_html=True,
                )

    with evidence_column:
        with st.container(border=True, key="evidence_card"):
            st.markdown("**Event evidence**")
            evidence = pd.DataFrame({
                "Measure": ["Actual sales", "AI forecast", "Variance"],
                "Value": [
                    float(row["Actual Sales"]),
                    float(row["Forecast Sales"]),
                    float(row["Actual Sales"]) - float(row["Forecast Sales"]),
                ],
            })
            st.bar_chart(
                evidence,
                x="Measure",
                y="Value",
                x_label=None,
                y_label="£",
                color=CYAN,
                height=220,
                width="stretch",
            )
            st.caption(
                f"Deviation: {row['Deviation %']:+.1f}% · "
                f"{row['Week']:%d %b %Y}"
            )

    section_header(
        "Generative analysis",
        "Management-ready interpretation and recommended actions.",
        ":material/auto_awesome:",
    )

    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}

    cache_key = (
        f"{confidence_pct}_{row['Week']}_{row['Type']}_"
        f"{row['Top Product']}_{row['Top Customer']}_{row['Top Country']}"
    )

    if cache_key not in st.session_state.llm_cache:
        with st.status("Synthesising the commercial signal...", expanded=False) as status:
            try:
                st.session_state.llm_cache[cache_key] = generate_insights(row)
                status.update(label="Decision brief ready", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="AI analysis service unavailable", state="error", expanded=False)
                st.error(
                    f"Forecast and RCA remain available, but Gemini did not return an analysis: {exc}",
                    icon=":material/cloud_off:",
                )

    if cache_key in st.session_state.llm_cache:
        with st.container(border=True, key="ai_panel"):
            st.markdown(st.session_state.llm_cache[cache_key])
