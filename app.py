import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from prophet import Prophet

from llm import generate_insights


# ---------------------------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SignalCast AI | Retail forecast intelligence",
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


def resolve_window(option: str):
    end = results["ds"].max()
    if option == "Last 26 weeks":
        return end - pd.Timedelta(weeks=26), end
    if option == "Last 52 weeks":
        return end - pd.Timedelta(weeks=52), end
    return results["ds"].min(), end


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
                <div class="brand-mark" aria-label="SignalCast waveform">
                    <span class="brand-wave" aria-hidden="true">∿</span>
                </div>
                <div>
                    <div class="brand-name">SignalCast <span>AI</span></div>
                    <div class="brand-sub">Forecast intelligence</div>
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
                    Monitoring · {results["ds"].max():%d %b}
                </div>
                """
            )

st.space("medium")


# ---------------------------------------------------------------------------
# EXECUTIVE PULSE
# ---------------------------------------------------------------------------

if page == "Executive pulse":
    page_header(
        "Executive intelligence",
        "Turn demand signals into decisive action.",
        "A forecast-led command center that separates meaningful retail exceptions from ordinary weekly noise.",
        f"Generated {datetime.now():%d %b %Y} · Prophet model · 95% prediction interval",
    )

    with st.container(border=True, key="hero_panel"):
        hero_left, hero_right = st.columns([2.1, 1], vertical_alignment="center")
        with hero_left:
            st.subheader(
                f"{len(anomalies)} meaningful events found across "
                f"{len(results)} weeks of retail activity"
            )
            st.write(
                "SignalCast learns the expected demand curve, evaluates every "
                "week against the model’s prediction interval, and elevates only "
                "the exceptions that merit investigation."
            )
            st.markdown(
                f":violet-badge[{high_count} demand spikes] "
                f":orange-badge[{low_count} demand drops] "
                f":green-badge[{accuracy_all:.1f}% forecast accuracy]"
            )
        with hero_right:
            st.metric(
                "Largest observed deviation",
                f"{largest['Deviation %']:+.1f}%",
                f"{largest['Week']} · {largest['Type']}",
                delta_color="violet",
                delta_arrow="off",
                border=True,
                chart_data=abs(results["y"] - results["yhat"]).tail(18).tolist(),
                chart_type="area",
            )

    section_header(
        "Portfolio pulse",
        "Performance across the complete model window.",
        ":material/monitoring:",
    )
    metric_columns = st.columns(4)
    interactive_metric_tile(
        metric_columns[0],
        tile_key="pulse_actual",
        metric_key="actual",
        label="Actual sales",
        value=f"£{total_sales_all / 1_000_000:.2f}M",
        delta="Observed retail value",
        frame=results,
        signal_frame=anomalies,
        chart_data=results["y"].tail(18).tolist(),
        chart_type="line",
        delta_color="off",
        delta_arrow="off",
    )
    interactive_metric_tile(
        metric_columns[1],
        tile_key="pulse_forecast",
        metric_key="forecast",
        label="AI forecast",
        value=f"£{forecast_sales_all / 1_000_000:.2f}M",
        delta=f"£{total_sales_all - forecast_sales_all:+,.0f} variance",
        frame=results,
        signal_frame=anomalies,
        chart_data=results["yhat"].tail(18).tolist(),
        chart_type="line",
        delta_color="violet",
    )
    interactive_metric_tile(
        metric_columns[2],
        tile_key="pulse_accuracy",
        metric_key="accuracy",
        label="Forecast accuracy",
        value=f"{accuracy_all:.1f}%",
        delta="Stable model performance",
        frame=results,
        signal_frame=anomalies,
        chart_data=(
            100
            - abs(results["y"] - results["yhat"])
            / results["y"].replace(0, pd.NA)
            * 100
        ).fillna(0).tail(18).tolist(),
        chart_type="area",
        delta_color="green",
        delta_arrow="off",
    )
    interactive_metric_tile(
        metric_columns[3],
        tile_key="pulse_anomalies",
        metric_key="anomalies",
        label="Signals detected",
        value=len(anomalies),
        delta=f"{high_count} high · {low_count} low",
        frame=results,
        signal_frame=anomalies,
        chart_data=[high_count, low_count],
        chart_type="bar",
        delta_color="orange",
        delta_arrow="off",
    )

    section_header(
        "Intelligence pipeline",
        "Seven automated stages convert transaction noise into an executive-ready decision signal.",
        ":material/account_tree:",
    )
    pipeline_steps = [
        ("01", "Retail transactions", "1.07M raw records", "database"),
        ("02", "Quality controls", "Cleaned and validated", "verified"),
        ("03", "Weekly aggregation", f"{len(results)} periods", "calendar_view_week"),
        ("04", "Prophet forecast", "Seasonality aware", "model_training"),
        ("05", "Interval analysis", "95% confidence", "analytics"),
        ("06", "Root-cause scan", "Driver attributed", "manage_search"),
        ("07", "Decision support", "Action generated", "lightbulb"),
    ]
    first_row = st.columns(4)
    second_row = st.columns(3)
    for column, (index, name, state, icon) in zip(
        first_row + second_row, pipeline_steps
    ):
        with column:
            with st.container(
                border=True,
                height=150,
                key=f"pipeline_stage_{index}",
            ):
                st.badge(index, color="violet")
                st.markdown(f":material/{icon}: **{name}**")
                st.caption(state)

    section_header(
        "Business value",
        "How forecast-based monitoring changes the operating rhythm.",
        ":material/rocket_launch:",
    )
    value_columns = st.columns(4)
    value_items = [
        (
            "trending_up",
            "Anticipate demand",
            "See unusual movement against an adaptive forecast instead of a rigid threshold.",
        ),
        (
            "notifications_active",
            "Reduce alert noise",
            "Focus investigation on statistically meaningful exceptions with business impact.",
        ),
        (
            "hub",
            "Explain the cause",
            "Connect every event to its strongest product, customer, market, and driver.",
        ),
        (
            "bolt",
            "Accelerate action",
            "Translate model output into clear inventory and revenue decisions.",
        ),
    ]
    for column, (icon, title, copy) in zip(value_columns, value_items):
        with column:
            with st.container(
                border=True,
                height=175,
                key=f"value_card_{icon}",
            ):
                st.markdown(f":material/{icon}: **{title}**")
                st.caption(copy)


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
        value=95,
        step=1,
        format="%d%%",
        help=(
            "Lower confidence creates a narrower prediction interval and can "
            "surface more anomalies. Higher confidence is more conservative."
        ),
    )
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

    metric_columns = st.columns(4)
    interactive_metric_tile(
        metric_columns[0],
        tile_key="studio_actual",
        metric_key="actual",
        label="Actual sales",
        value=f"£{total_sales / 1_000_000:.2f}M",
        delta=f"{len(filtered)} weekly observations",
        frame=filtered,
        signal_frame=filtered_anomalies,
        chart_data=filtered["y"].tolist(),
        chart_type="line",
        delta_color="off",
        delta_arrow="off",
    )
    interactive_metric_tile(
        metric_columns[1],
        tile_key="studio_forecast",
        metric_key="forecast",
        label="AI forecast",
        value=f"£{forecast_sales / 1_000_000:.2f}M",
        delta=f"£{variance:+,.0f} total variance",
        frame=filtered,
        signal_frame=filtered_anomalies,
        chart_data=filtered["yhat"].tolist(),
        chart_type="line",
        delta_color="violet",
    )
    interactive_metric_tile(
        metric_columns[2],
        tile_key="studio_accuracy",
        metric_key="accuracy",
        label="Forecast accuracy",
        value=f"{accuracy:.1f}%",
        delta="Mean absolute fit",
        frame=filtered,
        signal_frame=filtered_anomalies,
        chart_data=(
            100
            - abs(filtered["y"] - filtered["yhat"])
            / filtered["y"].replace(0, pd.NA)
            * 100
        ).fillna(0).tolist(),
        chart_type="area",
        delta_color="green",
        delta_arrow="off",
    )
    interactive_metric_tile(
        metric_columns[3],
        tile_key="studio_anomalies",
        metric_key="anomalies",
        label="Anomaly signals",
        value=len(filtered_anomalies),
        delta=(
            f"{(filtered_anomalies['AnomalyType'] == 'High Sales').sum()} high · "
            f"{(filtered_anomalies['AnomalyType'] == 'Low Sales').sum()} low"
        ),
        frame=filtered,
        signal_frame=filtered_anomalies,
        chart_data=[
            int((filtered_anomalies["AnomalyType"] == "High Sales").sum()),
            int((filtered_anomalies["AnomalyType"] == "Low Sales").sum()),
        ],
        chart_type="bar",
        delta_color="orange",
        delta_arrow="off",
    )

    section_header(
        "Forecast trajectory" if chart_view == "Trajectory" else "Forecast variance",
        (
            "Actual sales, AI forecast, confidence envelope, and detected exceptions."
            if chart_view == "Trajectory"
            else "Weekly distance above or below the AI forecast baseline."
        ),
        ":material/show_chart:",
    )
    with st.container(border=True, key="forecast_chart_panel"):
        if chart_view == "Trajectory":
            st.markdown(
                ":green-badge[Actual sales] "
                ":violet-badge[AI forecast] "
                + (
                    f":gray-badge[{confidence_pct}% confidence boundaries]"
                    if show_confidence
                    else ""
                )
            )
            trajectory_data = filtered.rename(
                columns={
                    "y": "Actual sales",
                    "yhat": "AI forecast",
                    "yhat_lower": "Lower confidence",
                    "yhat_upper": "Upper confidence",
                }
            )
            trajectory_series = ["Actual sales", "AI forecast"]
            trajectory_colors = [CYAN, VIOLET]
            if show_confidence:
                trajectory_series.extend(
                    ["Lower confidence", "Upper confidence"]
                )
                trajectory_colors.extend(["#64748B66", "#64748B66"])

            st.line_chart(
                trajectory_data,
                x="ds",
                y=trajectory_series,
                x_label="Week",
                y_label="Weekly sales (£)",
                color=trajectory_colors,
                height=510,
                width="stretch",
            )
        else:
            st.markdown(
                ":green-badge[Above forecast] "
                ":red-badge[Below forecast]"
            )
            variance_data = filtered[["ds", "y", "yhat"]].copy()
            variance_data["Variance"] = (
                variance_data["y"] - variance_data["yhat"]
            )
            variance_data["Direction"] = variance_data["Variance"].apply(
                lambda value: (
                    "Above forecast"
                    if value >= 0
                    else "Below forecast"
                )
            )
            st.bar_chart(
                variance_data,
                x="ds",
                y="Variance",
                x_label="Week",
                y_label="Variance to forecast (£)",
                color="Direction",
                height=510,
                width="stretch",
            )

        with st.container(border=True, key="chart_controls_dock"):
            view_column, layer_column = st.columns(
                [1, 1.45],
                vertical_alignment="bottom",
            )
            with view_column:
                st.segmented_control(
                    "Chart view",
                    ["Trajectory", "Variance"],
                    default="Trajectory",
                    required=True,
                    width="stretch",
                    key="forecast_chart_view",
                )
            with layer_column:
                st.pills(
                    "Chart layers",
                    default_layers,
                    selection_mode="multi",
                    default=default_layers,
                    disabled=chart_view == "Variance",
                    key="forecast_layers",
                    width="stretch",
                    help="Show or hide analytical layers on the trajectory.",
                )

        if (
            chart_view == "Trajectory"
            and show_anomalies
            and not filtered_anomalies.empty
        ):
            st.caption("Detected events in this analysis window")
            event_table = filtered_anomalies[
                ["ds", "AnomalyType", "y"]
            ].rename(
                columns={
                    "ds": "Week",
                    "AnomalyType": "Signal",
                    "y": "Actual sales",
                }
            )
            st.dataframe(
                event_table,
                hide_index=True,
                width="stretch",
                height="content",
                column_config={
                    "Week": st.column_config.DateColumn(
                        "Week",
                        format="DD MMM YYYY",
                    ),
                    "Signal": st.column_config.TextColumn("Signal"),
                    "Actual sales": st.column_config.NumberColumn(
                        "Actual sales (£)",
                        format="localized",
                    ),
                },
            )

    diagnostic_tabs = st.tabs(
        [
            ":material/health_metrics: Model diagnostics",
            ":material/priority_high: Priority signal",
            ":material/auto_awesome: AI decision brief",
        ]
    )
    with diagnostic_tabs[0]:
        diagnostic_columns = st.columns(3)
        diagnostic_columns[0].metric(
            "Mean absolute error",
            f"£{mae:,.0f}",
            "Average miss per week",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        diagnostic_columns[1].metric(
            "Root mean square error",
            f"£{rmse:,.0f}",
            "Penalises larger misses",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        diagnostic_columns[2].metric(
            "Prediction interval",
            f"{confidence_pct}%",
            "Statistical signal boundary",
            delta_color="violet",
            delta_arrow="off",
            border=True,
        )
    with diagnostic_tabs[1]:
        st.markdown(
            f"**{largest['Top Product']}** generated the strongest detected "
            f"exception in **{largest['Week']}**."
        )
        st.markdown(
            f":red-badge[{largest['Deviation %']:+.1f}% vs forecast] "
            f":violet-badge[£{abs(largest['Actual Sales'] - largest['Forecast Sales']):,.0f} impact] "
            f":orange-badge[{largest['Type']}]"
        )
    with diagnostic_tabs[2]:
        st.markdown(
            f"The model detected **{len(anomalies)} statistically significant "
            f"events** across the complete window: {high_count} unexpected "
            f"demand spikes and {low_count} demand drop. The largest movement "
            f"occurred in **{largest['Week']}** at "
            f"**{largest['Deviation %']:+.1f}%** versus expectation."
        )
        st.info(
            "Prioritise inventory review around the identified product driver, "
            "investigate low-demand causes, and maintain weekly exception monitoring.",
            icon=":material/lightbulb:",
        )


# ---------------------------------------------------------------------------
# ANOMALY INTELLIGENCE
# ---------------------------------------------------------------------------

else:
    page_header(
        "Root-cause workspace",
        "Move from signal to explanation.",
        "Select a detected event to inspect financial impact, commercial drivers, and an AI-generated management response.",
        f"{len(reports_df)} statistically significant events ready for investigation",
    )

    selector_labels = {
        index: f"{row['Week']} · {row['Type']}"
        for index, row in reports_df.iterrows()
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
    row = reports_df.loc[selected_index]
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
        "The strongest commercial contributors behind the selected event.",
        ":material/hub:",
    )
    root_cause_column, comparison_column = st.columns([1.15, 1])
    with root_cause_column:
        with st.container(border=True, height=310):
            st.badge(
                str(row["Type"]),
                icon=":material/priority_high:",
                color="orange" if row["Type"] == "Low Sales" else "violet",
            )
            st.subheader(str(row["Top Product"]))
            attribute_columns = st.columns(2)
            with attribute_columns[0].container(border=True, height=112):
                st.caption("Top customer")
                st.subheader(str(row["Top Customer"]))
            with attribute_columns[1].container(border=True, height=112):
                st.caption("Market")
                st.subheader(str(row["Top Country"]))
            st.caption(f"Primary driver · {row['Root Cause Driver']}")
    with comparison_column:
        with st.container(border=True, height=310):
            st.markdown("**Financial contribution**")
            comparison_data = pd.DataFrame(
                {
                    "Measure": [
                        "Actual sales",
                        "AI forecast",
                        "Root-cause contribution",
                    ],
                    "Value": [
                        float(row["Actual Sales"]),
                        float(row["Forecast Sales"]),
                        abs(float(row["Driver Increase"])),
                    ],
                }
            )
            st.bar_chart(
                comparison_data,
                x="Measure",
                y="Value",
                x_label=None,
                y_label="Value (£)",
                color=CYAN,
                horizontal=True,
                width="stretch",
                height=220,
            )

    section_header(
        "Generative analysis",
        "Turn this event into a concise management-ready action brief.",
        ":material/auto_awesome:",
    )
    with st.container(border=True, key="ai_panel"):
        analysis_column, status_column = st.columns(
            [3, 1],
            vertical_alignment="center",
        )
        with analysis_column:
            st.markdown("**Gemini-powered decision support**")
            st.caption(
                "The assistant combines forecast deviation with product, customer, "
                "market, and driver context to generate risks and recommended actions."
            )
        with status_column:
            st.badge(
                "Automatic analysis",
                icon=":material/bolt:",
                color="blue",
            )

    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}

    cache_key = f"{row['Week']}_{row['Top Product']}"
    brief_slot = st.container()
    with brief_slot:
        if cache_key not in st.session_state.llm_cache:
            with st.status(
                "Synthesising the commercial signal...",
                expanded=True,
            ) as status:
                try:
                    st.session_state.llm_cache[cache_key] = generate_insights(row)
                    status.update(
                        label="Decision brief ready",
                        state="complete",
                        expanded=False,
                    )
                except Exception:
                    status.update(
                        label="AI analysis service unavailable",
                        state="error",
                        expanded=False,
                    )
                    st.error(
                        "The forecast and root-cause results remain available, "
                        "but the Gemini service did not return an analysis.",
                        icon=":material/cloud_off:",
                    )

        if cache_key in st.session_state.llm_cache:
            with st.container(border=True):
                st.markdown(st.session_state.llm_cache[cache_key])

    section_header(
        "Signal register",
        "Explore, sort, and export the complete anomaly audit trail.",
        ":material/table_chart:",
    )
    with st.container(border=True, key="register_panel"):
        table_header, export_column = st.columns(
            [4, 1],
            vertical_alignment="center",
        )
        with table_header:
            st.caption(
                "Columns are formatted for rapid executive review. "
                "Use the table toolbar to search or download."
            )
        with export_column:
            csv = reports_df.to_csv(index=False).encode()
            st.download_button(
                "Export CSV",
                csv,
                "Anomaly_Report.csv",
                "text/csv",
                icon=":material/download:",
                width="stretch",
            )

        register_columns = [
            "Week",
            "Type",
            "Actual Sales",
            "Forecast Sales",
            "Deviation %",
            "Top Product",
            "Top Country",
            "Driver Increase",
        ]
        st.dataframe(
            reports_df[register_columns],
            hide_index=True,
            width="stretch",
            height="content",
            column_config={
                "Week": st.column_config.DateColumn(
                    "Week",
                    format="DD MMM YYYY",
                    pinned=True,
                ),
                "Type": st.column_config.TextColumn("Signal"),
                "Actual Sales": st.column_config.NumberColumn(
                    "Actual sales (£)",
                    format="localized",
                ),
                "Forecast Sales": st.column_config.NumberColumn(
                    "AI forecast (£)",
                    format="localized",
                ),
                "Deviation %": st.column_config.NumberColumn(
                    "Deviation",
                    format="%+.1f%%",
                ),
                "Top Product": st.column_config.TextColumn(
                    "Leading product",
                    width="large",
                ),
                "Top Country": st.column_config.TextColumn("Market"),
                "Driver Increase": st.column_config.NumberColumn(
                    "Driver contribution (£)",
                    format="localized",
                ),
            },
        )
