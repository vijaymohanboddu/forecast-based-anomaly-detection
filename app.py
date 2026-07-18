import pickle
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from llm import generate_insights

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Retail AI Dashboard",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# STYLE
# --------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

div[data-testid="metric-container"]{
    background:#111827;
    border:1px solid #2d3748;
    border-radius:12px;
    padding:18px;
}

h1{
    color:#0ea5e9;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with open("dashboard_data.pkl","rb") as f:
    data = pickle.load(f)

results = data["results"]
anomalies = data["anomalies"]
reports_df = data["reports_df"]
mae = data["mae"]
rmse = data["rmse"]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

# st.sidebar.title("Retail AI")
st.sidebar.image(
    "https://img.icons8.com/fluency/96/artificial-intelligence.png",
    width=70
)

st.sidebar.title("Retail AI")

st.sidebar.caption("ISB Capstone Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Summary",
        "Dashboard",
        "Anomalies"
    ]
)

# --------------------------------------------------
# KPI VALUES
# --------------------------------------------------

total_sales = results["y"].sum()
forecast_sales = results["yhat"].sum()

accuracy = (
    100
    - abs(results["y"]-results["yhat"]).mean()
    / results["y"].mean()
    *100
)

# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------

# --------------------------------------------------
# EXECUTIVE SUMMARY
# --------------------------------------------------

if page == "Executive Summary":

    c1, c2 = st.columns([4, 1])

    with c1:
        st.title("📊 Retail AI Decision Support Dashboard")

    with c2:
        st.metric(
            "Generated",
            datetime.now().strftime("%d-%b-%Y")
        )

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Total Weekly Observations", len(results))
        st.metric("Detected Anomalies", len(anomalies))
        st.metric("Forecast Accuracy", f"{accuracy:.2f}%")

    with c2:
        st.metric("MAE", f"{mae:,.0f}")
        st.metric("RMSE", f"{rmse:,.0f}")
        st.metric("Total Sales", f"£{results['y'].sum():,.0f}")

    st.divider()

    st.subheader("Business Objective")

    st.write("""
        Detect abnormal sales patterns automatically using AI forecasting so that business users can investigate unusual events before they impact inventory, revenue, or customer satisfaction.
        """)

    st.subheader("Solution Architecture")

    st.graphviz_chart("""
        digraph {

            rankdir=LR;

            node [
                shape=box
                style="rounded,filled"
                fillcolor="#1f77b4"
                fontcolor="white"
                fontsize=12
            ]

            A [label="Retail\\nTransactions"]
            B [label="Data\\nCleaning"]
            C [label="Weekly\\nAggregation"]
            D [label="Prophet\\nForecast"]
            E [label="Anomaly\\nDetection"]
            F [label="Root Cause\\nAnalysis"]
            G [label="AI Decision\\nSupport Dashboard"]

            A -> B
            B -> C
            C -> D
            D -> E
            E -> F
            F -> G

        }
    """)

    st.subheader("Business Impact")

    st.info("""
    • Automated weekly sales monitoring

    • Early identification of unusual demand

    • Faster root-cause investigation

    • Better inventory planning

    • Improved executive decision making
    """)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("📊 Retail AI Decision Support Dashboard")

    st.caption("Forecast-Based Anomaly Detection using Facebook Prophet")

    start_date, end_date = st.slider(
        "Select Time Range",
        min_value=results["ds"].min().to_pydatetime(),
        max_value=results["ds"].max().to_pydatetime(),
        value=(
            results["ds"].min().to_pydatetime(),
            results["ds"].max().to_pydatetime()
        )
    )

    filtered = results[
        (results["ds"] >= start_date) &
        (results["ds"] <= end_date)
    ]

    filtered_anomalies = anomalies[
        (anomalies["ds"] >= start_date) &
        (anomalies["ds"] <= end_date)
    ]

    total_sales = filtered["y"].sum()
    forecast_sales = filtered["yhat"].sum()

    accuracy = (
        100
        - abs(filtered["y"] - filtered["yhat"]).mean()
        / filtered["y"].mean()
        * 100
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"£{total_sales:,.0f}")
    c2.metric("Forecast Sales", f"£{forecast_sales:,.0f}")
    c3.metric("Accuracy", f"{accuracy:.2f}%")
    c4.metric("Anomalies", len(filtered_anomalies))

    st.divider()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered["ds"],
            y=filtered["y"],
            mode="lines",
            name="Actual",
            line=dict(width=3)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=filtered["ds"],
            y=filtered["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(dash="dash")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=filtered["ds"],
            y=filtered["yhat_upper"],
            line=dict(width=0),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=filtered["ds"],
            y=filtered["yhat_lower"],
            fill="tonexty",
            fillcolor="rgba(0,150,255,0.15)",
            line=dict(width=0),
            name="95% Confidence"
        )
    )

    high = filtered_anomalies[
        filtered_anomalies["AnomalyType"] == "High Sales"
    ]

    low = filtered_anomalies[
        filtered_anomalies["AnomalyType"] == "Low Sales"
    ]

    fig.add_trace(
        go.Scatter(
            x=high["ds"],
            y=high["y"],
            mode="markers+text",
            text=["⬆"] * len(high),
            textposition="top center",
            marker=dict(size=14, color="red"),
            name="High Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=low["ds"],
            y=low["y"],
            mode="markers+text",
            text=["⬇"] * len(low),
            textposition="bottom center",
            marker=dict(size=14, color="lime"),
            name="Low Sales"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        legend_orientation="h",
        xaxis_title="Week",
        yaxis_title="Weekly Sales (£)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")

    st.divider()

    # --------------------------------------------------
    # AI EXECUTIVE SUMMARY
    # --------------------------------------------------

    high_count = len(anomalies[anomalies["AnomalyType"] == "High Sales"])
    low_count = len(anomalies[anomalies["AnomalyType"] == "Low Sales"])

    largest = reports_df.iloc[
        reports_df["Deviation %"].abs().idxmax()
    ]

    st.subheader("🤖 AI Executive Summary")

    st.success(f"""
    ### Key Insights

    • **{len(anomalies)} statistically significant anomalies** were detected.

    • **{high_count} High Sales** anomalies indicate unexpected demand spikes.

    • **{low_count} Low Sales** anomalies indicate unexpected demand drops.

    • The **largest deviation** occurred during **{largest['Week']}**
    with a deviation of **{largest['Deviation %']:.1f}%**.

    • Overall forecasting performance remains stable with

    - **MAE:** {mae:,.0f}
    - **RMSE:** {rmse:,.0f}

    ### Recommended Business Actions

    ✅ Review inventory planning before seasonal peaks.

    ✅ Investigate unexpected demand drops.

    ✅ Monitor high-impact products more frequently.

    ✅ Continue weekly anomaly monitoring using the forecasting model.
    """)

# --------------------------------------------------
# ANOMALY PAGE
# --------------------------------------------------

if page == "Anomalies":

    st.title("Detected Anomalies")

    choice = st.selectbox(
        "Select Anomaly",
        reports_df.apply(
            lambda x: f"{x['Week']}  |  {x['Type']}  |  {x['Top Product'][:40]}",
            axis=1
        )
    )

    selected = reports_df.iloc[
        reports_df.apply(
            lambda x: f"{x['Week']}  |  {x['Type']}  |  {x['Top Product'][:40]}",
            axis=1
        ).tolist().index(choice)
    ]

    row = selected

    # row = reports_df[
    #     reports_df["Week"]==choice
    # ].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Sales Impact",
        f"£{abs(row['Actual Sales']-row['Forecast Sales']):,.0f}"
    )

    c2.metric(
        "Driver Increase",
        f"£{row['Driver Increase']:,.0f}"
    )

    c3.metric(
        "Type",
        row["Type"]
    )

    c1,c2 = st.columns(2)

    with c1:
        st.metric(
            "Actual Sales",
            f"£{row['Actual Sales']:,.0f}"
        )

        st.metric(
            "Forecast",
            f"£{row['Forecast Sales']:,.0f}"
        )

        st.metric(
            "Deviation %",
            f"{row['Deviation %']:.1f}%"
        )

    with c2:

        st.write("### Root Cause")

        st.write(
            f"**Product:** {row['Top Product']}"
        )

        st.write(
            f"**Customer:** {row['Top Customer']}"
        )

        st.write(
            f"**Country:** {row['Top Country']}"
        )

        st.write(
            f"**Driver:** {row['Root Cause Driver']}"
        )

    st.divider()

    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}

    key = f"{row['Week']}_{row['Top Product']}"

    if st.button("🤖 Generate AI Insights"):

        if key not in st.session_state.llm_cache:

            with st.spinner("Generating AI business analysis..."):
                # st.write(row)
                # st.write(row.index.tolist())
                # st.write("Selected row:")
                # st.dataframe(row.to_frame())
                st.session_state.llm_cache[key] = generate_insights(row)

    st.markdown(
        st.session_state.llm_cache.get(
            key,
            "*Click **Generate AI Insights** to generate an executive business analysis.*"
        )
    )

    st.divider()

    st.dataframe(
        reports_df,
        use_container_width=True,
        hide_index=True
    )

    csv = reports_df.to_csv(index=False).encode()

    st.download_button(
        "Download CSV",
        csv,
        "Anomaly_Report.csv",
        "text/csv"
    )

    st.subheader("Top Contributing Product")

    fig = go.Figure()

    fig.add_bar(
        x=[row["Top Product"]],
        y=[row["Driver Increase"]],
        text=[f"£{row['Driver Increase']:,.0f}"],
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Product",
        yaxis_title="Sales Contribution (£)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )