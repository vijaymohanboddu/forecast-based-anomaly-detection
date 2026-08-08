# 📈 iADAS — Intelligent Anomaly Detection & Alerting System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-orange)
![GenAI](https://img.shields.io/badge/GenAI-Gemini-purple)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-v2.0%20Released-success)

## AI-Powered Retail Forecasting, Anomaly Detection & Decision Support

**iADAS** is an end-to-end AI-powered retail analytics and decision-support system.

It combines:

- **Prophet-based weekly sales forecasting**
- **Forecast-interval anomaly detection**
- **Dynamic anomaly sensitivity**
- **Transaction-level root cause analysis**
- **Generative AI executive interpretation using Gemini**
- **Interactive Streamlit dashboards**
- **Executive-ready business recommendations**

The objective is to move from:

> **Signal → Explanation → Action**

rather than simply reporting that an anomaly occurred.

### 🚀 Live Demo

**iADAS:**  
https://i-adas.streamlit.app/

---

# Version 2.0

Version 2.0 turns the original forecasting prototype into an interactive **AI decision-support application**.

### What's new in v2.0

- ✅ Interactive **iADAS Streamlit application**
- ✅ Executive Pulse dashboard
- ✅ Forecast Studio with adjustable anomaly sensitivity
- ✅ Anomaly Intelligence workspace
- ✅ Dynamic anomaly detection based on prediction intervals
- ✅ Transaction-derived root cause analysis
- ✅ Gemini-powered management insights
- ✅ Automated executive summaries and recommendations
- ✅ Dynamic monitoring date
- ✅ High/Low anomaly visual coding
- ✅ Forecast and anomaly visualization
- ✅ Dark Mode and Light Mode
- ✅ Responsive management-oriented UI
- ✅ Cached Gemini responses to reduce repeated API calls
- ✅ Live Streamlit deployment

---

# 🎯 Business Problem

Retail managers monitor thousands of transactions but often struggle to identify abnormal sales patterns in time.

Traditional anomaly detection systems commonly rely on fixed thresholds. These approaches can generate excessive false positives because they do not account for:

- Seasonality
- Long-term trends
- Business growth
- Holiday effects
- Expected demand variability

iADAS addresses this by forecasting expected sales and identifying events where actual sales move outside the model's expected prediction interval.

The system then goes one step further by explaining **why the anomaly may have occurred** and translating the evidence into **management actions**.

---

# 💡 Solution Architecture

```text
Raw Retail Transactions
          │
          ▼
     Data Cleaning
          │
          ▼
  Feature Engineering
          │
          ▼
 Weekly Sales Aggregation
          │
          ▼
   Prophet Forecast
          │
          ▼
 Prediction Intervals
          │
          ▼
  Anomaly Detection
          │
          ▼
 Transaction-level RCA
          │
          ▼
 Gemini Business Analysis
          │
          ▼
 Executive Recommendations
```

---

# 🧠 Application Architecture

iADAS is organized into three primary workspaces.

### 1. Executive Pulse

Management-oriented view of:

- Current forecast signal
- Actual vs expected sales
- High and low sales anomalies
- Overall anomaly activity
- Key business indicators

The dashboard reflects the **active anomaly sensitivity** selected in Forecast Studio.

### 2. Forecast Studio

Used to explore the forecasting model and anomaly sensitivity.

Users can adjust the prediction interval and dynamically refresh the anomaly register.

This allows users to see how the number and composition of detected anomalies changes as detection sensitivity changes.

### 3. Anomaly Intelligence

The root-cause workspace moves from:

> **Signal → Explanation**

For each detected anomaly, the application can present:

- Actual sales
- Forecast sales
- Variance
- Percentage deviation
- Leading product
- Top customer
- Top country / market
- Transaction-derived driver
- Contribution metrics
- Week-over-week evidence
- Gemini-generated management interpretation
- Recommended actions

Where evidence is insufficient, the system treats the explanation as a hypothesis rather than presenting it as fact.

---

# 📊 Dataset

**Online Retail II Dataset**

- Source: Kaggle / UCI Machine Learning Repository
- Period: December 2009 – December 2011
- Original transactions: **1,067,371**

Dataset:

https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

---

# 🧹 Data Cleaning

The following preprocessing steps were performed:

- Removed duplicate transactions
- Removed cancelled invoices
- Removed missing product descriptions
- Removed invalid quantities
- Removed zero-priced items
- Converted `InvoiceDate` to datetime
- Created `TotalSales`

Final clean dataset:

**1,007,913 transactions**

---

# 📈 Forecasting Model

### Model

**Prophet**

The model captures:

- Long-term sales trend
- Yearly seasonality
- Expected weekly sales

Sales are aggregated into weekly observations before forecasting.

### Forecast Dataset

- Weekly observations: **106**
- Average weekly sales: approximately **£193,172**
- MAE: **£27,549**
- RMSE: **£36,860**

---

# 📐 Forecast-Based Anomaly Detection

Instead of using a static rule such as:

```text
Actual Sales > Fixed Threshold
```

iADAS evaluates actual sales against the model's expected prediction interval.

Conceptually:

```text
             Upper Prediction Bound
                    ▲
                    │
       HIGH         │   ← High Sales Anomaly
                    │
──────────── Forecast ─────────────
                    │
       LOW          │   ← Low Sales Anomaly
                    │
                    ▼
             Lower Prediction Bound
```

### Dynamic Sensitivity

The original model used a **95% prediction interval**, producing the baseline set of **5 meaningful anomalies**:

- 4 High Sales
- 1 Low Sales

In v2.0, the prediction-interval sensitivity is exposed in the application so users can explore a broader or narrower anomaly set interactively.

---

# 🔎 Root Cause Analysis

Detected anomalies are enriched using transaction-level evidence.

The RCA layer evaluates concentration and contribution across:

- Product
- Customer
- Country / Market
- Transaction-derived drivers
- Week-over-week changes

This helps move from:

> **"Sales were unusual."**

to:

> **"Sales were unusual, and these transaction-level factors appear to explain the deviation."**

---

# 🤖 Generative AI — Gemini

iADAS uses **Google Gemini** to transform anomaly and RCA evidence into a concise management brief.

The AI output is structured into:

### Executive Summary

What happened and why it matters.

### Business Interpretation

Evidence-based interpretation of the event.

### Business Risks

Potential inventory, customer, revenue and operational risks.

### Recommended Actions

3–5 prioritized management actions.

### Management Takeaway

One concise senior-leadership takeaway.

The prompt instructs the model to use only supplied evidence, avoid unsupported figures, and distinguish observed facts from hypotheses.

Gemini responses are cached to reduce repeated API calls for the same anomaly.

---

# 📊 Model Performance

| Metric | Value |
|---|---:|
| Original Transactions | 1,067,371 |
| Clean Transactions | 1,007,913 |
| Weekly Observations | 106 |
| Forecast Model | Prophet |
| Average Weekly Sales | £193,172 |
| MAE | £27,549 |
| RMSE | £36,860 |
| Baseline Anomalies at 95% | 5 |

**Important:** the anomaly count is dynamic in v2.0 and depends on the active prediction-interval sensitivity.

---

# 🎯 Key Results

- 📊 Analysed **1,067,371** retail transactions
- 🧹 Cleaned dataset to **1,007,913** records
- 📅 Aggregated sales into **106 weekly observations**
- 🤖 Built a **Prophet forecasting model**
- 🎯 Achieved **MAE = £27,549**
- 📉 Detects anomalies using forecast prediction intervals
- 🎚️ Allows users to dynamically adjust anomaly sensitivity
- 🔎 Performs transaction-derived root cause analysis
- 🧠 Generates AI-powered management interpretation
- 💼 Converts analytical signals into recommended business actions
- 🖥️ Delivered as an interactive Streamlit application

---

# 🖥️ Application

### Executive Pulse

Management overview of the current forecast and anomaly landscape.

### Forecast Studio

Interactive forecast exploration with adjustable anomaly sensitivity and anomaly visualization.

### Anomaly Intelligence

Detailed anomaly investigation with financial impact, RCA evidence and Gemini-generated management response.

---

# 📁 Repository Structure

```text
forecast-based-anomaly-detection/
│
├── notebook/
│   └── forecast-based-anomaly-detection.ipynb
│
├── images/
│   ├── weekly_sales.png
│   ├── forecast1.png
│   ├── forecast2.png
│   └── forecast_final.png
│
├── reports/
│   ├── Anomaly_Report.xlsx
│   ├── Anomaly_Report.csv
│   └── Executive_Summary.xlsx
│
├── presentation/
│   └── ISB_AI_Retail_Sales_Forecasting.pdf
│
├── app.py
├── llm.py
├── dashboard_data.pkl
├── weekly_rca.pkl
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🛠️ Technologies Used

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Plotly
- Prophet
- Scikit-learn
- Streamlit
- Google Gemini / `google-genai`
- Google Colab
- OpenPyXL

---

# 🔐 Gemini Configuration

The Streamlit application expects the Gemini API key to be configured through Streamlit Secrets.

Example:

```toml
GEMINI_API_KEY = "your-api-key"
```

**Never commit the actual API key to GitHub.**

---

# ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the Gemini API key using Streamlit secrets.

Then run:

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

---

# 📌 Business Impact

Compared with static threshold monitoring, iADAS aims to:

- Reduce false-positive alerts
- Detect unusual sales behaviour relative to expected demand
- Identify likely commercial drivers
- Provide explainable root-cause analysis
- Support inventory planning
- Improve demand visibility
- Enable proactive operational decision-making
- Reduce the gap between analytical detection and management action

---

# 🗺️ Project Evolution

### Version 1.0

**Forecast → Detect**

- Data cleaning
- Weekly aggregation
- Prophet forecasting
- Prediction-interval anomaly detection
- Initial RCA
- Analytical reports

### Version 2.0

**Forecast → Detect → Explain → Act**

- Interactive Streamlit application
- Dynamic anomaly sensitivity
- Executive Pulse
- Forecast Studio
- Anomaly Intelligence
- Transaction-derived RCA
- Gemini-powered management analysis
- Executive recommendations
- Responsive UI
- Dark / Light mode
- Cached AI responses
- Live deployment

---

# 🚀 Future Roadmap

Potential future enhancements:

- ⏳ Real-time retail data ingestion
- ⏳ Automated email / Slack alerts
- ⏳ Persistent anomaly history
- ⏳ Multi-product forecasting
- ⏳ Multi-market forecasting
- ⏳ Automated alert prioritisation
- ⏳ Feedback loop for RCA validation
- ⏳ Production monitoring and model drift detection

---

# 👤 Author

**Vijay Mohan Boddu**

Engineering Manager | AI Enthusiast | ISB – AI in Business

---

# 📄 License

This project is released under the MIT License.
