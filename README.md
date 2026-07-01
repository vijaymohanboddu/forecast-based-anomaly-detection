# 📈 Forecast-Based Anomaly Detection for Retail Sales

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

An end-to-end AI solution for forecasting weekly retail sales and detecting meaningful business anomalies using Facebook Prophet.

This project demonstrates how predictive AI can replace static threshold-based monitoring with forecast-driven anomaly detection, reducing false positives while providing explainable business insights.

---

## Business Problem

Traditional anomaly detection systems use fixed thresholds to detect unusual sales activity.

This approach generates excessive false positives because it ignores:

- Seasonality
- Long-term trends
- Business growth
- Holiday effects

This project addresses the problem by forecasting expected sales and detecting anomalies only when actual sales significantly deviate from the forecast.

---

## Solution Architecture

```
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
 Prophet Forecast Model
            │
            ▼
Prediction Interval Analysis
            │
            ▼
 Sales Anomaly Detection
            │
            ▼
 Root Cause Analysis
            │
            ▼
 Business Recommendations
```

---

## Dataset

**Online Retail II Dataset**

- Source: Kaggle (UCI Machine Learning Repository)
- Period: December 2009 – December 2011
- Original Transactions: **1,067,371**

Dataset Link:

https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

---

## Data Cleaning

The following preprocessing steps were performed:

- Removed duplicate transactions
- Removed cancelled invoices
- Removed missing product descriptions
- Removed invalid quantities
- Removed zero-priced items
- Converted InvoiceDate to datetime
- Created TotalSales feature

Final clean dataset:

**1,007,913 transactions**

---

## Forecasting Model

Model Used:

**Facebook Prophet**

The model learns:

- Long-term sales trend
- Yearly seasonality
- Expected weekly sales

Forecasts are generated for weekly aggregated sales.

---

## Model Performance

| Metric | Value |
|---------|------:|
| Original Transactions | 1,067,371 |
| Clean Transactions | 1,007,913 |
| Weekly Observations | 106 |
| Forecast Model | Prophet |
| MAE | £27,549 |
| RMSE | £36,860 |
| High Sales Anomalies | 4 |
| Low Sales Anomalies | 1 |

---

## Project Outputs

The project generates:

- Weekly Sales Forecast
- Forecast vs Actual Comparison
- Forecast-Based Anomaly Detection
- Root Cause Analysis
- Executive Business Report
- Executive Summary Dashboard

---

## Repository Structure

```
forecast-based-anomaly-detection/

│
├── notebook/
│   └── forecast-based-anomaly-detection.ipynb
│
├── images/
│   ├── weekly_sales.png
│   ├── prophet_forecast.png
│   ├── prophet_components.png
│   └── anomaly_detection.png
│
├── reports/
│   ├── Anomaly_Report.xlsx
│   ├── Anomaly_Report.csv
│   └── Executive_Summary.xlsx
│
├── presentation/
│   └── ISB_AI_Retail_Sales_Forecasting.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Prophet
- Scikit-learn
- Google Colab

---

## Future Enhancements

- Interactive Streamlit Dashboard
- Customer-Level Forecasting
- Product-Level Forecasting
- LLM-powered Business Insights
- Real-Time Sales Monitoring
- Automated Email Alerts

---

## Author

**Vijay Mohan Boddu**

Engineering Manager | AI Enthusiast | ISB – AI in Business

---

## License

This project is released under the MIT License.
