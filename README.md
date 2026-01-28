# Sales-Forecasting
# 📊 Online Sales Forecast Dashboard

A Streamlit web app for forecasting sales/collection performance using **Prophet**, **Random Forest**, or **XGBoost**. Users can upload an Excel file, explore historical sales, generate forecasts, and download the forecasted data.

---

## 🔹 Features

- Upload your own Excel file for analysis (`.xlsx` or `.xls`).  
- Choose from **Prophet**, **Random Forest**, or **XGBoost** models for forecasting.  
- Flexible **forecast horizon** (1–12 months) and **training period** selection.  
- Filter by **outlets**.  
- View **Year-wise sales** (last 4 years + forecast) with interactive Plotly bar charts.  
- Monthly sales trends (last 12 months + next forecasted months).  
- KPI summary: Total sales, forecasted sales, average monthly forecast.  
- Forecast data table with **Excel/CSV download**.  
- AI-generated summary insights for easy interpretation.  
- Indian currency formatting (₹, K, L, Cr).

---

## 🔹 Requirements

Python libraries required are listed in [`requirements.txt`](requirements.txt):

```bash
streamlit
pandas
numpy
plotly
prophet
scikit-learn
xgboost
openpyxl
