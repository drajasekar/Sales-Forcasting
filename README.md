# Sales-Forecasting

# 📊 Online Sales Forecast Dashboard

A Streamlit web app for forecasting sales/collection performance using **Prophet**, **Random Forest**, or **XGBoost**. Users can upload an Excel file, explore historical sales, generate forecasts, and download the forecasted data.

---

## 🔹 Features

* Upload your own Excel file for analysis (`.xlsx` or `.xls`).
* Choose from **Prophet**, **Random Forest**, or **XGBoost** models for forecasting.
* Flexible **forecast horizon** (1–12 months) and **training period** selection.
* Filter by **outlets**.
* View **Year-wise sales** (last 4 years + forecast) with interactive Plotly bar charts.
* Monthly sales trends (last 12 months + next forecasted months).
* KPI summary: Total sales, forecasted sales, average monthly forecast.
* Forecast data table with **Excel/CSV download**.
* AI-generated summary insights for easy interpretation.
* Indian currency formatting (₹, K, L, Cr).

---

## 🔹 Requirements

Python libraries required are listed in [`requirements.txt`](requirements.txt):

```
streamlit
pandas
numpy
plotly
prophet
scikit-learn
xgboost
openpyxl
```

---

## 🔹 Usage

1. Clone the repository or copy the files to your local system.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. In the app:

* Upload your Excel file containing the following columns:

  * `Order_YearMonth` → Date of the order (`YYYY-MM` format recommended)
  * `Total_OrderPrice` → Total order value
  * `ONL_Outlet` → Outlet name (optional)

* Select the **forecast model**.

* Adjust **forecast horizon** and **training years**.

* Explore visualizations, KPIs, and forecast tables.

* Download the forecast CSV for reporting.

---

## 🔹 Excel File Requirements

* **File type:** `.xlsx` or `.xls`
* **Columns required:**

  * `YearMonth` → Order Date (monthly granularity)
  * `Price` → Sales/Collection value (numeric)
  * `Outlet` → Outlet name (for filtering, optional)

> Column names are **case-sensitive**. Extra columns are ignored.

---

## 🔹 Notes

* The app automatically formats sales values into Indian numbering system (K, L, Cr).
* For large Excel files (~90k+ rows), initial load and model fitting may take some time.
* Prophet requires either `pystan` or `cmdstanpy` (installed automatically with recent `prophet` versions).
---

## 🔹 Author

Rajasekar D.
