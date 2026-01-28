import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Online Sales Forecast", layout="wide")
st.title("Sales Forecast")

# =========================================================
# HELPERS - Indian Numbering System
# =========================================================
def format_currency(v):
    if pd.isna(v):
        return "₹0"
    v = max(v, 0)
    if v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    elif v >= 1e5:
        return f"₹{v/1e5:.2f} L"
    elif v >= 1e3:
        return f"₹{v/1e3:.2f} K"
    else:
        return f"₹{v:,.0f}"

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    # ---------------------------------------------
    # NORMALIZE COLUMN NAMES (AUTO-DETECT FORMAT)
    # ---------------------------------------------
    column_map = {}

    if {"SS_SourceSite", "Bill_YearMonth", "TotalAmount"}.issubset(df.columns):
        column_map = {
            "SS_SourceSite": "ONL_Outlet",
            "Bill_YearMonth": "Order_YearMonth",
            "TotalAmount": "Total_OrderPrice"
        }

    elif {"ONL_Outlet", "Order_YearMonth", "Total_OrderPrice"}.issubset(df.columns):
        column_map = {}  # already correct

    else:
        raise ValueError(
            "Invalid Excel format. Required columns:\n"
            "• SS_SourceSite, Bill_YearMonth, TotalAmount\n"
            "OR\n"
            "• ONL_Outlet, Order_YearMonth, Total_OrderPrice"
        )

    df = df.rename(columns=column_map)

    # ---------------------------------------------
    # STANDARDIZE DATA TYPES
    # ---------------------------------------------
    df["Order_YearMonth"] = pd.to_datetime(
        df["Order_YearMonth"].astype(str).str[:7] + "-01",
        errors="coerce"
    )

    df["Total_OrderPrice"] = pd.to_numeric(
        df["Total_OrderPrice"], errors="coerce"
    ).fillna(0)

    df["ONL_Outlet"] = df["ONL_Outlet"].fillna("Unknown")

    return df
# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
if uploaded_file is None:
    st.info("Please upload an Excel file to proceed.")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================
try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"❌ Data Load Failed: {e}")
    st.stop()

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("Controls")
model_choice = st.sidebar.selectbox("Forecast Model", ["Prophet", "Random Forest", "XGBoost"])
selected_outlet = st.sidebar.selectbox("Outlet", ["All"] + sorted(df["ONL_Outlet"].unique()))
horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 12, 6)
conf_pct = st.sidebar.slider("Safety Buffer (%)", 10, 80, 55) / 100

# Training Years Filter
min_year = int(df["Order_YearMonth"].dt.year.min())
max_year = int(df["Order_YearMonth"].dt.year.max())
train_years = st.sidebar.slider(
    "Training Years",
    min_value=min_year,
    max_value=max_year,
    value=(max_year-3, max_year),
    step=1
)
st.sidebar.caption(f"📚 Model trained using data from **{train_years[0]} to {train_years[1]}**")

# =========================================================
# FILTER DATA
# =========================================================
df_f = df.copy()
if selected_outlet != "All":
    df_f = df_f[df_f["ONL_Outlet"] == selected_outlet]

df_f = df_f[
    (df_f["Order_YearMonth"].dt.year >= train_years[0]) &
    (df_f["Order_YearMonth"].dt.year <= train_years[1])
]

# =========================================================
# PREPARE MONTHLY SALES
# =========================================================
monthly_sales = df_f.groupby("Order_YearMonth")["Total_OrderPrice"].sum().sort_index().asfreq("MS", fill_value=0)
ml_df = monthly_sales.reset_index()
ml_df["month_idx"] = ml_df["Order_YearMonth"].dt.month
ml_df["year_idx"] = ml_df["Order_YearMonth"].dt.year
ml_df["time_step"] = np.arange(len(ml_df))

last_date = ml_df["Order_YearMonth"].max()
future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")

# =========================================================
# FORECAST
# =========================================================
if model_choice == "Prophet":
    p_df = ml_df.rename(columns={"Order_YearMonth": "ds", "Total_OrderPrice": "y"})
    model = Prophet(yearly_seasonality=True)
    model.fit(p_df)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    yhat = model.predict(future).tail(horizon)["yhat"].values
else:
    X = ml_df[["month_idx", "year_idx", "time_step"]]
    y = ml_df["Total_OrderPrice"]
    future_X = pd.DataFrame({
        "month_idx": future_dates.month,
        "year_idx": future_dates.year,
        "time_step": np.arange(len(X), len(X) + horizon)
    })
    model = RandomForestRegressor(n_estimators=300) if model_choice == "Random Forest" else XGBRegressor()
    model.fit(X, y)
    yhat = model.predict(future_X)

yhat = np.clip(yhat, 0, None)
res = pd.DataFrame({"ds": future_dates, "yhat": yhat})

# =========================================================
# YEAR-WISE SALES (LAST 4 + NEXT 1)
# =========================================================
hist_year = ml_df.assign(Year=ml_df["Order_YearMonth"].dt.year).groupby("Year")["Total_OrderPrice"].sum().reset_index()
fc_year = res.assign(Year=res["ds"].dt.year).groupby("Year")["yhat"].sum().reset_index()

st.subheader("Year-wise Sales (Last 4 Years + Next 1 Forecast)")
fig_y = go.Figure()
fig_y.add_trace(go.Bar(
    x=hist_year["Year"], y=hist_year["Total_OrderPrice"],
    name="Historical", marker_color="#1f2937",
    text=hist_year["Total_OrderPrice"].apply(format_currency), textposition='auto'
))
fig_y.add_trace(go.Bar(
    x=fc_year["Year"], y=fc_year["yhat"],
    name="Forecast", marker_color="#f59e0b",
    text=fc_year["yhat"].apply(format_currency), textposition='auto'
))
fig_y.update_layout(
    template="plotly_white",
    height=450,
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False
    ),
    xaxis=dict(
        title="",
        showgrid=False,
        zeroline=False
    )
)
st.plotly_chart(fig_y, use_container_width=True)

# =========================================================
# MONTHLY SALES TREND (LAST 12 + NEXT 6)
# =========================================================
st.subheader(f"📈 Monthly Sales Trend (Last 12 + Next {horizon})")

hist_12 = ml_df.tail(12).copy()
fc_6 = res.head(horizon).copy()
fc_6 = fc_6.rename(columns={"yhat": "Total_OrderPrice", "ds": "Order_YearMonth"})

plot_df = pd.concat([hist_12[["Order_YearMonth", "Total_OrderPrice"]], fc_6[["Order_YearMonth", "Total_OrderPrice"]]])
plot_df = plot_df.sort_values("Order_YearMonth")
plot_df["Type"] = ["Historical"] * len(hist_12) + ["Forecast"] * len(fc_6)

# KPI for last 12 + next 6
total_last_12 = hist_12["Total_OrderPrice"].sum()
total_next_horizon = fc_6["Total_OrderPrice"].sum()
avg_monthly_next_horizon = fc_6["Total_OrderPrice"].mean()

st.markdown(f"📌 **Key Performance Indicators ({selected_outlet})**")
k1, k2, k3 = st.columns(3)
k1.metric("Total Sales Last 12 Months", format_currency(total_last_12))
k2.metric(f"Forecast Next {horizon} Months", format_currency(total_next_horizon))
k3.metric("Avg Monthly Forecast", format_currency(avg_monthly_next_horizon))

# Bar chart
fig_m = go.Figure()
for t, c in [("Historical", "#1f2937"), ("Forecast", "#f59e0b")]:
    d = plot_df[plot_df["Type"] == t]
    fig_m.add_trace(go.Bar(
        x=d["Order_YearMonth"].dt.strftime("%b %Y"),
        y=d["Total_OrderPrice"],
        name=t,
        marker_color=c,
        text=[format_currency(x) for x in d["Total_OrderPrice"]],
        textposition="outside"
    ))

fig_m.update_layout(
    template="plotly_white",
    height=500,
    barmode="group",
    yaxis=dict(
        title="",
        showticklabels=False,
        showgrid=False,
        zeroline=False
    ),
    xaxis=dict(
        title="Month",
        showgrid=False,
        zeroline=False
    ),
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig_m, use_container_width=True)

# =========================================================
# FORECAST DATA TABLE
# =========================================================
st.subheader("📥 Forecast Data")

table_df = pd.DataFrame({
    "Month": res["ds"].dt.strftime("%b %Y"),
    "Forecast": res["yhat"].apply(format_currency)
})

st.dataframe(
    table_df,
    use_container_width=True,
    height=(len(table_df) + 1) * 35
)

st.download_button(
    "Download CSV",
    table_df.to_csv(index=False).encode("utf-8"),
    file_name="sales_forecast.csv",
    mime="text/csv"
)

# =========================================================
# AI SUMMARY / INSIGHT
# =========================================================
avg_monthly_actual = hist_12["Total_OrderPrice"].mean()
avg_monthly_forecast = fc_6["Total_OrderPrice"].mean()
growth_pct = ((avg_monthly_forecast - avg_monthly_actual) / avg_monthly_actual * 100
             if avg_monthly_actual > 0 else 0)
peak_month = fc_6.loc[fc_6["Total_OrderPrice"].idxmax(), "Order_YearMonth"]
peak_value = fc_6["Total_OrderPrice"].max()

trend_word = (
    "strong growth 📈" if growth_pct > 10 else
    "moderate growth ↗️" if growth_pct > 0 else
    "a declining trend 📉"
)

summary_text = f"""
🔍 **AI Sales Insight**

• The outlet shows **{trend_word}** in the upcoming months.  
• Average monthly sales expected to move from **{format_currency(avg_monthly_actual)}** to **{format_currency(avg_monthly_forecast)}**.  
• Peak forecast in **{peak_month.strftime('%b %Y')}** with revenue around **{format_currency(peak_value)}**.  
• Model trained on data from **{train_years[0]} to {train_years[1]}**.  
• Plan marketing and inventory around peak months for maximum efficiency.
"""
st.info(summary_text)
