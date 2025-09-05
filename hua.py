import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import statsmodels.api as sm
from scipy.stats import norm

# --- Page config ---
st.set_page_config(page_title="ARIMA–GARCH–VaR Analyzer", layout="wide")

# --- Custom CSS ---
page_bg = """
<style>
/* Background gradient lebih gelap */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #134e5e, #71b280);
    color: black;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(30,30,30,0.9);
    border-radius: 10px;
    padding: 15px;
    color: white;
}
/* Tombol navigasi di sidebar */
.sidebar-btn {
    display: block;
    width: 100%;
    padding: 10px;
    margin: 6px 0;
    font-size: 14px;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    text-align: center;
    cursor: pointer;
    text-decoration: none;
    color: white !important;
}
.sidebar-btn.upload { background-color: #1d3557; }
.sidebar-btn.arima { background-color: #457b9d; }
.sidebar-btn.vol { background-color: #a8dadc; }
.sidebar-btn.var { background-color: #e63946; }
/* Box untuk teks */
.custom-box {
    background-color: #ffffffcc;
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("📊 ARIMA – ARCH – GARCH – TGARCH with VaR")

# --- Fungsi cek stasioneritas ---
def check_stationarity(series, signif=0.05):
    adf_result = adfuller(series.dropna())
    return adf_result[1] < signif, adf_result[1]

# --- Session State untuk Navigation ---
if "menu" not in st.session_state:
    st.session_state.menu = "Upload Data"

# --- Sidebar Navigation dengan tombol ---
st.sidebar.markdown("### 📌 Navigation")
if st.sidebar.button("📂 Upload Data", key="btn1"):
    st.session_state.menu = "Upload Data"
if st.sidebar.button("🔮 ARIMA Modeling", key="btn2"):
    st.session_state.menu = "ARIMA Modeling"
if st.sidebar.button("📈 Volatility Modeling", key="btn3"):
    st.session_state.menu = "Volatility Modeling"
if st.sidebar.button("⚠️ Value at Risk (VaR)", key="btn4"):
    st.session_state.menu = "Value at Risk (VaR)"

menu = st.session_state.menu

# ================= PAGE 1: UPLOAD DATA =================
if menu == "Upload Data":
    uploaded_file = st.file_uploader("Upload your stock data (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Konversi kolom Date
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            data.set_index("Date", inplace=True)
            data.index = data.index.tz_localize(None)

        st.markdown('<div class="custom-box">📄 Data preview:</div>', unsafe_allow_html=True)
        st.dataframe(data.head())

        # --- Filter tanggal ---
        st.subheader("Filter Data by Date Range")
        min_date = data.index.min().date()
        max_date = data.index.max().date()

        default_start = max_date.replace(year=max_date.year - 2)

        start_date = st.date_input(
            "Start Date:",
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )

        end_date = st.date_input(
            "End Date:",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

        # Konversi input jadi datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Potong data sesuai tanggal
        mask = (data.index >= start_date) & (data.index <= end_date)
        data = data.loc[mask]

        st.write(f"📊 Data after filtering from {start_date.date()} to {end_date.date()}:")
        st.dataframe(data.head())
        st.write(f"Total rows: {len(data)}")

        # Pilih kolom harga
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        column = st.selectbox("Select closing price column", numeric_cols)
        series = data[column].astype(float)

        # Return log
        data['Return_log'] = np.log(series / series.shift(1))
        data = data.dropna()
        st.line_chart(data['Return_log'])

        # Stationarity test
        st.subheader("Stationarity Check (ADF Test)")
        stationary, pval = check_stationarity(data['Return_log'])
        diff_count = 0
        series_to_use = data['Return_log'].copy()
        while not stationary:
            diff_count += 1
            series_to_use = series_to_use.diff().dropna()
            stationary, pval = check_stationarity(series_to_use)
            st.write(f"Differencing ke-{diff_count} → p-value: {pval:.5f}")
        if diff_count == 0:
            st.success("✅ Data sudah stasioner")
        else:
            st.success(f"✅ Data stasioner setelah differencing {diff_count} kali")

        # Simpan state
        st.session_state.data = data
        st.session_state.series_to_use = series_to_use

# ================= PAGE 2: ARIMA MODELING =================
elif menu == "ARIMA Modeling":
    if "series_to_use" not in st.session_state:
        st.warning("⚠️ Please upload data first.")
    else:
        series_to_use = st.session_state.series_to_use
        st.subheader("ARIMA Grid Search (AIC Comparison)")
        if "run_arima" not in st.session_state:
            st.session_state.run_arima = False
        if st.button("Run ARIMA Grid Search"):
            st.session_state.run_arima = True
        if st.session_state.run_arima:
            aic_results = []
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(series_to_use, order=(p, d, q)).fit()
                            aic_results.append({"order": (p, d, q), "AIC": model.aic, "BIC": model.bic})
                        except:
                            continue
            results_df = pd.DataFrame(aic_results).sort_values("AIC")
            st.dataframe(results_df)
            best_order = results_df.iloc[0]["order"]
            st.success(f"✅ Best ARIMA order: {best_order}")
            best_model = ARIMA(series_to_use, order=best_order).fit()
            residuals = pd.Series(best_model.resid, index=series_to_use.index)
            st.session_state.residuals = residuals
            st.session_state.best_model = best_model
            st.line_chart(residuals)

# ================= PAGE 3: VOLATILITY MODELING =================
elif menu == "Volatility Modeling":
    if "residuals" not in st.session_state:
        st.warning("⚠️ Please run ARIMA first.")
    else:
        residuals = st.session_state.residuals
        results = []

        # --- ARCH Modeling ---
        st.subheader("ARCH Modeling")
        model_arch = arch_model(residuals, mean="Zero", vol="ARCH", p=1, dist='t')
        fit_arch = model_arch.fit(disp="off")
        st.line_chart(fit_arch.conditional_volatility, use_container_width=True)

        # ARCH Test
        arch_test = het_arch(residuals)
        st.write(f"ARCH Test p-value: {arch_test[1]:.5f}")
        if arch_test[1] < 0.05:
            st.success("✅ Heteroskedastisitas terdeteksi → lanjut GARCH")
        else:
            st.info("Data tidak menunjukkan ARCH effect yang signifikan.")

        # --- GARCH Modeling ---
        st.subheader("GARCH Modeling")
        model_garch = arch_model(residuals, mean="Zero", vol="GARCH", p=1, q=1, dist='t')
        fit_garch = model_garch.fit(disp="off")
        st.line_chart(fit_garch.conditional_volatility, use_container_width=True)

        # --- Sign Bias Test ---
        st.subheader("Sign Bias Test (for asymmetry)")
        std_resid = fit_garch.resid / fit_garch.conditional_volatility
        Y = std_resid**2
        lagged_resid = std_resid.shift(1).dropna()
        Y = Y.iloc[1:]
        D_neg = (lagged_resid < 0).astype(int)

        X = pd.DataFrame({"const": 1, "D_neg": D_neg})
        ols_model = sm.OLS(Y, X).fit(cov_type="HC1")
        st.write(ols_model.summary())

        # --- TGARCH jika perlu ---
        if ols_model.pvalues["D_neg"] < 0.05:
            st.warning("❗ Sign Bias effect terdeteksi → lanjut TGARCH")
            st.subheader("TGARCH Modeling")
            model_tgarch = arch_model(residuals, vol="GARCH", p=1, o=1, q=1, dist="t")
            fit_tgarch = model_tgarch.fit(disp="off")
            st.line_chart(fit_tgarch.conditional_volatility, use_container_width=True)
            models = {"ARCH(1)": fit_arch, "GARCH(1,1)": fit_garch, "TGARCH(1,1)": fit_tgarch}
        else:
            st.info("Tidak ada sign bias signifikan → TGARCH tidak diperlukan")
            models = {"ARCH(1)": fit_arch, "GARCH(1,1)": fit_garch}

        # --- Bandingkan model ---
        st.subheader("Model Comparison (AIC & LogLik)")
        for name, fit in models.items():
            results.append({"Model": name, "AIC": fit.aic, "LogLik": fit.loglikelihood})
        results_df = pd.DataFrame(results).sort_values(by=["AIC", "LogLik"], ascending=[True, False])
        st.dataframe(results_df)

        best_model_name = results_df.iloc[0]["Model"]
        st.success(f"📌 Best Model Selected: {best_model_name}")
        st.session_state.cond_vol = models[best_model_name].conditional_volatility

# ================= PAGE 4: VALUE AT RISK =================
elif menu == "Value at Risk (VaR)":
    if "cond_vol" not in st.session_state or "best_model" not in st.session_state:
        st.warning("⚠️ Run ARIMA & Volatility Modeling first.")
    else:
        cond_vol = st.session_state.cond_vol
        best_model = st.session_state.best_model
        st.subheader("Value at Risk (VaR)")
        W = st.number_input("Portfolio Value (Rp)", min_value=1_000_000, value=50_000_000)
        conf_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
        alpha = 1 - conf_level
        Z_alpha = norm.ppf(1 - alpha)
        horizon = st.number_input("Prediction Horizon (days)", min_value=1, max_value=30, value=5)
        forecast = best_model.get_forecast(steps=horizon)
        predicted_returns = forecast.predicted_mean.values
        predicted_vols = cond_vol.tail(horizon).values
        results = []
        for i in range(horizon):
            VaR_t = W * (predicted_returns[i] - Z_alpha * predicted_vols[i])
            results.append([f"H+{i+1}", predicted_returns[i], predicted_vols[i], VaR_t])
        results_df = pd.DataFrame(results, columns=["Day", "Predicted_Return", "Volatility", "VaR"])
        st.dataframe(results_df)
        st.line_chart(results_df.set_index("Day")[["VaR", "Volatility"]])
