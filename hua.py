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

# --- Page Config ---
st.set_page_config(page_title="ARIMA–GARCH–VaR Analyzer", layout="wide")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Background lebih gelap */
    .stApp {
        background-color: #1e1e1e;
        color: #f5f5f5;
    }

    /* Sidebar navigation tombol */
    div[data-baseweb="radio"] > div {
        background-color: #2c2c2c;
        border-radius: 8px;
        padding: 6px;
    }
    div[data-baseweb="radio"] label {
        color: white !important;
        font-size: 14px !important;
        font-weight: bold;
    }

    /* Tombol Run ARIMA Grid Search */
    div.stButton > button:first-child {
        background-color: #457b9d;
        color: white !important;
        font-weight: bold;
        border-radius: 6px;
        padding: 8px 16px;
    }
    div.stButton > button:hover {
        background-color: #1d3557;
        color: white !important;
    }

    /* Semua teks judul/subheader punya background berbeda */
    h1, h2, h3 {
        background-color: #2c2c2c;
        padding: 6px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 ARIMA – ARCH – GARCH – TGARCH with VaR")

# --- Fungsi cek stasioneritas ---
def check_stationarity(series, signif=0.05):
    adf_result = adfuller(series.dropna())
    return adf_result[1] < signif, adf_result[1]

# --- Sidebar Navigation ---
menu = st.sidebar.radio("Navigation", ["Upload Data", "ARIMA Modeling", "Volatility Modeling", "Value at Risk (VaR)"])

# ================= PAGE 1: UPLOAD DATA =================
if menu == "Upload Data":
    uploaded_file = st.file_uploader("Upload your stock data (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Konversi kolom Date
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            data.set_index("Date", inplace=True)

        st.write("📄 Data preview:")
        st.dataframe(data.head())

        # --- Filter tanggal ---
        st.subheader("Filter Data by Date Range")
        min_date = data.index.min().date()
        max_date = data.index.max().date()

        start_date = st.date_input("Select start date:", value=max_date.replace(year=max_date.year - 2),
                                   min_value=min_date, max_value=max_date)
        end_date = st.date_input("Select end date:", value=max_date,
                                 min_value=min_date, max_value=max_date)

        # Hilangkan timezone di index
        data.index = data.index.tz_localize(None)

        # Konversi input jadi datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Potong data sesuai tanggal
        mask = (data.index >= start_date) & (data.index <= end_date)
        data = data.loc[mask]

        st.write(f"📊 Data after filtering from {start_date.date()} to {end_date.date()}:")
        st.dataframe(data.head())
        st.write(f"Total rows: {len(data)}")

        # Pilih hanya kolom numerik (misalnya Close/Harga)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        column = st.selectbox("Select closing price column", numeric_cols)
        series = data[column].astype(float)

        # Hitung return log
        data['Return_log'] = np.log(series / series.shift(1))
        data = data.dropna()
        st.line_chart(data['Return_log'])

        # Cek stasioneritas
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

        # simpan ke session_state
        st.session_state.data = data
        st.session_state.series_to_use = series_to_use

# ================= PAGE 2: ARIMA MODELING =================
elif menu == "ARIMA Modeling":
    if "series_to_use" not in st.session_state:
        st.warning("⚠️ Please upload data first in 'Upload Data' page.")
    else:
        series_to_use = st.session_state.series_to_use

        st.subheader("ARIMA Grid Search (AIC Comparison)")
        if "run_arima" not in st.session_state:
            st.session_state.run_arima = False

        if st.button("Run ARIMA Grid Search"):
            st.session_state.run_arima = True

        if st.session_state.run_arima:
            aic_results = []
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(series_to_use, order=(p, d, q)).fit()
                            aic_results.append({
                                "order": (p, d, q),
                                "AIC": model.aic,
                                "BIC": model.bic
                            })
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

            st.subheader("Residuals of Best ARIMA Model")
            st.line_chart(residuals)

# ================= PAGE 3: VOLATILITY MODELING =================
elif menu == "Volatility Modeling":
    if "residuals" not in st.session_state:
        st.warning("⚠️ Please run ARIMA first in 'ARIMA Modeling' page.")
    else:
        residuals = st.session_state.residuals

        st.subheader("ARCH Modeling")
        model_arch = arch_model(residuals, mean="Zero", vol="ARCH", p=1)
        fit_arch = model_arch.fit(disp="off")
        st.line_chart(fit_arch.conditional_volatility, use_container_width=True)

        st.subheader("GARCH Modeling")
        model_garch = arch_model(residuals, mean="Zero", vol="GARCH", p=1, q=1)
        fit_garch = model_garch.fit(disp="off")
        st.line_chart(fit_garch.conditional_volatility, use_container_width=True)

        # --- Uji Sign Bias ---
        st.subheader("Sign Bias Test (for asymmetry)")
        std_resid = fit_garch.resid / fit_garch.conditional_volatility
        Y = std_resid**2
        lagged_resid = std_resid.shift(1).dropna()
        Y = Y.iloc[1:]
        D_neg = (lagged_resid < 0).astype(int)

        X = pd.DataFrame({"const": 1, "D_neg": D_neg})
        ols_model = sm.OLS(Y, X).fit(cov_type="HC1")
        if ols_model.pvalues["D_neg"] < 0.05:
            st.warning("❗ Sign Bias effect terdeteksi → lanjut TGARCH")
        else:
            st.info("Tidak ada sign bias signifikan → TGARCH opsional")

        st.subheader("TGARCH Modeling")
        model_tgarch = arch_model(residuals, vol="GARCH", p=1, o=1, q=1, dist="t")
        fit_tgarch = model_tgarch.fit(disp="off")
        st.line_chart(fit_tgarch.conditional_volatility, use_container_width=True)

        # --- Bandingkan model ---
        st.subheader("Model Comparison (AIC & LogLik)")
        models = {"ARCH(1)": fit_arch, "GARCH(1,1)": fit_garch, "TGARCH(1,1)": fit_tgarch}
        results = [{"Model": name, "AIC": fit.aic, "LogLik": fit.loglikelihood} for name, fit in models.items()]
        results_df = pd.DataFrame(results).sort_values(by=["AIC", "LogLik"], ascending=[True, False])
        st.dataframe(results_df)

        best_model_name = results_df.iloc[0]["Model"]
        st.success(f"📌 Best Model Selected: {best_model_name}")

        if best_model_name == "ARCH(1)":
            cond_vol = fit_arch.conditional_volatility
        elif best_model_name == "GARCH(1,1)":
            cond_vol = fit_garch.conditional_volatility
        else:
            cond_vol = fit_tgarch.conditional_volatility

        st.session_state.cond_vol = cond_vol

# ================= PAGE 4: VALUE AT RISK =================
elif menu == "Value at Risk (VaR)":
    if "cond_vol" not in st.session_state or "best_model" not in st.session_state:
        st.warning("⚠️ Please run ARIMA and Volatility Modeling first.")
    else:
        cond_vol = st.session_state.cond_vol
        series_to_use = st.session_state.series_to_use
        best_model = st.session_state.best_model

        st.subheader("Value at Risk (VaR) from Best Model")

        preset_values = [10_000_000, 50_000_000, 100_000_000, 500_000_000]
        selected_preset = st.selectbox("Choose preset portfolio value:", preset_values)
        custom_value = st.number_input("Or enter custom portfolio value (Rp):", min_value=1_000_000, value=selected_preset)
        W = custom_value

        conf_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
        alpha = 1 - conf_level
        Z_alpha = norm.ppf(1 - alpha)

        horizon = st.number_input("Prediction Horizon (days)", min_value=1, max_value=30, value=5)

        forecast = best_model.get_forecast(steps=horizon)
        predicted_returns = forecast.predicted_mean.values
        predicted_vols = cond_vol.tail(horizon).values

        var_values = []
        for i in range(horizon):
            R_hat = predicted_returns[i]
            sigma = predicted_vols[i]
            VaR_t = W * (R_hat - Z_alpha * sigma)
            var_values.append(VaR_t)

        results_df = pd.DataFrame({
            "Day": [f"H+{i+1}" for i in range(horizon)],
            "Predicted_Return": predicted_returns,
            "Volatility": predicted_vols,
            "VaR": var_values
        })

        st.write(f"📊 Estimated {horizon}-Day VaR ({int(conf_level*100)}% confidence):")
        st.dataframe(results_df)
        st.line_chart(results_df.set_index("Day")[["VaR", "Volatility"]])
