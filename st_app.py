# ============================================================
# ECO 5012B - Topic 2: Trade Tensions and Financial Markets
# Streamlit Interactive Application
# Replication of Ferrari Minesso, Kurcz & Pagliari (2022)
#
# Run with: streamlit run st_app.py
# Requires: data.csv in the same directory
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Trade Tensions & Financial Markets",
    page_icon="p",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA AND ESTIMATE MODEL
# Cached so it only runs once per session
# ============================================================

@st.cache_data
def load_and_model():
    """
    Load data.csv and estimate the baseline OLS model.
    Returns the raw dataframe, regression dataframe, and
    a dictionary of model parameters.
    """
    df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

    # Create lagged variables if not already present
    if "lag_return" not in df.columns:
        df["lag_return"] = df["sp500_log_return"].shift(1)
    if "lag_vix" not in df.columns:
        df["lag_vix"] = df["vix"].shift(1)

    # Ensure TPU is standardised
    if "tpu_standardised" not in df.columns:
        df["tpu_standardised"] = (
            (df["tpu"] - df["tpu"].mean()) / df["tpu"].std()
        )

    # Prepare regression dataset
    reg = df[["sp500_log_return", "lag_return",
              "tpu_standardised", "lag_vix"]].dropna()

    X = sm.add_constant(
        reg[["tpu_standardised", "lag_return", "lag_vix"]]
    )
    y = reg["sp500_log_return"]

    # OLS with Newey-West HAC standard errors (10 lags)
    fit = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10}
    )
    reg["fitted"] = fit.fittedvalues

    params = {
        "alpha":      float(fit.params["const"]),
        "gamma":      float(fit.params["tpu_standardised"]),
        "delta":      float(fit.params["lag_return"]),
        "lam":        float(fit.params["lag_vix"]),
        "r2":         float(fit.rsquared),
        "gamma_pval": float(fit.pvalues["tpu_standardised"]),
        "n_obs":      int(fit.nobs),
    }
    return df, reg, params


df, reg_df, params = load_and_model()

# Compute summary statistics for the slider
tpu_mean  = float(df["tpu"].mean())
tpu_std   = float(df["tpu"].std())
last_ret  = float(df["sp500_log_return"].dropna().iloc[-1])
last_vix  = float(df["vix"].dropna().iloc[-1])
last_date = df["sp500_log_return"].dropna().index[-1]

# ============================================================
# HEADER
# ============================================================

st.title("Trade Tensions and Financial Markets")
st.markdown("""
**Replication of Ferrari Minesso, Kurcz & Pagliari (2022)**
*Do Words Hurt More Than Actions? — Journal of Applied Econometrics, 37(6), 1138–1159*
""")
st.markdown("---")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Model Controls")
st.sidebar.markdown("---")

# Step 3.2: Interactive slider for TPU level
st.sidebar.subheader("1. Set Trade Policy Uncertainty")
tpu_slider = st.sidebar.slider(
    label="TPU Index Level",
    min_value=float(df["tpu"].min()),
    max_value=float(df["tpu"].max()),
    value=float(tpu_mean),
    step=1.0,
    help="Move the slider to simulate different trade uncertainty levels. "
         "Higher values represent more elevated trade tensions."
)

# Convert slider value to standardised units for the model
tpu_std_input = (tpu_slider - tpu_mean) / tpu_std

st.sidebar.markdown("---")

# Step 3.3: Radio button for sentiment state (asymmetric feature)
st.sidebar.subheader("2. Select Sentiment State")
st.sidebar.markdown(
    "Reflects the paper\'s *words hurt more than actions* finding: "
    "trade rhetoric amplifies market uncertainty beyond what the "
    "TPU index alone measures."
)

sentiment = st.sidebar.radio(
    label="Policy Sentiment / Rhetoric State",
    options=[
        "Normal sentiment",
        "Low sentiment (Trade Easing)",
        "High sentiment (Trade Tightening)"
    ],
    index=0
)

# Step 3.3: Adjust gamma by 50% based on sentiment state
# This operationalises the asymmetric effects in the paper:
# Tightening rhetoric amplifies market impact (x1.5)
# Easing rhetoric dampens market impact (x0.5)
base_gamma = params["gamma"]

if "Easing" in sentiment:
    adj_gamma   = base_gamma * 0.50
    state_col   = "green"
    state_label = "Low - Easing"
    multiplier  = 0.5
    state_note  = "gamma reduced to 50% - positive rhetoric dampens uncertainty"
elif "Tightening" in sentiment:
    adj_gamma   = base_gamma * 1.50
    state_col   = "red"
    state_label = "High - Tightening"
    multiplier  = 1.5
    state_note  = "gamma amplified to 150% - escalatory rhetoric heightens risk"
else:
    adj_gamma   = base_gamma
    state_col   = "steelblue"
    state_label = "Normal"
    multiplier  = 1.0
    state_note  = "gamma at baseline - standard market conditions"

# ============================================================
# ONE-PERIOD-AHEAD FORECAST
# r_{t+1} = alpha + gamma_adj*TPU_input + delta*r_last + lambda*VIX_last
# ============================================================

forecast = (
    params["alpha"]
    + adj_gamma      * tpu_std_input
    + params["delta"] * last_ret
    + params["lam"]   * last_vix
)
forecast_pct = forecast * 100

# Baseline forecast (without sentiment adjustment) for delta
baseline_fcst = (
    params["alpha"]
    + base_gamma      * tpu_std_input
    + params["delta"] * last_ret
    + params["lam"]   * last_vix
) * 100
delta_fcst = forecast_pct - baseline_fcst

# ============================================================
# METRICS ROW (Step 3.2 - st.metric required by brief)
# ============================================================

st.subheader("One-Period-Ahead Forecast")

c1, c2, c3, c4 = st.columns(4)

with c1:
    # Step 3.2: st.metric displaying one-period-ahead forecast
    st.metric(
        label="S&P 500 Forecast Return (t+1)",
        value=f"{forecast_pct:.4f}%",
        delta=f"{delta_fcst:+.4f}% vs baseline",
        delta_color="inverse"
    )

with c2:
    st.metric(
        label="Current TPU Level",
        value=f"{tpu_slider:.0f}",
        delta=f"{tpu_slider - tpu_mean:+.0f} vs historical mean"
    )

with c3:
    st.metric(
        label="Uncertainty Coefficient (gamma)",
        value=f"{adj_gamma:.6f}",
        delta=f"x{multiplier:.1f} adjustment"
    )

with c4:
    st.metric(
        label="Sentiment State",
        value=state_label
    )

st.caption(
    f"Baseline gamma = {base_gamma:.6f} | "
    f"p-value = {params['gamma_pval']:.4f} | "
    f"R2 = {params['r2']:.4f} | "
    f"n = {params['n_obs']:,} | "
    f"HAC standard errors (Newey-West, 10 lags) | "
    f"{state_note}"
)

st.markdown("---")

# ============================================================
# MAIN CHART (Step 3.4 - updates automatically with controls)
# ============================================================

st.subheader("S&P 500 Returns: Historical Series + Live Forecast")

# Prepare smoothed data for the chart
plot_df = reg_df.copy()
plot_df["actual_ma"] = plot_df["sp500_log_return"].rolling(
    10, min_periods=3).mean()
plot_df["fitted_ma"] = plot_df["fitted"].rolling(
    10, min_periods=3).mean()
plot_df = plot_df.dropna(subset=["actual_ma", "fitted_ma"])

fig = go.Figure()

# Actual returns
fig.add_trace(go.Scatter(
    x=plot_df.index,
    y=plot_df["actual_ma"] * 100,
    mode="lines",
    name="Actual Return (10d MA)",
    line=dict(color="#1f77b4", width=1.6),
    opacity=0.88
))

# OLS fitted values
fig.add_trace(go.Scatter(
    x=plot_df.index,
    y=plot_df["fitted_ma"] * 100,
    mode="lines",
    name="OLS Fitted (10d MA)",
    line=dict(color="#d62728", width=1.6, dash="dash"),
    opacity=0.88
))

# Live forecast point (updates with slider and radio button)
next_date = pd.Timestamp(last_date) + pd.Timedelta(days=1)
fig.add_trace(go.Scatter(
    x=[next_date],
    y=[forecast_pct],
    mode="markers+text",
    name=f"Live Forecast ({state_label})",
    marker=dict(
        color=state_col, size=15, symbol="star",
        line=dict(color="black", width=1)
    ),
    text=[f"{forecast_pct:.3f}%"],
    textposition="top center",
    textfont=dict(size=10, color=state_col)
))

# Trade war event annotations using add_shape (avoids Plotly bug)
events = {
    "2018-03-22": "Steel & Al tariffs",
    "2018-07-06": "$34bn tariffs",
    "2019-05-10": "Escalation 25%",
    "2019-08-23": "Aug tweet",
    "2020-01-15": "Phase One deal",
}

for date_str, label in events.items():
    ts = pd.Timestamp(date_str)
    if plot_df.index[0] <= ts <= plot_df.index[-1]:
        fig.add_shape(
            type="line",
            x0=date_str, x1=date_str,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="orange", width=1.2, dash="dot"),
            opacity=0.5
        )

fig.update_layout(
    title=(
        f"S&P 500 Returns | Forecast: {forecast_pct:.4f}% | "
        f"State: {state_label} | TPU = {tpu_slider:.0f}"
    ),
    xaxis_title="Date",
    yaxis_title="Log Return (x100, %)",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right",  x=1
    ),
    height=480,
    template="plotly_white",
    hovermode="x unified"
)

# Step 3.4: Chart updates automatically with slider and radio button
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================
# COEFFICIENT TABLE AND INTERPRETATION
# ============================================================

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Model Coefficients")
    coeff_df = pd.DataFrame({
        "Parameter": [
            "alpha (intercept)",
            "gamma (TPU impact)",
            "delta (lagged return)",
            "lambda (lagged VIX)"
        ],
        "Baseline Estimate": [
            f"{params['alpha']:.6f}",
            f"{base_gamma:.6f}",
            f"{params['delta']:.6f}",
            f"{params['lam']:.6f}"
        ],
        "State-Adjusted gamma": [
            "-",
            f"{adj_gamma:.6f}",
            "-",
            "-"
        ],
        "Economic meaning": [
            "Baseline daily drift in returns",
            "1 s.d. TPU rise -> change in return",
            "Mean reversion control (AR term)",
            "Pre-existing risk aversion control"
        ]
    })
    st.dataframe(coeff_df, use_container_width=True)

with col_r:
    st.subheader("Words Hurt More: The Asymmetry")
    st.markdown(f"""
Ferrari Minesso et al. (2022) find that **trade rhetoric alone**
drives significant movements in financial markets before any
tariff is formally enacted. This is the *words hurt more*
mechanism.

The radio button above implements this asymmetry:

| State | Multiplier | Adjusted gamma |
|-------|:---:|:---:|
| Normal | x1.0 | `{base_gamma:.6f}` |
| Low (Easing) | x0.5 | `{base_gamma*0.5:.6f}` |
| High (Tightening) | x1.5 | `{base_gamma*1.5:.6f}` |

The paper finds the 3T-Index explains up to **23% of Chinese
stock market variance**, far exceeding the contribution of
actual tariff implementations.

The aggregate S&P 500 is largely insulated (gamma = {base_gamma:.4f},
p = {params["gamma_pval"]:.3f}), consistent with Section 3.1 of
the paper. Significant effects are concentrated in Chinese
equities and EME currencies.
    """)

st.markdown("---")

# ============================================================
# RESEARCH EXTENSION (Step 3.1)
# ============================================================

st.subheader("Research Extension Proposal")
st.markdown("""
**Research Gap:** Ferrari Minesso et al. (2022) demonstrate that
US aggregate equities are largely insulated from trade tensions,
while Chinese equities and EME currencies bear the brunt. The
asymmetric effects across asset classes have not been explored
interactively.

**Extension Question:** Does the *words hurt more* asymmetry
persist when the same local projection framework is applied to
exchange rates and sovereign bond spreads, where the paper
predicts larger and more significant effects?

**Economic Motivation:** The paper's FEVD shows the 3T-Index
explains 4-20% of EME financial market variance compared to
near-zero for US aggregate equities. Extending this application
to the USD/CNY exchange rate and EMBI+ bond spreads would
allow users to observe this asymmetry directly, operationalising
the paper\'s core finding across multiple financial markets.

**How the Streamlit app illustrates this:** Users can already
observe that under High Tightening sentiment, the forecast
return becomes more negative (gamma amplified by 1.5x). A
natural extension would add an asset class selector to the
sidebar, allowing users to switch between S&P 500, USD/CNY,
and EMBI+ and observe how the sensitivity differs across markets.
""")

st.markdown("---")
st.caption(
    "ECO 5012B | Replication: Ferrari Minesso, Kurcz & Pagliari (2022) | "
    "Data: Yahoo Finance (^GSPC), Caldara et al. (2019) policyuncertainty.com, "
    "FRED (VIXCLS, DGS10). "
    "AI disclosure: code structure developed with AI assistance; "
    "all economic interpretation is the author\'s own work."
)
