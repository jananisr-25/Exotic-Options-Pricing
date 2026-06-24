"""
Exotic Options Pricing Engine
Run: streamlit run app.py
"""

# ── stdlib / third-party ───────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# ALL MATH — copied verbatim from notebook cells
# ══════════════════════════════════════════════════════════════════════════════

# ── GBM ───────────────────────────────────────────────────────────────────────
def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=42):
    rng  = np.random.default_rng(seed)
    dt   = T / n_steps
    half = n_paths // 2
    z    = rng.standard_normal((half, n_steps))
    z    = np.vstack([z, -z])
    inc  = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z
    lp   = np.concatenate([np.zeros((n_paths,1)), np.cumsum(inc, axis=1)], axis=1)
    return S0 * np.exp(lp)

def simulate_gbm_terminal(S0, mu, sigma, T, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    z   = rng.standard_normal(n_paths)
    return S0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)

# ── Black-Scholes ─────────────────────────────────────────────────────────────
def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return d1, d1 - sigma*np.sqrt(T)

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(S-K,0) if option_type=="call" else max(K-S,0)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# ── Greeks ────────────────────────────────────────────────────────────────────
def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 1e-6:
        return dict(delta=0, gamma=0, vega=0, theta=0, rho=0)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    nd1 = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S*nd1*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
        rho   =  K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S*nd1*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    gamma = nd1 / (S*sigma*np.sqrt(T))
    vega  = S*nd1*np.sqrt(T) / 100
    return dict(delta=float(delta), gamma=float(gamma),
                vega=float(vega), theta=float(theta), rho=float(rho))

def fd_greeks(pricer_fn, S, K, T, r, sigma, option_type, **kwargs):
    dS = S*0.01; dsig = 0.001; dr = 0.0001; dt = 1/365
    p0   = pricer_fn(S,    K, T,    r,    sigma,      option_type, **kwargs)["price"]
    pup  = pricer_fn(S+dS, K, T,    r,    sigma,      option_type, **kwargs)["price"]
    pdn  = pricer_fn(S-dS, K, T,    r,    sigma,      option_type, **kwargs)["price"]
    psig = pricer_fn(S,    K, T,    r,    sigma+dsig, option_type, **kwargs)["price"]
    pt   = pricer_fn(S,    K, T-dt, r,    sigma,      option_type, **kwargs)["price"] if T>dt else p0
    pr   = pricer_fn(S,    K, T,    r+dr, sigma,      option_type, **kwargs)["price"]
    return dict(
        delta = float((pup-pdn)/(2*dS)),
        gamma = float((pup-2*p0+pdn)/(dS**2)),
        vega  = float((psig-p0)/dsig/100),
        theta = float((pt-p0)/dt/365),
        rho   = float((pr-p0)/dr/100),
    )

# ── European ──────────────────────────────────────────────────────────────────
def price_european(S, K, T, r, sigma, option_type="call",
                   method="black_scholes", n_paths=50_000, n_steps=200):
    if method == "black_scholes":
        return {"price": black_scholes(S,K,T,r,sigma,option_type), "method":"Black-Scholes"}
    elif method == "binomial":
        dt = T/n_steps; u = np.exp(sigma*np.sqrt(dt)); d = 1/u
        p  = (np.exp(r*dt)-d)/(u-d); disc = np.exp(-r*dt)
        j  = np.arange(n_steps+1); ST = S*u**(n_steps-2*j)
        V  = np.maximum(ST-K,0) if option_type=="call" else np.maximum(K-ST,0)
        for _ in range(n_steps):
            V = disc*(p*V[:-1]+(1-p)*V[1:])
        return {"price": float(V[0]), "method":"Binomial (CRR)"}
    else:  # monte_carlo
        ST  = simulate_gbm_terminal(S, r, sigma, T, n_paths)
        pay = np.maximum(ST-K,0) if option_type=="call" else np.maximum(K-ST,0)
        dp  = np.exp(-r*T)*pay
        p   = dp.mean(); se = dp.std()/np.sqrt(n_paths)
        return {"price":float(p),"se":float(se),
                "ci_low":float(p-1.96*se),"ci_high":float(p+1.96*se),"method":"Monte Carlo"}

# ── American ──────────────────────────────────────────────────────────────────
def price_american(S, K, T, r, sigma, option_type="put", n_steps=300):
    dt = T/n_steps; u = np.exp(sigma*np.sqrt(dt)); d = 1/u
    p  = (np.exp(r*dt)-d)/(u-d); disc = np.exp(-r*dt)
    j  = np.arange(n_steps+1); ST = S*(u**(n_steps-j))*(d**j)
    V  = np.maximum(ST-K,0) if option_type=="call" else np.maximum(K-ST,0)
    boundary = []
    for i in range(n_steps-1,-1,-1):
        j_i = np.arange(i+1); S_i = S*(u**(i-j_i))*(d**j_i)
        hold = disc*(p*V[:i+1]+(1-p)*V[1:i+2])
        exer = np.maximum(S_i-K,0) if option_type=="call" else np.maximum(K-S_i,0)
        V    = np.maximum(hold, exer)
        mask = exer > hold
        boundary.append((i*dt, S_i[mask].max() if mask.any() else np.nan))
    boundary.reverse()
    eu = black_scholes(S, K, T, r, sigma, option_type)
    return {"price":float(V[0]), "method":"Binomial American",
            "boundary":boundary, "european":eu, "premium":float(V[0])-eu}

# ── Asian ─────────────────────────────────────────────────────────────────────
def price_asian(S, K, T, r, sigma, option_type="call",
                average_type="arithmetic", n_paths=50_000, n_steps=252):
    paths  = simulate_gbm_paths(S, r, sigma, T, n_steps, n_paths)
    prices = paths[:,1:]
    avg    = prices.mean(axis=1) if average_type=="arithmetic" else np.exp(np.log(prices).mean(axis=1))
    pay    = np.maximum(avg-K,0) if option_type=="call" else np.maximum(K-avg,0)
    dp     = np.exp(-r*T)*pay
    p = dp.mean(); se = dp.std()/np.sqrt(n_paths)
    return {"price":float(p),"se":float(se),
            "ci_low":float(p-1.96*se),"ci_high":float(p+1.96*se),
            "method":f"Monte Carlo Asian ({average_type})"}

# ── Barrier ───────────────────────────────────────────────────────────────────
def price_barrier(S, K, T, r, sigma, barrier, barrier_type="down-and-out",
                  option_type="call", n_paths=50_000, n_steps=252, rebate=0.0):
    paths = simulate_gbm_paths(S, r, sigma, T, n_steps, n_paths)
    dir_, knock = barrier_type.split("-and-")
    crossed = paths.max(axis=1)>=barrier if dir_=="up" else paths.min(axis=1)<=barrier
    ST  = paths[:,-1]
    base = np.maximum(ST-K,0) if option_type=="call" else np.maximum(K-ST,0)
    pay  = np.where(crossed, rebate, base) if knock=="out" else np.where(crossed, base, rebate)
    dp   = np.exp(-r*T)*pay
    p = dp.mean(); se = dp.std()/np.sqrt(n_paths)
    return {"price":float(p),"se":float(se),
            "ci_low":float(p-1.96*se),"ci_high":float(p+1.96*se),
            "method":f"MC Barrier ({barrier_type})"}

# ── Convergence helpers ───────────────────────────────────────────────────────
def mc_convergence(S, K, T, r, sigma, option_type, path_sizes):
    out = []
    for n in path_sizes:
        ST  = simulate_gbm_terminal(S, r, sigma, T, n)
        pay = np.maximum(ST-K,0) if option_type=="call" else np.maximum(K-ST,0)
        out.append({"paths": n, "price": float(np.exp(-r*T)*pay.mean())})
    return pd.DataFrame(out)

def binomial_convergence(S, K, T, r, sigma, option_type, node_sizes):
    out = []
    for n in node_sizes:
        res = price_european(S, K, T, r, sigma, option_type, "binomial", n_steps=n)
        out.append({"nodes": n, "price": res["price"]})
    return pd.DataFrame(out)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Options Pricing Engine",
                   page_icon="📊", layout="wide")

# Minimal clean styling
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .result-box {
        background: #ffffff; border: 1px solid #dee2e6; border-radius: 8px;
        padding: 24px 28px; margin-bottom: 16px;
    }
    .price-label { color: #6c757d; font-size: 13px; margin-bottom: 4px; }
    .price-value { color: #212529; font-size: 42px; font-weight: 700; line-height: 1.1; }
    .greek-box {
        background: #ffffff; border: 1px solid #dee2e6; border-radius: 8px;
        padding: 16px 12px; text-align: center;
    }
    .greek-label { color: #6c757d; font-size: 11px; text-transform: uppercase;
                   letter-spacing: 0.8px; margin-bottom: 6px; }
    .greek-value { color: #212529; font-size: 20px; font-weight: 600; }
    .info-row { color: #6c757d; font-size: 12px; margin-top: 6px; }
    div.stButton > button {
        background-color: #2c3e50; color: white; border: none;
        border-radius: 6px; padding: 10px 0; font-size: 14px;
        font-weight: 600; width: 100%; letter-spacing: 0.5px;
    }
    div.stButton > button:hover { background-color: #34495e; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
    .section-title {
        font-size: 13px; font-weight: 600; color: #495057;
        text-transform: uppercase; letter-spacing: 0.8px;
        margin: 20px 0 10px; border-bottom: 2px solid #e9ecef; padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — all inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Options Pricing Engine")
    st.caption("exotic_options_pricing_notebook.ipynb")
    st.divider()

    st.markdown("**Market Parameters**")
    spot   = st.number_input("Spot Price (S₀)",     value=100.0, min_value=0.01, step=1.0,  format="%.2f")
    strike = st.number_input("Strike Price (K)",     value=100.0, min_value=0.01, step=1.0,  format="%.2f")
    T      = st.number_input("Maturity (years)",     value=1.0,   min_value=0.01, step=0.25, format="%.2f")
    r_pct  = st.number_input("Risk-Free Rate (%)",   value=5.0,   min_value=0.0,  max_value=30.0, step=0.25, format="%.2f")
    v_pct  = st.number_input("Volatility σ (%)",     value=20.0,  min_value=0.1,  max_value=200.0,step=1.0,  format="%.1f")
    r      = r_pct / 100.0
    sigma  = v_pct  / 100.0

    st.divider()
    st.markdown("**Option Settings**")
    option_cat  = st.selectbox("Option Type", ["European", "American", "Asian", "Barrier"])
    option_type = st.radio("Call / Put", ["Call", "Put"], horizontal=True).lower()

    # Method
    if option_cat == "European":
        method = st.selectbox("Pricing Method", ["Black-Scholes", "Monte Carlo", "Binomial"])
        method_key = {"Black-Scholes":"black_scholes","Monte Carlo":"monte_carlo","Binomial":"binomial"}[method]
    elif option_cat == "American":
        st.caption("Method: Binomial Tree (early exercise)")
        method = "Binomial"
    elif option_cat == "Asian":
        st.caption("Method: Monte Carlo")
        avg_type = st.radio("Average Type", ["Arithmetic", "Geometric"], horizontal=True).lower()
    elif option_cat == "Barrier":
        st.caption("Method: Monte Carlo")
        barrier_val  = st.number_input("Barrier Level (H)", value=85.0, step=1.0, format="%.2f")
        barrier_type = st.selectbox("Barrier Type",
                          ["down-and-out","down-and-in","up-and-out","up-and-in"])
        rebate = st.number_input("Rebate", value=0.0, step=0.5, format="%.2f")

    # MC settings
    needs_mc = option_cat in ("Asian","Barrier") or \
               (option_cat=="European" and method=="Monte Carlo")
    if needs_mc:
        st.divider()
        st.markdown("**Simulation**")
        n_paths = st.select_slider("Paths", [10_000,30_000,50_000,100_000], value=50_000)
        n_steps = st.select_slider("Steps / Path", [63,126,252,504], value=252)
    else:
        n_paths, n_steps = 50_000, 252

    st.divider()
    run = st.button("Price Option", use_container_width=True)


# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("Exotic Options Pricing Engine")
st.caption(f"Pricing {option_cat} {option_type.upper()} · S={spot} · K={strike} · T={T}y · r={r_pct}% · σ={v_pct}%")

if not run:
    st.info("Configure parameters in the sidebar and click **Price Option** to compute.")
    st.stop()


# ─────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────
with st.spinner("Computing…"):
    if option_cat == "European":
        result = price_european(spot, strike, T, r, sigma, option_type, method_key, n_paths, n_steps)
    elif option_cat == "American":
        result = price_american(spot, strike, T, r, sigma, option_type, n_steps=300)
    elif option_cat == "Asian":
        result = price_asian(spot, strike, T, r, sigma, option_type, avg_type, n_paths, n_steps)
    elif option_cat == "Barrier":
        result = price_barrier(spot, strike, T, r, sigma, barrier_val,
                               barrier_type, option_type, n_paths, n_steps, rebate)

    price_val = result["price"]

    # Greeks — analytic for European/American, FD for exotics
    if option_cat in ("European","American"):
        greeks = bs_greeks(spot, strike, T, r, sigma, option_type)
        g_note = "Analytic (Black-Scholes)"
    elif option_cat == "Asian":
        greeks = fd_greeks(price_asian, spot, strike, T, r, sigma, option_type,
                           average_type=avg_type, n_paths=10_000, n_steps=63)
        g_note = "Finite Difference"
    elif option_cat == "Barrier":
        greeks = fd_greeks(price_barrier, spot, strike, T, r, sigma, option_type,
                           barrier=barrier_val, barrier_type=barrier_type,
                           n_paths=10_000, n_steps=63, rebate=rebate)
        g_note = "Finite Difference"

    # Paths for analytics tab
    viz_paths = simulate_gbm_paths(spot, r, sigma, T, n_steps, 80, seed=7)
    t_days    = np.linspace(0, T*252, viz_paths.shape[1])


# ─────────────────────────────────────────────
# TWO TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["Option Calculator", "Analytics"])


# ══════════════════════════════════════════════
# TAB 1 — OPTION CALCULATOR
# ══════════════════════════════════════════════
with tab1:
    col_price, col_greeks = st.columns([1, 2], gap="large")

    # Price box
    with col_price:
        itm = (option_type=="call" and spot>strike) or (option_type=="put" and spot<strike)
        ci_line = ""
        if "se" in result:
            ci_line = f"<div class='info-row'>95% CI &nbsp;[{result['ci_low']:.4f}, {result['ci_high']:.4f}] &nbsp;·&nbsp; SE ±{result['se']:.4f}</div>"

        st.markdown(f"""
        <div class="result-box">
            <div class="price-label">
                {option_cat} {option_type.upper()} &nbsp;·&nbsp; {result['method']}
                &nbsp;·&nbsp; {'ITM' if itm else 'OTM'}
            </div>
            <div class="price-value">₹{price_val:.4f}</div>
            {ci_line}
        </div>
        """, unsafe_allow_html=True)

        # Intrinsic / Time value
        intr = max(spot-strike,0) if option_type=="call" else max(strike-spot,0)
        tv   = max(price_val - intr, 0)
        c1, c2 = st.columns(2)
        c1.metric("Intrinsic Value", f"₹{intr:.4f}")
        c2.metric("Time Value",      f"₹{tv:.4f}")

        if option_cat == "American":
            st.metric("Early Exercise Premium", f"₹{result['premium']:.4f}",
                      help="American price minus equivalent European price")

    # Greeks
    with col_greeks:
        st.markdown(f'<div class="section-title">Greeks &nbsp;<span style="font-weight:400;text-transform:none;font-size:11px;color:#adb5bd;">({g_note})</span></div>', unsafe_allow_html=True)

        g1, g2, g3, g4, g5 = st.columns(5)
        for col, sym, name, val in [
            (g1, "Δ", "Delta",  greeks["delta"]),
            (g2, "Γ", "Gamma",  greeks["gamma"]),
            (g3, "Θ", "Theta",  greeks["theta"]),
            (g4, "ν", "Vega",   greeks["vega"]),
            (g5, "ρ", "Rho",    greeks["rho"]),
        ]:
            col.markdown(f"""
            <div class="greek-box">
                <div class="greek-label">{sym} {name}</div>
                <div class="greek-value">{val:+.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")  # spacer

        # Greeks vs Spot chart
        S_range  = np.linspace(max(spot*0.5,1), spot*1.5, 200)
        gdf      = pd.DataFrame([bs_greeks(s, strike, T, r, sigma, option_type) for s in S_range])
        gdf["S"] = S_range

        gcol1, gcol2 = st.columns(2)
        for col, key, label, color in [
            (gcol1, "delta", "Delta (Δ)", "#2c3e50"),
            (gcol2, "gamma", "Gamma (Γ)", "#e74c3c"),
        ]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gdf["S"], y=gdf[key], mode="lines",
                                     line=dict(color=color, width=2), showlegend=False))
            fig.add_vline(x=spot, line_dash="dash", line_color="#888", line_width=1)
            fig.update_layout(height=200, margin=dict(l=0,r=0,t=24,b=0),
                              title=dict(text=label, font=dict(size=12)),
                              xaxis=dict(title="Spot", showgrid=True, gridcolor="#f0f0f0"),
                              yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                              plot_bgcolor="white", paper_bgcolor="white")
            col.plotly_chart(fig, use_container_width=True)

        gcol3, gcol4 = st.columns(2)
        for col, key, label, color in [
            (gcol3, "vega",  "Vega (ν)",  "#8e44ad"),
            (gcol4, "theta", "Theta (Θ)", "#e67e22"),
        ]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gdf["S"], y=gdf[key], mode="lines",
                                     line=dict(color=color, width=2), showlegend=False))
            fig.add_vline(x=spot, line_dash="dash", line_color="#888", line_width=1)
            fig.update_layout(height=200, margin=dict(l=0,r=0,t=24,b=0),
                              title=dict(text=label, font=dict(size=12)),
                              xaxis=dict(title="Spot", showgrid=True, gridcolor="#f0f0f0"),
                              yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                              plot_bgcolor="white", paper_bgcolor="white")
            col.plotly_chart(fig, use_container_width=True)

    # Payoff diagram
    st.markdown('<div class="section-title">Payoff Profile</div>', unsafe_allow_html=True)
    S_exp  = np.linspace(max(spot*0.5,1), spot*1.5, 300)
    intr_c = np.maximum(S_exp-strike,0) if option_type=="call" else np.maximum(strike-S_exp,0)
    bs_now = np.array([black_scholes(s,strike,T,  r,sigma,option_type) for s in S_exp])
    bs_hlf = np.array([black_scholes(s,strike,T/2,r,sigma,option_type) for s in S_exp])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_exp, y=intr_c, name="At expiry",
                             line=dict(color="#e74c3c", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=S_exp, y=bs_now, name=f"Now (T={T:.2f}y)",
                             line=dict(color="#2c3e50", width=2)))
    fig.add_trace(go.Scatter(x=S_exp, y=bs_hlf, name=f"Mid (T={T/2:.2f}y)",
                             line=dict(color="#8e44ad", width=1.5, dash="dot")))
    fig.add_vline(x=spot,   line_dash="dash", line_color="#3498db", line_width=1,
                  annotation_text=f"S={spot}", annotation_font_color="#3498db")
    fig.add_vline(x=strike, line_dash="dash", line_color="#e74c3c", line_width=1,
                  annotation_text=f"K={strike}", annotation_font_color="#e74c3c")
    fig.update_layout(height=340, margin=dict(l=0,r=0,t=8,b=0),
                      xaxis=dict(title="Spot at expiry", showgrid=True, gridcolor="#f0f0f0"),
                      yaxis=dict(title="Option value",   showgrid=True, gridcolor="#f0f0f0"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                  font=dict(size=12)),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # Download
    report = f"""# Options Pricing Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Parameters
Spot={spot}, Strike={strike}, T={T}y, r={r_pct}%, σ={v_pct}%
Option: {option_cat} {option_type.upper()} | Method: {result['method']}

## Price
₹{price_val:.6f}
{"SE: ±" + f"{result['se']:.4f}  |  95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]" if 'se' in result else ""}

## Greeks ({g_note})
Delta = {greeks['delta']:+.6f}
Gamma = {greeks['gamma']:.6f}
Vega  = {greeks['vega']:.6f}
Theta = {greeks['theta']:+.6f}
Rho   = {greeks['rho']:+.6f}
"""
    st.download_button("⬇ Download Report", data=report,
                       file_name=f"options_{option_cat.lower()}_{option_type}.md",
                       mime="text/markdown")


# ══════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════
with tab2:

    # ── Monte Carlo convergence ──────────────────────────────────────────
    st.markdown('<div class="section-title">Monte Carlo Convergence</div>', unsafe_allow_html=True)
    with st.spinner("Running MC convergence…"):
        path_sizes = [500, 1_000, 5_000, 10_000, 30_000, 50_000, 100_000]
        mc_conv    = mc_convergence(spot, strike, T, r, sigma, option_type, path_sizes)
    bs_ref = black_scholes(spot, strike, T, r, sigma, option_type)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_conv["paths"], y=mc_conv["price"],
                             mode="lines+markers", name="MC Price",
                             line=dict(color="#2c3e50", width=2),
                             marker=dict(size=6, color="#2c3e50")))
    fig.add_hline(y=bs_ref, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                  annotation_text=f"Black-Scholes ₹{bs_ref:.4f}",
                  annotation_font_color="#e74c3c", annotation_position="right")
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=8,b=0),
                      xaxis=dict(title="Number of Paths (log)", type="log",
                                 showgrid=True, gridcolor="#f0f0f0"),
                      yaxis=dict(title="Option Price", showgrid=True, gridcolor="#f0f0f0"),
                      legend=dict(font=dict(size=12)),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # ── Binomial convergence ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Binomial Tree Convergence</div>', unsafe_allow_html=True)
    with st.spinner("Running Binomial convergence…"):
        node_sizes = [10, 25, 50, 100, 200, 500, 1000]
        bi_conv    = binomial_convergence(spot, strike, T, r, sigma, option_type, node_sizes)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bi_conv["nodes"], y=bi_conv["price"],
                              mode="lines+markers", name="Binomial Price",
                              line=dict(color="#8e44ad", width=2),
                              marker=dict(size=6, color="#8e44ad")))
    fig2.add_hline(y=bs_ref, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                   annotation_text=f"Black-Scholes ₹{bs_ref:.4f}",
                   annotation_font_color="#e74c3c", annotation_position="right")
    fig2.update_layout(height=300, margin=dict(l=0,r=0,t=8,b=0),
                       xaxis=dict(title="Number of Tree Nodes",
                                  showgrid=True, gridcolor="#f0f0f0"),
                       yaxis=dict(title="Option Price", showgrid=True, gridcolor="#f0f0f0"),
                       legend=dict(font=dict(size=12)),
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

    # ── GBM / MC path visualisation ──────────────────────────────────────
    st.markdown('<div class="section-title">GBM / Monte Carlo Path Simulation</div>',
                unsafe_allow_html=True)
    n_show = st.slider("Paths to display", 10, 100, 40, step=10)
    show_paths = simulate_gbm_paths(spot, r, sigma, T, n_steps, n_show, seed=99)
    t_ax = np.linspace(0, T*252, show_paths.shape[1])

    fig3 = go.Figure()
    for i in range(show_paths.shape[0]):
        fig3.add_trace(go.Scatter(x=t_ax, y=show_paths[i], mode="lines",
                                   showlegend=False,
                                   line=dict(width=0.8, color="rgba(44,62,80,0.18)")))
    fig3.add_trace(go.Scatter(x=t_ax, y=show_paths.mean(axis=0),
                               mode="lines", name="Mean path",
                               line=dict(color="#e74c3c", width=2)))
    fig3.add_hline(y=strike, line_dash="dash", line_color="#3498db", line_width=1,
                   annotation_text=f" K = {strike}", annotation_font_color="#3498db",
                   annotation_position="right")
    fig3.update_layout(height=340, margin=dict(l=0,r=0,t=8,b=0),
                        xaxis=dict(title="Trading Days", showgrid=True, gridcolor="#f0f0f0"),
                        yaxis=dict(title="Stock Price",  showgrid=True, gridcolor="#f0f0f0"),
                        legend=dict(font=dict(size=12)),
                        plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

    # Terminal distribution
    ST_dist = simulate_gbm_terminal(spot, r, sigma, T, 30_000)
    fig4 = go.Figure(go.Histogram(x=ST_dist, nbinsx=60,
                                   marker=dict(color="#2c3e50", opacity=0.6,
                                               line=dict(color="white", width=0.3)),
                                   showlegend=False))
    fig4.add_vline(x=strike, line_dash="dash", line_color="#e74c3c", line_width=1.5,
                   annotation_text=f" K={strike}", annotation_font_color="#e74c3c")
    fig4.add_vline(x=spot, line_dash="dash", line_color="#3498db", line_width=1.5,
                   annotation_text=f" S₀={spot}", annotation_font_color="#3498db")
    fig4.update_layout(height=260, margin=dict(l=0,r=0,t=8,b=0),
                        title=dict(text="Terminal Price Distribution S_T", font=dict(size=13)),
                        xaxis=dict(title="S_T",  showgrid=True, gridcolor="#f0f0f0"),
                        yaxis=dict(title="Count", showgrid=True, gridcolor="#f0f0f0"),
                        plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

    # ── Barrier path visualisation ────────────────────────────────────────
    if option_cat == "Barrier":
        st.markdown('<div class="section-title">Barrier Path Visualisation</div>',
                    unsafe_allow_html=True)
        bar_paths = simulate_gbm_paths(spot, r, sigma, T, n_steps, 60, seed=55)
        dir_, knock = barrier_type.split("-and-")
        if dir_ == "up":
            knocked = bar_paths.max(axis=1) >= barrier_val
        else:
            knocked = bar_paths.min(axis=1) <= barrier_val
        t_ax2 = np.linspace(0, T*252, bar_paths.shape[1])

        fig5 = go.Figure()
        for i in range(bar_paths.shape[0]):
            color = "rgba(231,76,60,0.25)" if knocked[i] else "rgba(44,62,80,0.18)"
            fig5.add_trace(go.Scatter(x=t_ax2, y=bar_paths[i], mode="lines",
                                       showlegend=False, line=dict(width=0.9, color=color)))
        # legend proxies
        fig5.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Knocked out",
                                   line=dict(color="#e74c3c", width=2)))
        fig5.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Survived",
                                   line=dict(color="#2c3e50", width=2)))
        fig5.add_hline(y=barrier_val, line_dash="dash", line_color="#f39c12", line_width=2,
                       annotation_text=f" H={barrier_val} ({barrier_type})",
                       annotation_font_color="#f39c12", annotation_position="right")
        fig5.add_hline(y=strike, line_dash="dot", line_color="#3498db", line_width=1,
                       annotation_text=f" K={strike}",
                       annotation_font_color="#3498db", annotation_position="right")
        pct_knocked = knocked.mean()*100
        fig5.update_layout(height=360, margin=dict(l=0,r=0,t=8,b=0),
                            title=dict(text=f"Barrier Paths · {pct_knocked:.1f}% knocked out",
                                       font=dict(size=13)),
                            xaxis=dict(title="Trading Days", showgrid=True, gridcolor="#f0f0f0"),
                            yaxis=dict(title="Stock Price",  showgrid=True, gridcolor="#f0f0f0"),
                            legend=dict(font=dict(size=12)),
                            plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig5, use_container_width=True)

    # ── American boundary ─────────────────────────────────────────────────
    if option_cat == "American":
        st.markdown('<div class="section-title">Early Exercise Boundary</div>',
                    unsafe_allow_html=True)
        bnd = result.get("boundary", [])
        b_t = [b[0]*252 for b in bnd if not np.isnan(b[1])]
        b_s = [b[1]     for b in bnd if not np.isnan(b[1])]
        if b_t:
            fig6 = go.Figure(go.Scatter(x=b_t, y=b_s, mode="lines",
                                         name="Exercise boundary",
                                         line=dict(color="#e67e22", width=2)))
            fig6.add_hline(y=strike, line_dash="dash", line_color="#3498db", line_width=1,
                           annotation_text=f" K={strike}")
            fig6.update_layout(height=300, margin=dict(l=0,r=0,t=8,b=0),
                                xaxis=dict(title="Days", showgrid=True, gridcolor="#f0f0f0"),
                                yaxis=dict(title="Critical Spot", showgrid=True, gridcolor="#f0f0f0"),
                                plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig6, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Method Comparison</div>', unsafe_allow_html=True)
    with st.spinner("Running comparison…"):
        rows = []
        for ot in ["call","put"]:
            bs_p  = black_scholes(spot,strike,T,r,sigma,ot)
            bi_p  = price_european(spot,strike,T,r,sigma,ot,"binomial",n_steps=300)["price"]
            mc_p  = price_european(spot,strike,T,r,sigma,ot,"monte_carlo",n_paths=20_000)["price"]
            am_p  = price_american(spot,strike,T,r,sigma,ot,n_steps=300)["price"]
            as_p  = price_asian(spot,strike,T,r,sigma,ot,"arithmetic",20_000,63)["price"]
            rows.append({
                "":               ot.upper(),
                "BS European":    f"₹{bs_p:.4f}",
                "Binomial Euro":  f"₹{bi_p:.4f}",
                "MC European":    f"₹{mc_p:.4f}",
                "American":       f"₹{am_p:.4f}",
                "Asian (arith)":  f"₹{as_p:.4f}",
                "AM–EU Premium":  f"₹{am_p-bs_p:.4f}",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)