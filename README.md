# Exotic-Options-Pricing & Derivatives Analytics Platform

An end-to-end derivatives pricing and risk analytics platform that converts market parameters into option values and risk sensitivities.

The platform supports both vanilla and exotic options and combines analytical and simulation-based pricing methods with interactive analytics and visualizations.

---

## Features

### Option Pricing

Supports pricing of:

- European Options
- American Options
- Asian Options
- Barrier Options

### Pricing Methodologies

- Black-Scholes Model
- Monte Carlo Simulation
- Binomial Tree Model

### Risk Analytics

Computation of key option Greeks:

- Delta (Δ)
- Gamma (Γ)
- Vega (ν)
- Theta (Θ)

### Model Validation

- Monte Carlo convergence analysis
- Binomial Tree convergence analysis
- Put-Call Parity validation

### Interactive Analytics Dashboard

Built using Streamlit and includes:

- Option Calculator
- Pricing Analytics
- Convergence Visualization
- Monte Carlo Path Simulations
- Barrier Path Visualization

---

## Project Structure

```text
derivatives_analytics/

├── European Options
├── American Options
├── Asian Options
├── Barrier Options
├── Greeks Computation
├── Validation Module
├── Visualization Module
└── Streamlit Dashboard
```

---

## User Inputs

The platform allows users to specify:

| Parameter | Description |
|------------|------------|
| S₀ | Current Asset Price |
| K | Strike Price |
| T | Time to Maturity |
| r | Risk-Free Rate |
| σ | Volatility |
| Option Type | European / American / Asian / Barrier |
| Option Style | Call / Put |
| Pricing Method | Black-Scholes / Monte Carlo / Binomial |

---

## Outputs

### Pricing Outputs

- Option Value
- Call / Put Price

### Risk Outputs

- Delta
- Gamma
- Vega
- Theta

### Analytics Outputs

- Monte Carlo Convergence Plots
- Binomial Convergence Plots
- Simulated Asset Paths
- Barrier Option Path Visualizations

---

## Methodology

### European Options

- Black-Scholes Closed Form Solution
- Monte Carlo Pricing
- Binomial Tree Pricing

### American Options

- Binomial Tree with Early Exercise Feature

### Asian Options

- Monte Carlo Simulation using Arithmetic Average Payoff

### Barrier Options

- Monte Carlo Simulation with Path Dependency and Barrier Conditions

---

## Key Concepts Implemented

- Geometric Brownian Motion (GBM)
- Risk-Neutral Valuation
- Monte Carlo Simulation
- Binomial Trees
- Option Greeks
- Path-Dependent Derivatives
- Numerical Convergence Analysis
- Put-Call Parity Validation

---

<img width="1917" height="941" alt="streamlit_1" src="https://github.com/user-attachments/assets/4894876d-1b48-4e4c-91aa-5c46d9dbed01" />
<img width="1907" height="922" alt="streamlit_2" src="https://github.com/user-attachments/assets/61333b0b-c547-46ac-833a-8bfb9a0732c1" />
  <img width="1917" height="912" alt="streamlit 3" src="https://github.com/user-attachments/assets/b480ce61-eacc-4840-ae89-e74bf5f89ace" />


## Libraries Used

- NumPy
- Pandas
- SciPy
- Matplotlib
- Streamlit

---

## Future Enhancements

- Implied Volatility Surface Construction
- Market Option Chain Integration
- Local Volatility Models
- Heston Model Implementation
- Calibration to Market Data
- Advanced Exotic Options
