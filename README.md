# Option Pricing & Greeks Visualizer (Streamlit)

This project provides a Streamlit-based interactive web application for pricing European call and put options using the Black-Scholes model. It also includes detailed analysis of the option Greeks and sensitivity plots.

---

## Features

* **Calculate European call and put option prices** using the Black-Scholes formula.
* **Compute key Greeks:**
    * Delta
    * Gamma
    * Theta
    * Vega
    * Rho
* **Analyze how option prices and Greeks change** with respect to:
    * Underlying asset price (S)
    * Strike price (K)
    * Volatility (σ)
    * Risk-free interest rate (r)
    * Dividend yield (q)
    * Time to maturity (T)
* **Visualize pricing and Greek sensitivities** using Matplotlib.

---

## Requirements

* Python 3.8+
* Streamlit
* NumPy
* SciPy
* Matplotlib

---

## Usage

Launch the app as described in the Installation section (`streamlit run app.py`). Once launched, the web application will open in your browser. You can then input the required option parameters (e.g., spot price, strike price, volatility, etc.) using the sliders and input fields.

The app will dynamically display:
* The calculated European Call and Put option prices.
* The computed values for all the Greeks (Delta, Gamma, Theta, Vega, Rho).
* Interactive plots showing how option prices and Greeks change in response to variations in input variables, allowing for detailed sensitivity analysis.

