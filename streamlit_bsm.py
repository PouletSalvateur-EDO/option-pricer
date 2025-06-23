import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# --- Black-Scholes Core Functions ---

def get_intermediate(params):
    S, K, r, sigma, T, q = params['S'], params['K'], params['r'], params['sigma'], params['T'], params['q']
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_call(params, d1, d2):
    S, K, r, T, q = params['S'], params['K'], params['r'], params['T'], params['q']
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(params, d1, d2):
    S, K, r, T, q = params['S'], params['K'], params['r'], params['T'], params['q']
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

def get_greeks(params):
    S, K, r, sigma, T, q, n = params['S'], params['K'], params['r'], params['sigma'], params['T'], params['q'], params['n']
    d1, d2 = get_intermediate(params)

    # Delta : dérivé du prix de l'option par le prix du sous-jacent (dP / dS)
    d_c = np.exp(-q * T) * norm.cdf(d1)
    d_p = np.exp(-q * T) * (norm.cdf(d1) - 1)

    # Gamma : dérivé seconde du prix de l'option par le prix du sous-jacent (dP2 / dS2)
    g = (np.exp(-q * T) /(S * sigma * np.sqrt(T))) * norm.pdf(d1)

    # Theta : dérivé du prix de l'option par le temps à maturité (dP / dt),
    # changement du prix de l'option par jour (theta negatif => option perd chaque jour)
    first_theta_term = -(S * sigma * np.exp(-q * T) * norm.pdf(d1))/(2 * np.sqrt(T))
    sec_c = r * K * np.exp(-r * T) * norm.cdf(d2)
    third_c = q * S * np.exp(-q * T) * norm.cdf(d1)
    t_c = (1/n) * (first_theta_term - sec_c + third_c)

    sec_p = r * K * np.exp(-r * T) * norm.cdf(-d2)
    third_p = q * S * np.exp(-q * T) * norm.cdf(-d1)
    t_p = (1/n) * (first_theta_term + sec_p - third_p)

    # Vega : dérivé du prix de l'option par la volatilité (dP / dsigma)
    # changement du prix par pourcentage de volatilité
    v = (1/100) * S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)

    # Rho : dérivé du prix de l'option par le taux d'intérêt (dP / dr)
    # changement du prix par pourcentage de taux d'intérêt
    r_c = (1/100)* K * T * np.exp(-r * T) * norm.cdf(d2)
    r_p = -(1/100)* K * T * np.exp(-r * T) * norm.cdf(-d2)

    # Return as a single dictionary, with tuples for call/put specific Greeks
    return {
        'Delta' : (d_c, d_p),
        'Gamma' : g, # Gamma is the same for call and put by put-call parity
        'Theta' : (t_c, t_p),
        'Vega' : v, # Vega is the same for call and put
        'Rho' : (r_c, r_p)
    }

# --- Plotting Functions ---

def plot_sensitivity_to_dividend(params):
    original_q = params['q']
    q_range = np.linspace(0.0, 0.2, 100) 

    to_plot = []
    for q_val in q_range:
        temp_params = params.copy()
        temp_params['q'] = q_val
        d1, d2 = get_intermediate(temp_params)
        call_price = black_scholes_call(temp_params, d1, d2)
        to_plot.append(call_price)

    params['q'] = original_q # Restore original q in params

    fig, ax = plt.subplots(figsize=(10, 6)) # Create a new figure for this plot
    ax.plot(q_range * 100, to_plot, label="Call Price vs Dividend Yield", color="red")
    ax.set_xlabel("Rendement du dividende (q en %)")
    ax.set_ylabel("Prix du Call (€)")
    ax.set_title("Sensibilité du prix du Call au rendement du dividende (q)")
    ax.grid(True)
    ax.legend()
    return fig # Return the figure object

def sensitivity_subplot_call(fig, params):
    fig.clear() 
    axs = fig.subplots(3, 2) 

    original_params = params.copy() 
    S, K, r, sigma, T, q = params['S'], params['K'], params['r'], params['sigma'], params['T'], params['q']
    param_text = f"S={S}, K={K}, r={r:.2%}, σ={sigma:.2%}, T={T}, q={q:.2%}"
    fig.suptitle(f"Sensibilité du prix du Call\nParamètres: {param_text}", fontsize=16)

    # 1. Sensitivity to S (Delta)
    S_range = np.linspace(S * 0.7, S * 1.3, 100)
    call_prices_S = []
    for s_val in S_range:
        temp_params = original_params.copy()
        temp_params['S'] = s_val
        d1, d2 = get_intermediate(temp_params)
        call_prices_S.append(black_scholes_call(temp_params, d1, d2))
    axs[0, 0].plot(S_range, call_prices_S, color='blue')
    axs[0, 0].set_title("Effet du prix du sous-jacent (S)")
    axs[0, 0].set_xlabel("Prix du sous-jacent (S)")
    axs[0, 0].set_ylabel("Prix du Call")
    axs[0, 0].grid(True)

    # 2. Sensitivity to r (Rho)
    r_range = np.linspace(0.0001, 0.2, 100)
    call_prices_r = []
    for r_val in r_range:
        temp_params = original_params.copy()
        temp_params['r'] = r_val
        d1, d2 = get_intermediate(temp_params)
        call_prices_r.append(black_scholes_call(temp_params, d1, d2))
    axs[0, 1].plot(r_range, call_prices_r, color='green')
    axs[0, 1].set_title("Effet du taux sans risque (r)")
    axs[0, 1].set_xlabel("Taux sans risque (r)")
    axs[0, 1].set_ylabel("Prix du Call")
    axs[0, 1].grid(True)

    # 3. Sensitivity to T (Theta)
    T_range = np.linspace(0.01, params['T'] * 2 if params['T'] * 2 > 0.01 else 2.0, 100)
    call_prices_T = []
    for t_val in T_range:
        temp_params = original_params.copy()
        temp_params['T'] = t_val
        d1, d2 = get_intermediate(temp_params)
        call_prices_T.append(black_scholes_call(temp_params, d1, d2))
    axs[1, 0].plot(T_range, call_prices_T, color='red')
    axs[1, 0].set_title("Effet du temps à maturité (T)")
    axs[1, 0].set_xlabel("Temps à maturité (T)")
    axs[1, 0].set_ylabel("Prix du Call")
    axs[1, 0].grid(True)

    # 4. Sensitivity to Sigma (Vega)
    sigma_range = np.linspace(0.01, 1, 100)
    call_prices_sigma = []
    for sigma_val in sigma_range:
        temp_params = original_params.copy()
        temp_params['sigma'] = sigma_val
        d1, d2 = get_intermediate(temp_params)
        call_prices_sigma.append(black_scholes_call(temp_params, d1, d2))
    axs[1, 1].plot(sigma_range, call_prices_sigma, color='purple')
    axs[1, 1].set_title("Effet de la volatilité (σ)")
    axs[1, 1].set_xlabel("Volatilité (σ)")
    axs[1, 1].set_ylabel("Prix du Call")
    axs[1, 1].grid(True)

    # 5. Sensitivity to q (Dividend Yield)
    q_range = np.linspace(0.0, 0.2, 100)
    call_prices_q = []
    for q_val in q_range:
        temp_params = original_params.copy()
        temp_params['q'] = q_val
        d1, d2 = get_intermediate(temp_params)
        call_prices_q.append(black_scholes_call(temp_params, d1, d2))
    axs[2, 0].plot(q_range, call_prices_q, color='orange')
    axs[2, 0].set_title("Effet du rendement du dividende (q)")
    axs[2, 0].set_xlabel("Rendement du dividende (q)")
    axs[2, 0].set_ylabel("Prix du Call")
    axs[2, 0].grid(True)

    axs[2, 1].axis('off')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def sensitivity_subplot_put(fig, params):
    fig.clear() 
    axs = fig.subplots(3, 2) 

    original_params = params.copy() 

    S, K, r, sigma, T, q = params['S'], params['K'], params['r'], params['sigma'], params['T'], params['q']
    param_text = f"S={S}, K={K}, r={r:.2%}, σ={sigma:.2%}, T={T}, q={q:.2%}"
    fig.suptitle(f"Sensibilité du prix du Put\nParamètres: {param_text}", fontsize=16)

    # 1. Sensitivity to S (Delta)
    S_range = np.linspace(S * 0.7, S * 1.3, 100)
    put_prices_S = []
    for s_val in S_range:
        temp_params = original_params.copy()
        temp_params['S'] = s_val
        d1, d2 = get_intermediate(temp_params)
        put_prices_S.append(black_scholes_put(temp_params, d1, d2))
    axs[0, 0].plot(S_range, put_prices_S, color='blue')
    axs[0, 0].set_title("Effet du prix du sous-jacent (S)")
    axs[0, 0].set_xlabel("Prix du sous-jacent (S)")
    axs[0, 0].set_ylabel("Prix du Put")
    axs[0, 0].grid(True)

    # 2. Sensitivity to r (Rho)
    r_range = np.linspace(0.0001, 0.2, 100)
    put_prices_r = []
    for r_val in r_range:
        temp_params = original_params.copy()
        temp_params['r'] = r_val
        d1, d2 = get_intermediate(temp_params)
        put_prices_r.append(black_scholes_put(temp_params, d1, d2))
    axs[0, 1].plot(r_range, put_prices_r, color='green')
    axs[0, 1].set_title("Effet du taux sans risque (r)")
    axs[0, 1].set_xlabel("Taux sans risque (r)")
    axs[0, 1].set_ylabel("Prix du Put")
    axs[0, 1].grid(True)

    # 3. Sensitivity to T (Theta)
    T_range = np.linspace(0.01, params['T'] * 2 if params['T'] * 2 > 0.01 else 2.0, 100)
    put_prices_T = []
    for t_val in T_range:
        temp_params = original_params.copy()
        temp_params['T'] = t_val
        d1, d2 = get_intermediate(temp_params)
        put_prices_T.append(black_scholes_put(temp_params, d1, d2))
    axs[1, 0].plot(T_range, put_prices_T, color='red')
    axs[1, 0].set_title("Effet du temps à maturité (T)")
    axs[1, 0].set_xlabel("Temps à maturité (T)")
    axs[1, 0].set_ylabel("Prix du Put")
    axs[1, 0].grid(True)

    # 4. Sensitivity to Sigma (Vega)
    sigma_range = np.linspace(0.01, 1, 100)
    put_prices_sigma = []
    for sigma_val in sigma_range:
        temp_params = original_params.copy()
        temp_params['sigma'] = sigma_val
        d1, d2 = get_intermediate(temp_params)
        put_prices_sigma.append(black_scholes_put(temp_params, d1, d2))
    axs[1, 1].plot(sigma_range, put_prices_sigma, color='purple')
    axs[1, 1].set_title("Effet de la volatilité (σ)")
    axs[1, 1].set_xlabel("Volatilité (σ)")
    axs[1, 1].set_ylabel("Prix du Put")
    axs[1, 1].grid(True)

    # 5. Sensitivity to q (Dividend Yield)
    q_range = np.linspace(0.0, 0.2, 100)
    put_prices_q = []
    for q_val in q_range:
        temp_params = original_params.copy()
        temp_params['q'] = q_val
        d1, d2 = get_intermediate(temp_params)
        put_prices_q.append(black_scholes_put(temp_params, d1, d2))
    axs[2, 0].plot(q_range, put_prices_q, color='orange')
    axs[2, 0].set_title("Effet du rendement du dividende (q)")
    axs[2, 0].set_xlabel("Rendement du dividende (q)")
    axs[2, 0].set_ylabel("Prix du Put")
    axs[2, 0].grid(True)

    axs[2, 1].axis('off')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def sensitivity_all_greeks_C(fig, params):
    fig.clear() 
    axs = fig.subplots(3, 2) 

    original_params = params.copy() 
    S, K, r, sigma, T, q = params['S'], params['K'], params['r'], params['sigma'], params['T'], params['q']

    
    s_range = np.linspace(original_params['S'] * 0.7, original_params['S'] * 1.3, 100)

    delta_values, gamma_values, vega_values, theta_values, rho_values = [], [], [], [], []

    for current_S in s_range:
        temp_params = original_params.copy() 
        temp_params['S'] = current_S
        
        greeks_for_s = get_greeks(temp_params) 
        
        delta_values.append(greeks_for_s['Delta'][0]) # Call Delta
        gamma_values.append(greeks_for_s['Gamma'])
        vega_values.append(greeks_for_s['Vega'])
        theta_values.append(greeks_for_s['Theta'][0]) # Call Theta
        rho_values.append(greeks_for_s['Rho'][0]) # Call Rho

    param_text = f"S={original_params['S']}, K={K}, r={r:.2%}, σ={sigma:.2%}, T={T}, q={q:.2%}"
    fig.suptitle(f"Sensibilité des Greeks (Call) au prix du sous-jacent (S)\nParamètres: {param_text}", fontsize=16)

    axs[0, 0].plot(s_range, delta_values, color='blue')
    axs[0, 0].set_title("Delta")
    axs[0, 0].set_xlabel("Prix du sous-jacent (S)")
    axs[0, 0].set_ylabel("Delta")
    axs[0, 0].grid(True)

    axs[0, 1].plot(s_range, gamma_values, color='orange')
    axs[0, 1].set_title("Gamma")
    axs[0, 1].set_xlabel("Prix du sous-jacent (S)")
    axs[0, 1].set_ylabel("Gamma")
    axs[0, 1].grid(True)

    axs[1, 0].plot(s_range, vega_values, color='green')
    axs[1, 0].set_title("Vega (pour 1%)")
    axs[1, 0].set_xlabel("Prix du sous-jacent (S)")
    axs[1, 0].set_ylabel("Vega")
    axs[1, 0].grid(True)

    axs[1, 1].plot(s_range, theta_values, color='red')
    axs[1, 1].set_title("Theta (par jour)")
    axs[1, 1].set_xlabel("Prix du sous-jacent (S)")
    axs[1, 1].set_ylabel("Theta")
    axs[1, 1].grid(True)

    axs[2, 0].plot(s_range, rho_values, color='purple')
    axs[2, 0].set_title("Rho (pour 1%)")
    axs[2, 0].set_xlabel("Prix du sous-jacent (S)")
    axs[2, 0].set_ylabel("Rho")
    axs[2, 0].grid(True)

    axs[2, 1].axis('off')  
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def sensitivity_all_greeks_P(fig, params):
    fig.clear() 
    axs = fig.subplots(3, 2) 

    original_params = params.copy() 
    S, K, r, sigma, T, q = params['S'], params['K'], params['r'], params['sigma'], params['T'], params['q']

    
    s_range = np.linspace(original_params['S'] * 0.7, original_params['S'] * 1.3, 100)

    delta_values, gamma_values, vega_values, theta_values, rho_values = [], [], [], [], []

    for current_S in s_range:
        temp_params = original_params.copy() 
        temp_params['S'] = current_S
        
        greeks_for_s = get_greeks(temp_params) 
        
        delta_values.append(greeks_for_s['Delta'][1]) # Put Delta
        gamma_values.append(greeks_for_s['Gamma'])
        vega_values.append(greeks_for_s['Vega'])
        theta_values.append(greeks_for_s['Theta'][1]) # Put Theta
        rho_values.append(greeks_for_s['Rho'][1]) # Put Rho

    param_text = f"S={original_params['S']}, K={K}, r={r:.2%}, σ={sigma:.2%}, T={T}, q={q:.2%}"
    fig.suptitle(f"Sensibilité des Greeks (Put) au prix du sous-jacent (S)\nParamètres: {param_text}", fontsize=16)

    axs[0, 0].plot(s_range, delta_values, color='blue')
    axs[0, 0].set_title("Delta")
    axs[0, 0].set_xlabel("Prix du sous-jacent (S)")
    axs[0, 0].set_ylabel("Delta")
    axs[0, 0].grid(True)

    axs[0, 1].plot(s_range, gamma_values, color='orange')
    axs[0, 1].set_title("Gamma")
    axs[0, 1].set_xlabel("Prix du sous-jacent (S)")
    axs[0, 1].set_ylabel("Gamma")
    axs[0, 1].grid(True)

    axs[1, 0].plot(s_range, vega_values, color='green')
    axs[1, 0].set_title("Vega (pour 1%)")
    axs[1, 0].set_xlabel("Prix du sous-jacent (S)")
    axs[1, 0].set_ylabel("Vega")
    axs[1, 0].grid(True)

    axs[1, 1].plot(s_range, theta_values, color='red')
    axs[1, 1].set_title("Theta (par jour)")
    axs[1, 1].set_xlabel("Prix du sous-jacent (S)")
    axs[1, 1].set_ylabel("Theta")
    axs[1, 1].grid(True)

    axs[2, 0].plot(s_range, rho_values, color='purple')
    axs[2, 0].set_title("Rho (pour 1%)")
    axs[2, 0].set_xlabel("Prix du sous-jacent (S)")
    axs[2, 0].set_ylabel("Rho")
    axs[2, 0].grid(True)

    axs[2, 1].axis('off')  
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

# --- STREAMLIT INTERFACE ---

st.set_page_config(page_title="Black-Scholes + Greeks", layout="centered")

st.title("Black-Scholes Pricing with Greeks & Dividend Sensitivity")

# Harmonized Input Parameters in Sidebar
with st.sidebar:
    st.header("Paramètres du modèle")
    S = st.number_input("Prix de l’actif sous-jacent (S)", min_value=0.01, value=100.0)
    K = st.number_input("Prix d’exercice (K)", min_value=0.01, value=100.0)
    r = st.slider("Taux sans risque (r)", min_value=0.0, max_value=0.2, value=0.05, step=0.001, format="%.3f")
    sigma = st.slider("Volatilité (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f")
    T = st.slider("Temps jusqu’à échéance (T en années)", min_value=0.01, max_value=2.0, value=1.0, step=0.01, format="%.2f")
    q = st.slider("Rendement du dividende (q)", min_value=0.0, max_value=0.2, value=0.02, step=0.001, format="%.3f")
    n = st.number_input("Nombre de jours dans l’année (pour Theta)", min_value=1, value=365)
    st.header("Paramètres du Graphique")
    l = st.number_input("Longueur de la  figuere", min_value=1, value=10)
    w = st.number_input("largeur de la  figuere", min_value=1, value=10)
params = {
    'S': S, 'K': K, 'r': r, 'sigma': sigma, 'T': T, 'q': q, 'n': n
}

d1, d2 = get_intermediate(params)

call_price = black_scholes_call(params, d1, d2)
put_price = black_scholes_put(params, d1, d2)

st.subheader("Prix des options")
col1, col2 = st.columns(2)
col1.metric("Call", f"{call_price:.2f} €")
col2.metric("Put", f"{put_price:.2f} €")

greeks_values = get_greeks(params) 

st.subheader("Sensibilités (Greeks)")

greek_cols = st.columns(3)
# Accessing tuple elements for Delta, Theta, Rho
greek_cols[0].markdown(f"**Delta (Call)**: {greeks_values['Delta'][0]:.4f}")
greek_cols[0].markdown(f"**Delta (Put)**: {greeks_values['Delta'][1]:.4f}")
greek_cols[1].markdown(f"**Gamma**: {greeks_values['Gamma']:.4f}")
greek_cols[1].markdown(f"**Vega**: {greeks_values['Vega']:.4f}")
greek_cols[2].markdown(f"**Theta (Call)**: {greeks_values['Theta'][0]:.4f}")
greek_cols[2].markdown(f"**Theta (Put)**: {greeks_values['Theta'][1]:.4f}")
greek_cols[0].markdown(f"**Rho (Call)**: {greeks_values['Rho'][0]:.4f}")
greek_cols[0].markdown(f"**Rho (Put)**: {greeks_values['Rho'][1]:.4f}")

# --- New Plotting Section ---
st.subheader("Visualisation de la Sensibilité")

# Choix du graphe
option = st.radio("Choisissez le graphique :",
                  ["Sensibilité du prix du Call", "Sensibilité du prix du Put", \
                   "Sensibilité des Greeks (Call)", "Sensibilité des Greeks (Put)"])

main_plot_fig = plt.figure(figsize=(l, w))
if option == "Sensibilité du prix du Call":
    fig_to_display = sensitivity_subplot_call(main_plot_fig, params) 
elif option == "Sensibilité du prix du Put":
    fig_to_display = sensitivity_subplot_put(main_plot_fig, params) 
elif option == "Sensibilité des Greeks (Call)": 
    fig_to_display = sensitivity_all_greeks_C(main_plot_fig, params) 
else:
    fig_to_display = sensitivity_all_greeks_P(main_plot_fig, params)

st.pyplot(fig_to_display) 







