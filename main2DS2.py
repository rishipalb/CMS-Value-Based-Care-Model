import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt  

# ✅ 1️⃣ Define the function BEFORE using it
def calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c):
    """
    Calculate shared savings (b) and rate of change (db/dN) in an ACO incentive model.
    
    Parameters:
    - N: Team Size
    - e_q: Quality Effort
    - e_c: Cost Control Effort
    - sigma_q: Quality Standard Deviation
    - sigma_c: Cost Standard Deviation
    
    Returns:
    - N_range: Team size range
    - b_values: Shared savings over team sizes
    - db_dN_values: Rate of change in shared savings
    """
    N_range = np.linspace(1, N, 100)  # Generate team sizes
    b_values = np.zeros_like(N_range)  # Shared savings values
    db_dN_values = np.zeros_like(N_range)  # Rate of change

    # Initial shared savings estimate
    b = N * (e_q + e_c)  

    for i, n in enumerate(N_range):
        db_dN = (e_q + e_c) / n
        b_values[i] = b
        db_dN_values[i] = db_dN
        b += db_dN * (N_range[1] - N_range[0])  # Increment savings

    return N_range, b_values, db_dN_values

# ✅ 2️⃣ Now, define global user inputs
st.sidebar.header("Global Financial Parameters")

# User inputs with unique keys
INITIAL_INVESTMENT = st.sidebar.number_input("Initial Investment ($M)", min_value=100, max_value=1000, value=500, step=50, key="initial_investment")
TOTAL_PROJECTED_SAVINGS = st.sidebar.number_input("Total Projected Savings ($B)", min_value=1, max_value=500, value=100, step=1, key="total_savings")
YEARS = st.sidebar.slider("Project Duration (Years)", min_value=1, max_value=20, value=5, step=1, key="project_years")

st.sidebar.header("ACO Incentive Structure Parameters")

# ACO inputs with unique keys
N = st.sidebar.slider("Team Size (N)", 1, 100, 20, key="team_size")
e_q = st.sidebar.number_input("Quality Effort (e_q)", 0.0, 10.0, 1.0, key="quality_effort")
e_c = st.sidebar.number_input("Cost Control Effort (e_c)", 0.0, 10.0, 1.0, key="cost_effort")
sigma_q = st.sidebar.number_input("Quality Standard Deviation (σ_q)", 0.1, 5.0, 1.0, key="quality_std_dev")
sigma_c = st.sidebar.number_input("Cost Standard Deviation (σ_c)", 0.1, 5.0, 1.0, key="cost_std_dev")

# ✅ 3️⃣ Now, call the function (AFTER defining it)
N_range, b_values, db_dN_values = calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c)
avg_b = np.mean(b_values)  # Dynamically calculated shared savings

# ✅ Define ACO-Adjusted Savings once (ensures consistency)
aco_adjusted_savings = TOTAL_PROJECTED_SAVINGS * (avg_b / 100)

# Sidebar: Navigation Menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Cost-Benefit Analysis",
    "Monte Carlo Simulation",
    "Implementation Cost vs. Savings",
    "Implementation Timeline",
    "Cost and Savings Breakdown",
    "ACO Incentive Structure Model"
])

# ✅ 4️⃣ Implement Cost-Benefit Analysis (Using the Fixed Function)
if page == "Cost-Benefit Analysis":
    st.title("Cost-Benefit Analysis")
    
    # ROI Calculation
    roi = TOTAL_PROJECTED_SAVINGS / (INITIAL_INVESTMENT / 1000)
    adjusted_savings = TOTAL_PROJECTED_SAVINGS * (avg_b / 100)
    adjusted_roi = roi * (avg_b / 100)

    # Display Results
    st.metric("Initial Investment", f"${INITIAL_INVESTMENT}M")
    st.metric("Total Projected Savings", f"${TOTAL_PROJECTED_SAVINGS}B")
    st.metric("Base ROI", f"{roi:.2f}x")
    st.metric("ACO-Adjusted Savings", f"${aco_adjusted_savings:.2f}B")
    st.metric("ACO-Adjusted ROI", f"{adjusted_roi:.2f}x")


# Monte Carlo Simulation Section
elif page == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation with ACO Sensitivity Analysis")

    num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
    base_annual_savings = TOTAL_PROJECTED_SAVINGS / YEARS  # Uses user-defined global YEARS
    savings_std_dev = st.slider("Savings Volatility ($B)", 1.0, 10.0, 5.0)

    # Recalculate ACO incentives dynamically
    N_range, b_values, db_dN_values = calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c)
    avg_b = np.mean(b_values)

    # Adjusted Monte Carlo savings
    adjusted_mean = base_annual_savings * (avg_b / 100)
    adjusted_std_dev = savings_std_dev * (1 + (e_q + e_c) / 20)

    # Run simulations
    simulated_savings = np.random.normal(adjusted_mean, adjusted_std_dev, num_simulations)
    total_simulated_savings = simulated_savings * YEARS  
    roi_simulations = total_simulated_savings / (INITIAL_INVESTMENT / 1000)

    # Create visualization
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Savings Distribution", "ROI Probability"))

    # Savings Histogram
    fig.add_trace(
        go.Histogram(
            x=total_simulated_savings, 
            name="Adjusted Savings",
            marker_color='#330C73',
            opacity=0.75
        ),
        row=1, col=1
    )
    
    # ROI Cumulative Probability
    sorted_roi = np.sort(roi_simulations)
    prob = np.arange(1, len(sorted_roi) + 1) / len(sorted_roi)
    fig.add_trace(
        go.Scatter(
            x=sorted_roi, 
            y=prob,
            name="ROI Probability",
            line=dict(color='#FF4B4B')
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"Monte Carlo Simulation Results (n={num_simulations})",
        showlegend=False
    )
    st.plotly_chart(fig)

    # Key Metrics
    st.subheader("Risk Analysis Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probability of Positive ROI", 
                f"{(roi_simulations > 1).mean() * 100:.1f}%")
    with col2:
        st.metric("95% Confidence Range", 
                f"${np.percentile(total_simulated_savings, 2.5):.1f}B - ${np.percentile(total_simulated_savings, 97.5):.1f}B")
    with col3:
        st.metric("Value at Risk (5%)", 
                f"${base_annual_savings * YEARS - np.percentile(total_simulated_savings, 5):.1f}B")

    # Interpretation Guide
    st.subheader("Interpretation Guide")
    st.markdown("""
    1. **Savings Histogram**: Shows the distribution of total adjusted savings over the project period.
    2. **ROI Cumulative Probability**: Helps understand the likelihood of reaching different ROI levels.
    3. **Confidence Intervals**: 
       - The 95% confidence range indicates the expected variation in outcomes.
       - **Value at Risk (VaR)** helps assess potential downside risk.
    """)


    st.metric("Total Implementation Cost", f"${INITIAL_INVESTMENT / 1000:.1f}B")

# Implementation Cost vs. Savings Section
elif page == "Implementation Cost vs. Savings":
    st.title("Implementation Cost vs. Savings Over Time")

    years_range = np.arange(1, YEARS + 1)  # Define the years
    base_savings = [TOTAL_PROJECTED_SAVINGS / YEARS * y for y in years_range]  # List of yearly savings
    implementation_cost = [INITIAL_INVESTMENT / 1000] + [0] * (YEARS - 1)  # Initial cost only in Year 1

    # ✅ Fix: Apply ACO Adjustment Correctly (Adjust Total Savings, Not Per Year)
    total_aco_adjusted_savings = TOTAL_PROJECTED_SAVINGS * (avg_b / 100)  # ✅ Corrected
    
    # ✅ Correct Adjusted Yearly Savings (Distribute over the years)
    adjusted_savings_per_year = [total_aco_adjusted_savings / YEARS] * YEARS  # List of yearly ACO-adjusted savings

    # ✅ Fix: Ensure y-values are lists, not single float values
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years_range, y=implementation_cost, name='Implementation Cost', marker_color='#e74c3c'))
    fig.add_trace(go.Bar(x=years_range, y=base_savings, name='Potential Savings', marker_color='#2ecc71'))
    fig.add_trace(go.Bar(x=years_range, y=adjusted_savings_per_year, name='ACO-Adjusted Savings', marker_color='#3498db'))  # ✅ Fixed

    fig.update_layout(
        barmode='group',
        title=f"Implementation Cost vs. Savings Over {YEARS} Years",
        xaxis_title='Years',
        yaxis_title='Amount ($B)',
        height=600
    )
    st.plotly_chart(fig)

    # ✅ Fix: Display correct Total ACO-Adjusted Savings
    st.metric("Total Projected Savings", f"${TOTAL_PROJECTED_SAVINGS}B")
    st.metric("ACO-Adjusted Savings", f"${total_aco_adjusted_savings:.2f}B")  # ✅ Fixed
    st.metric("Total Implementation Cost", f"${INITIAL_INVESTMENT / 1000:.1f}B")



# Implementation Timeline Section
elif page == "Implementation Timeline":
    st.title("Implementation Timeline")

    timeline_data = {
        "Phase": ["Phase 1", "Phase 2", "Phase 3"],
        "Years": [f"Year 1-{YEARS//3}", f"Year {YEARS//3+1}-{YEARS//3*2}", f"Year {YEARS//3*2+1}-{YEARS}"],
        "Objectives": [
            "Expand ACO incentives and participation",
            "Integrate additional conditions into bundled payment models",
            "Evaluate outcomes and refine payment structures"
        ]
    }

    df_timeline = pd.DataFrame(timeline_data)
    st.table(df_timeline)

# Cost and Savings Breakdown Section
elif page == "Cost and Savings Breakdown":
    st.title("Cost and Savings Breakdown")

    # ✅ Define Cost Breakdown Using Correct Percentages
    cost_labels = ["Infrastructure Development", "Provider Incentives", "Training & Support", "Administrative Systems", "Pilot Programs"]
    cost_percentages = [0.30, 0.25, 0.20, 0.15, 0.10]  # 30%, 25%, 20%, 15%, 10%
    cost_values = [INITIAL_INVESTMENT * p / 100 for p in cost_percentages]  # Dynamic calculation based on investment

    # ✅ Define Savings Breakdown (Total = TOTAL_PROJECTED_SAVINGS)
    savings_labels = ["ACOs", "Bundled Payments", "Readmissions"]
    savings_distribution = [0.5, 0.25, 0.25]  # 50%, 25%, 25%
    savings_components = [TOTAL_PROJECTED_SAVINGS * dist for dist in savings_distribution]

    # ✅ ACO-Adjusted Savings Calculation
    aco_adjusted_savings = TOTAL_PROJECTED_SAVINGS * (avg_b / 100)  # Apply ACO impact

    # ✅ Cost-to-Savings Ratio Calculation
    cost_to_savings_ratio = (INITIAL_INVESTMENT / 1000) / aco_adjusted_savings if aco_adjusted_savings > 0 else 0

    # ✅ Plot Cost Breakdown
    fig1, ax1 = plt.subplots()
    ax1.pie(cost_values, labels=cost_labels, autopct=lambda p: f'${p*INITIAL_INVESTMENT/100:.1f}M', startangle=140)
    ax1.set_title(f"Cost Breakdown (Total: ${INITIAL_INVESTMENT}M)")

    # ✅ Plot Savings Breakdown
    fig2, ax2 = plt.subplots()
    ax2.pie(savings_components, labels=savings_labels, 
           autopct=lambda p: f'${p*TOTAL_PROJECTED_SAVINGS/100:.1f}B',
           startangle=140, colors=['#2ecc71', '#3498db', '#9b59b6'])
    ax2.set_title(f"Savings Breakdown (Total: ${TOTAL_PROJECTED_SAVINGS:.1f}B)")

    # ✅ Display Both Charts in Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)  # ✅ Show Cost Breakdown Chart
    with col2:
        st.pyplot(fig2)  # ✅ Show Savings Breakdown Chart

    # ✅ Display Key Metrics
    st.metric("Total Savings", f"${TOTAL_PROJECTED_SAVINGS:.1f}B")
    st.metric("ACO-Adjusted Savings", f"${aco_adjusted_savings:.2f}B")  # ✅ Corrected ACO savings
    st.metric("Implementation Cost", f"${INITIAL_INVESTMENT}M")
    st.metric("Cost-to-Savings Ratio", f"{cost_to_savings_ratio:.3f}")  # ✅ Correctly calculated ratio

# ACO Incentive Structure Model Section
elif page == "ACO Incentive Structure Model":
    st.title("ACO Incentive Structure Model")

    # Ensure ACO incentive values are calculated
    N_range, b_values, db_dN_values = calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Shared Savings (b) vs Team Size (N)", "Rate of Change in Shared Savings (db/dN) vs Team Size (N)"))
    fig.add_trace(go.Scatter(x=N_range, y=b_values, mode='lines', name='Shared Savings (b)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_range, y=db_dN_values, mode='lines', name='Rate of Change (db/dN)'), row=2, col=1)
    
    st.plotly_chart(fig)

    # Explanation
    st.subheader("Model Interpretation")
    st.markdown("""
    **Insights from ACO Incentive Model:**
    - **Shared Savings (b):** Increases with team size (N), but at a diminishing rate.
    - **Rate of Change (db/dN):** Steeper for smaller teams, indicating stronger individual incentives.
    - **Optimal Team Size:** Large teams may reduce individual motivation but improve efficiency.
    """)

    # Reference
    st.subheader("Reference")
    st.markdown("""
    **Source:**  
    Frandsen, Brigham R. & Rebitzer, James B. (2014).  
    "Structuring Incentives within Organizations: The Case of Accountable Care Organizations"  
    **NBER Working Paper No. w20034.**  
    Available at [SSRN](https://ssrn.com/abstract=2424605)
    """)
