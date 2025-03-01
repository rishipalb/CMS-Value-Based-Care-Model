import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c):
    """
    Calculate shared savings (b) and rate of change (db/dN) in an ACO incentive model.
    """
    N_range = np.linspace(1, N, 100)
    b_values = np.zeros_like(N_range)
    db_dN_values = np.zeros_like(N_range)
    b = N * (e_q + e_c)

    for i, n in enumerate(N_range):
        db_dN = (e_q + e_c) / n
        b_values[i] = b
        db_dN_values[i] = db_dN
        b += db_dN * (N_range[1] - N_range[0])

    return N_range, b_values, db_dN_values

# Global Financial Parameters
st.sidebar.header("Health Care Markets Project")
st.sidebar.header("Global Financial Parameters")
INITIAL_INVESTMENT = st.sidebar.number_input("Initial Investment ($M)", min_value=100, max_value=1000, value=500, step=50, key="initial_investment")
ANNUAL_PROJECTED_SAVINGS = st.sidebar.number_input("Annual Projected Savings ($B)", min_value=1, max_value=50, value=5, step=1, key="total_savings")
YEARS = st.sidebar.slider("Project Duration (Years)", min_value=5, max_value=20, value=5, step=1, key="project_years")
DELAY_YEARS = st.sidebar.slider("Delay Before Savings Start (Years)", min_value=1, max_value=10, value=4, step=1, key="delay_years")  # Add this line

# ACO Parameters
st.sidebar.header("ACO Incentive Structure Parameters")
N = st.sidebar.slider("Team Size (N)", 1, 1000, 20, key="team_size")
e_q = st.sidebar.number_input("Quality Effort (e_q)", 0.0, 10.0, 1.0, key="quality_effort")
e_c = st.sidebar.number_input("Cost Control Effort (e_c)", 0.0, 10.0, 1.0, key="cost_effort")
sigma_q = st.sidebar.number_input("Quality Standard Deviation (σ_q)", 0.1, 5.0, 1.0, key="quality_std_dev")
sigma_c = st.sidebar.number_input("Cost Standard Deviation (σ_c)", 0.1, 5.0, 1.0, key="cost_std_dev")

# Calculate ACO incentives
N_range, b_values, db_dN_values = calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c)
avg_b = np.mean(b_values)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Cost-Benefit Analysis",
    "Monte Carlo Simulation",
    "Implementation Cost vs. Savings",
    "Implementation Timeline",
    "Cost and Savings Breakdown",
    "ACO Incentive Structure Model"
])

# Cost-Benefit Analysis
if page == "Cost-Benefit Analysis":
    st.title("Cost-Benefit Analysis")
    
    # Calculate ROI with delayed savings and annual savings rate
    effective_years = YEARS - DELAY_YEARS
    total_delayed_savings = ANNUAL_PROJECTED_SAVINGS * effective_years
    
    # Calculate ROI using the initial investment converted to billions
    investment_in_billions = INITIAL_INVESTMENT / 1000
    roi = total_delayed_savings / investment_in_billions
    
    # Calculate ACO-adjusted values based on the annual savings
    adjusted_annual_savings = ANNUAL_PROJECTED_SAVINGS * (avg_b / 100)
    adjusted_total_savings = adjusted_annual_savings * effective_years
    adjusted_roi = adjusted_total_savings / investment_in_billions

    # Display metrics
    st.metric("Initial Investment", f"${INITIAL_INVESTMENT}M")
    st.metric("Annual Projected Savings", f"${ANNUAL_PROJECTED_SAVINGS:.1f}B")
    st.metric("Total Projected Savings (Years {}-{})".format(DELAY_YEARS, YEARS), f"${total_delayed_savings:.1f}B")
    st.metric("Base ROI", f"{roi:.2f}x")
    st.metric("ACO-Adjusted Annual Savings", f"${adjusted_annual_savings:.2f}B")
    st.metric("ACO-Adjusted Total Savings", f"${adjusted_total_savings:.2f}B")
    st.metric("ACO-Adjusted ROI", f"{adjusted_roi:.2f}x")

# Monte Carlo Simulation
elif page == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation with ACO Sensitivity Analysis")

    num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
    effective_years = YEARS - DELAY_YEARS
    savings_std_dev = st.slider("Savings Volatility ($B)", 1.0, 10.0, 5.0)

    # Adjusted Monte Carlo savings
    adjusted_annual_savings = ANNUAL_PROJECTED_SAVINGS * (avg_b / 100)
    adjusted_std_dev = savings_std_dev * (1 + (e_q + e_c) / 20)

    # Run simulations
    simulated_annual_savings = np.random.normal(adjusted_annual_savings, adjusted_std_dev, num_simulations)
    total_simulated_savings = simulated_annual_savings * effective_years
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
                f"${ANNUAL_PROJECTED_SAVINGS * effective_years - np.percentile(total_simulated_savings, 5):.1f}B")
        

# Implementation Cost vs. Savings
elif page == "Implementation Cost vs. Savings":
    st.title("Implementation Cost vs. Savings Over Time")

    years_range = np.arange(1, YEARS + 1)
    
    # Implementation cost (only in year 1)
    implementation_cost = [INITIAL_INVESTMENT / 1000] + [0] * (YEARS - 1)
    
    # Calculate savings with delay
    base_savings = []
    adjusted_savings = []
    
    for year in years_range:
        if year <= DELAY_YEARS:  # Changed from < to <= to include the delay year
            base_savings.append(0)
            adjusted_savings.append(0)
        else:
            base_savings.append(ANNUAL_PROJECTED_SAVINGS)
            adjusted_savings.append(ANNUAL_PROJECTED_SAVINGS * (avg_b / 100))

    # Create visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years_range, y=implementation_cost, name='Implementation Cost', marker_color='#e74c3c'))
    fig.add_trace(go.Bar(x=years_range, y=base_savings, name='Potential Savings', marker_color='#2ecc71'))
    fig.add_trace(go.Bar(x=years_range, y=adjusted_savings, name='ACO-Adjusted Savings', marker_color='#3498db'))

    fig.update_layout(
        barmode='group',
        title=f"Implementation Cost vs. Savings Over {YEARS} Years (Savings Start Year {DELAY_YEARS + 1})",  # Updated to show correct start year
        xaxis_title='Years',
        yaxis_title='Amount ($B)',
        height=600
    )
    st.plotly_chart(fig)

    effective_years = YEARS - DELAY_YEARS
    total_base_savings = ANNUAL_PROJECTED_SAVINGS * effective_years
    total_adjusted_savings = (ANNUAL_PROJECTED_SAVINGS * (avg_b / 100)) * effective_years
    
    st.metric("Total Projected Savings", f"${total_base_savings:.2f}B")
    st.metric("Total ACO-Adjusted Savings", f"${total_adjusted_savings:.2f}B")
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

    # ✅ Calculate total savings over effective period
    effective_years = YEARS - DELAY_YEARS
    total_projected_savings = ANNUAL_PROJECTED_SAVINGS * effective_years
    
    # ✅ Calculate ACO-adjusted total savings
    aco_adjusted_total = total_projected_savings * (avg_b / 100)

    # ✅ Define Savings Breakdown (using ACO-adjusted numbers)
    savings_labels = ["ACOs", "Bundled Payments", "Readmissions"]
    savings_distribution = [0.5, 0.25, 0.25]  # 50%, 25%, 25%
    savings_components = [aco_adjusted_total * dist for dist in savings_distribution]  # Using ACO-adjusted total

    # ✅ ACO-Adjusted Savings Calculation
    aco_adjusted_savings = total_projected_savings * (avg_b / 100)  # Apply ACO impact

    # ✅ Cost-to-Savings Ratio Calculation
    cost_to_savings_ratio = (INITIAL_INVESTMENT / 1000) / aco_adjusted_savings if aco_adjusted_savings > 0 else 0

    # ✅ Plot Cost Breakdown
    fig1, ax1 = plt.subplots()
    ax1.pie(cost_values, labels=cost_labels, autopct=lambda p: f'${p*INITIAL_INVESTMENT/100:.1f}M', startangle=140)
    ax1.set_title(f"Cost Breakdown (Total: ${INITIAL_INVESTMENT}M)")

    # ✅ Plot Savings Breakdown (now using ACO-adjusted numbers)
    fig2, ax2 = plt.subplots()
    ax2.pie(savings_components, labels=savings_labels, 
           autopct=lambda p: f'${p*aco_adjusted_total/100:.1f}B',
           startangle=140, colors=['#2ecc71', '#3498db', '#9b59b6'])
    ax2.set_title(f"ACO-Adjusted Savings Breakdown\nTotal: ${aco_adjusted_total:.1f}B (Years {DELAY_YEARS}-{YEARS})")

    # ✅ Display Both Charts in Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)  # ✅ Show Cost Breakdown Chart
    with col2:
        st.pyplot(fig2)  # ✅ Show Savings Breakdown Chart

    # ✅ Display Key Metrics
    st.metric("Annual Projected Savings", f"${ANNUAL_PROJECTED_SAVINGS:.1f}B")
    st.metric("Total Projected Savings (Years {}-{})".format(DELAY_YEARS, YEARS), f"${total_projected_savings:.1f}B")
    st.metric("ACO-Adjusted Total Savings", f"${aco_adjusted_savings:.2f}B")  # ✅ Corrected ACO savings
    st.metric("Implementation Cost", f"${INITIAL_INVESTMENT}M")
    st.metric("Cost-to-Savings Ratio", f"{cost_to_savings_ratio:.3f}")  # ✅ Correctly calculated ratio


# ACO Incentive Structure Model
elif page == "ACO Incentive Structure Model":
    st.title("ACO Incentive Structure Model")

    # Calculate ACO incentive values
    N_range, b_values, db_dN_values = calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c)

    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=("Shared Savings (b) vs Team Size (N)", 
                                     "Rate of Change in Shared Savings (db/dN) vs Team Size (N)"))
    
    fig.add_trace(go.Scatter(x=N_range, y=b_values, mode='lines', name='Shared Savings (b)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_range, y=db_dN_values, mode='lines', name='Rate of Change (db/dN)'), row=2, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig)

    # Add explanation about delay impact
    st.info(f"""
    Model Interpretation with Delayed Savings:
    - Savings begin in Year {DELAY_YEARS}
    - Total effective savings period: {YEARS - DELAY_YEARS} years
    - ACO incentives are calculated on realized savings only
    """)

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
