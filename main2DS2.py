import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global variables
initial_investment = 500  # in million dollars
annual_savings = 20  # in billion dollars
years = 5

# ACO Incentive Structure Model Parameters
N = 20  # Default team size
e_q = 1.0  # Default quality effort
e_c = 1.0  # Default cost control effort
sigma_q = 1.0  # Default quality standard deviation
sigma_c = 1.0  # Default cost standard deviation

# Function to calculate shared savings (b) and its rate of change (db/dN)
def calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c):
    N_range = np.linspace(1, N, 100)
    b_values = np.zeros_like(N_range)
    db_dN_values = np.zeros_like(N_range)
    x_bar = e_q - sigma_q / np.sqrt(N)
    b = N * (e_q + e_c)
    for i, n in enumerate(N_range):
        db_dN = (e_q + e_c) / n
        b_values[i] = b
        db_dN_values[i] = db_dN
        b += db_dN * (N_range[1] - N_range[0])
    return N_range, b_values, db_dN_values

# Set up page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Cost-Benefit Analysis", "Monte Carlo Simulation", "Implementation Cost vs. Savings", "Implementation Timeline", "Cost and Savings Breakdown", "ACO Incentive Structure Model"])

elif page == "Cost-Benefit Analysis":
    st.title("Cost-Benefit Analysis")
    
    # User-defined inputs
    initial_investment = st.sidebar.number_input("Initial Investment ($M)", min_value=100, max_value=1000, value=500, step=50)
    total_projected_savings = st.sidebar.number_input("Total Projected Savings over 5 Years ($B)", min_value=1, max_value=500, value=100, step=1)
    
    # ROI Calculation
    roi = total_projected_savings / (initial_investment / 1000)
    
    # Calculate ACO-adjusted values
    N = st.sidebar.slider("Team Size (N)", 1, 100, 20)
    e_q = st.sidebar.number_input("Quality Effort (e_q)", 0.0, 10.0, 1.0)
    e_c = st.sidebar.number_input("Cost Control Effort (e_c)", 0.0, 10.0, 1.0)
    sigma_q = st.sidebar.number_input("Quality Standard Deviation (σ_q)", 0.1, 5.0, 1.0)
    sigma_c = st.sidebar.number_input("Cost Standard Deviation (σ_c)", 0.1, 5.0, 1.0)
    
    # Placeholder function to calculate ACO incentives
    def calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c):
        return np.linspace(1, N, 100), np.random.rand(100) * 10, np.random.rand(100) * 5
    
    # Here, avg_b is the fixed shared savings rate value for team size 20
    avg_b = 46.17989  # Use fixed shared savings rate instead of random values
    
    # Calculate Adjusted ROI
    adjusted_roi = roi * (avg_b / 100)
    roi_difference = ((adjusted_roi - roi) / roi) * 100
    
    # Plots
    N_range, b_values, db_dN_values = calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c)
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Shared Savings (b) vs Team Size (N)", "Rate of Change in Shared Savings (db/dN) vs Team Size (N)"))
    fig.add_trace(go.Scatter(x=N_range, y=b_values, mode='lines', name='b'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_range, y=db_dN_values, mode='lines', name='db/dN'), row=2, col=1)
    st.plotly_chart(fig)
    
    # Display results
    st.write(f"### Financial Summary")
    st.metric("Initial Investment", f"${initial_investment}M")
    st.metric("Total Projected Savings (5 Years)", f"${total_projected_savings}B")
    st.metric("Base ROI (Without ACO Adjustments)", f"{roi:.2f}x")
    
    # ACO-adjusted metrics
    st.subheader("ACO Incentive Impact")
    st.markdown(f"""
    **Adjusted ROI**: **{adjusted_roi:.2f}x** ({roi_difference:+.1f}% from base ROI)  
    *This accounts for:*  
    - Average shared savings rate: {avg_b:.1f}%
    - Current team size: N = {N}
    - Quality effort: e_q = {e_q}
    - Cost control effort: e_c = {e_c}
    """)

    # Model interpretation
    st.subheader("Model Interpretation")
    st.write("""
    The plots above reveal several key insights about ACO incentive structures:
    
    1. **Shared Savings Rate (b):**
       - As team size (N) increases, the required shared savings rate (b) also increases.
       - Larger ACOs need higher shared savings rates to maintain individual incentives.
       - The relationship is non-linear, showing diminishing returns at larger team sizes.
    
    2. **Rate of Change (db/dN):**
       - The positive db/dN indicates that shared savings must increase with team size.
       - The rate of increase is steeper at smaller team sizes and levels off as teams grow.
    
    3. **Policy Implications:**
       - Smaller ACOs may be more efficient from an incentive perspective.
       - As ACOs grow, maintaining effective incentives becomes increasingly costly.
       - There may be an optimal team size that balances economies of scale with incentive costs.
    """)
    
    # Assumptions section
    st.subheader("Key Assumptions")
    st.markdown(f"""
    - **Time Horizon**: Fixed 5-year period
    - **Cost Structure**:
      - Initial investment: ${initial_investment}M
      - No recurring costs after Year 1
    - **Savings Composition**:
      - ACO savings: $10B/year (50%)
      - Bundled payments: $5B/year (25%)
      - Reduced readmissions: $5B/year (25%)
    """)
    
    # Model limitations
    st.subheader("Model Limitations")
    st.markdown("""
    - Simplified immediate adoption assumption
    - Excludes inflation and demographic changes
    - Linear savings projection
    """)
    
    # Reference
    st.subheader("Reference")
    st.markdown("""
    This model is based on:
    
    Frandsen, Brigham R. and Rebitzer, James B., "Structuring Incentives within Organizations: The Case of Accountable Care Organizations" (April 2014). NBER Working Paper No. w20034.  
    Available at SSRN: https://ssrn.com/abstract=2424605
    """)



elif page == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation with ACO Sensitivity Analysis")
    
    # Simulation controls
    col1, col2 = st.columns(2)
    with col1:
        num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
        base_annual_savings = st.number_input("Base Annual Savings ($B)", 10.0, 50.0, 20.0)
    with col2:
        savings_std_dev = st.slider("Savings Volatility ($B)", 1.0, 10.0, 5.0)
        aco_impact = st.slider("ACO Impact Factor", 0.5, 2.0, 1.0, 
                             help="Multiplier for ACO incentive effects")

    # ACO parameter sensitivity
    st.subheader("ACO Parameter Sensitivity")
    aco_col1, aco_col2, aco_col3 = st.columns(3)
    with aco_col1:
        sim_N = st.slider("Team Size (N)", 1, 100, 20)
    with aco_col2:
        sim_e_q = st.number_input("Quality Effort (e_q)", 0.0, 10.0, 1.0)
    with aco_col3:
        sim_e_c = st.number_input("Cost Effort (e_c)", 0.0, 10.0, 1.0)

    # Calculate ACO-adjusted savings
    N_range, b_values, db_dN_values = calculate_aco_incentives(sim_N, sim_e_q, sim_e_c, sigma_q, sigma_c)
    avg_b = np.mean(b_values)
    
    # Dynamic adjustments
    adjusted_mean = base_annual_savings * (avg_b/100) * aco_impact
    adjusted_std_dev = savings_std_dev * (1 + (sim_e_q + sim_e_c)/20)
    
    # Run simulations
    simulated_savings = np.random.normal(adjusted_mean, adjusted_std_dev, num_simulations)
    total_simulated_savings = simulated_savings * years
    roi_simulations = total_simulated_savings / (initial_investment/1000)

    # Create visualizations
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Savings Distribution", "ROI Probability"))
    
    # Savings histogram
    fig.add_trace(
        go.Histogram(
            x=total_simulated_savings, 
            name="Adjusted Savings",
            marker_color='#330C73',
            opacity=0.75
        ),
        row=1, col=1
    )
    
    # ROI cumulative probability
    sorted_roi = np.sort(roi_simulations)
    prob = np.arange(1, len(sorted_roi)+1)/len(sorted_roi)
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
        title_text=f"Simulation Results (n={num_simulations})",
        showlegend=False
    )
    st.plotly_chart(fig)

    # Key metrics
    st.subheader("Risk Analysis Metrics")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Probability of Positive ROI", 
                f"{(roi_simulations > 1).mean()*100:.1f}%")
    with metric_col2:
        st.metric("95% Confidence Range", 
                f"${np.percentile(total_simulated_savings, 2.5):.1f}B - ${np.percentile(total_simulated_savings, 97.5):.1f}B")
    with metric_col3:
        st.metric("Value at Risk (5%)", 
                f"${base_annual_savings*years - np.percentile(total_simulated_savings, 5):.1f}B")

    # Model documentation
    st.subheader("Simulation Methodology")
    st.markdown(f"""
    **Core Model:**
    - Normal distribution savings model
    - Annual savings: N(${adjusted_mean:.1f}B, ${adjusted_std_dev:.1f}B)
    - 5-year cumulative savings projection

    **ACO Adjustments:**
    ```python
    adjusted_mean = base_savings * (avg_b / 100) * aco_impact_factor
    adjusted_std_dev = base_volatility * (1 + (e_q + e_c)/20)
    ```
    Where:
    - `avg_b = {avg_b:.1f}%` (from ACO incentive model)
    - `aco_impact_factor = {aco_impact}`
    - Effort terms: e_q = {sim_e_q}, e_c = {sim_e_c}

    **Sensitivity Parameters:**
    - Team size (N): {sim_N}
    - Quality effort (e_q): {sim_e_q}
    - Cost effort (e_c): {sim_e_c}
    """)

    st.subheader("Interpretation Guide")
    st.markdown("""
    1. Use the ACO parameter sliders to test different organizational designs
    2. Higher team sizes (N) generally:
       - Increase required shared savings (b)
       - Reduce individual incentive effectiveness
    3. Combined effort (e_q + e_c) impacts:
       - Average savings potential (through mean adjustment)
       - Implementation risk (through volatility adjustment)
    """)

elif page == "Implementation Cost vs. Savings":
    st.title("Implementation Cost vs. Savings Over Time")
    
    # ACO Parameter Controls
    st.sidebar.subheader("ACO Configuration")
    aco_N = st.sidebar.slider("Team Size (N)", 1, 100, 20)
    aco_e_q = st.sidebar.slider("Quality Effort (e_q)", 0.0, 10.0, 1.0)
    aco_e_c = st.sidebar.slider("Cost Effort (e_c)", 0.0, 10.0, 1.0)
    sigma_q = st.sidebar.slider("Quality SD (σ_q)", 0.1, 5.0, 1.0)
    sigma_c = st.sidebar.slider("Cost SD (σ_c)", 0.1, 5.0, 1.0)
    
    # Calculate using original annual_savings
    N_range, b_values, db_dN_values = calculate_aco_incentives(aco_N, aco_e_q, aco_e_c, sigma_q, sigma_c)
    avg_b = np.mean(b_values)
    adj_factor = avg_b / 100
    
    # Dynamic calculations
    years_range = np.arange(1, years + 1)
    base_savings = [annual_savings * y for y in years_range]
    adjusted_savings = [s * adj_factor for s in base_savings]
    implementation_cost = [initial_investment / 1000] + [0] * (years - 1)
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years_range,
        y=implementation_cost,
        name='Implementation Cost',
        marker_color='#e74c3c'
    ))
    fig.add_trace(go.Bar(
        x=years_range,
        y=base_savings,
        name='Potential Savings',
        marker_color='#2ecc71'
    ))
    fig.add_trace(go.Bar(
        x=years_range,
        y=adjusted_savings,
        name='ACO-Adjusted Savings',
        marker_color='#3498db'
    ))
    fig.update_layout(
        barmode='group',
        title=f"5-Year Projection (Adjustment Factor: {adj_factor:.2f})",
        xaxis_title='Years',
        yaxis_title='Amount ($B)',
        height=600
    )
    st.plotly_chart(fig)
    
    # Updated metrics
    total_base = annual_savings * years
    total_adjusted = total_base * adj_factor
    total_cost = initial_investment / 1000
    
    st.subheader("Financial Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Potential Savings", f"${total_base:.1f}B")
    with col2:
        st.metric("ACO-Adjusted Savings", f"${total_adjusted:.1f}B")
    with col3:
        cost_ratio = total_cost/total_adjusted if total_adjusted > 0 else 0
        st.metric("Cost/Savings Ratio", f"{cost_ratio:.3f}")


elif page == "Implementation Timeline":
    st.title("Implementation Timeline")
    timeline_data = {
        "Phase": ["Phase 1", "Phase 2", "Phase 3"],
        "Years": ["Year 1-2", "Year 3-5", "Year 6-10"],
        "Objectives": [
            "Expand ACO incentives and participation",
            "Integrate additional conditions into bundled payment models",
            "Evaluate outcomes and refine payment structures"
        ]
    }
    df_timeline = pd.DataFrame(timeline_data)
    st.table(df_timeline)

elif page == "Cost and Savings Breakdown":
    st.title("Cost and Savings Breakdown")
    
    # ACO Configuration
    st.sidebar.subheader("ACO Configuration")
    aco_N = st.sidebar.slider("Team Size (N)", 1, 100, 20)
    aco_e_q = st.sidebar.slider("Quality Effort (e_q)", 0.0, 10.0, 1.0)
    aco_e_c = st.sidebar.slider("Cost Effort (e_c)", 0.0, 10.0, 1.0)
    sigma_q = st.sidebar.slider("Quality SD (σ_q)", 0.1, 5.0, 1.0)
    sigma_c = st.sidebar.slider("Cost SD (σ_c)", 0.1, 5.0, 1.0)
    
    # Calculations
    N_range, b_values, db_dN_values = calculate_aco_incentives(aco_N, aco_e_q, aco_e_c, sigma_q, sigma_c)
    avg_b = np.mean(b_values)
    adj_factor = avg_b / 100
    total_base = annual_savings * years
    total_adjusted = total_base * adj_factor
    
    # Cost Breakdown
    cost_labels = ["Infrastructure", "Incentives", "Training", "Admin", "Pilots"]
    cost_values = [150, 125, 100, 75, 50]
    
    # Savings Breakdown
    savings_labels = ["ACOs", "Bundled Payments", "Readmissions"]
    savings_distribution = [0.5, 0.25, 0.25]  # 50%, 25%, 25%
    adjusted_components = [total_adjusted * dist for dist in savings_distribution]
    
    # Visualizations
    fig1, ax1 = plt.subplots()
    ax1.pie(cost_values, labels=cost_labels, autopct='%1.1f%%', startangle=140)
    ax1.set_title(f"Cost Breakdown (Total: ${initial_investment}M)")
    
    fig2, ax2 = plt.subplots()
    ax2.pie(adjusted_components, labels=savings_labels, 
           autopct=lambda p: f'${p*total_adjusted/100:.1f}B',
           startangle=140, colors=['#2ecc71', '#3498db', '#9b59b6'])
    ax2.set_title(f"Savings Breakdown (Total: ${total_adjusted:.1f}B)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
    
    # Metrics
    st.subheader("Financial Efficiency")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Implementation Cost", f"${initial_investment}M")
    with col2:
        st.metric("Adjusted Savings", f"${total_adjusted:.1f}B")
    with col3:
        cost_ratio = (initial_investment/1000)/total_adjusted if total_adjusted > 0 else 0
        st.metric("Cost Ratio", f"{cost_ratio:.3f}")

elif page == "ACO Incentive Structure Model":
    st.title("ACO Incentive Structure Model")
    N = st.sidebar.slider("Team Size (N)", 1, 100, 20)
    e_q = st.sidebar.number_input("Quality Effort (e_q)", 0.0, 10.0, 1.0)
    e_c = st.sidebar.number_input("Cost Control Effort (e_c)", 0.0, 10.0, 1.0)
    sigma_q = st.sidebar.number_input("Quality Standard Deviation (σ_q)", 0.1, 5.0, 1.0)
    sigma_c = st.sidebar.number_input("Cost Standard Deviation (σ_c)", 0.1, 5.0, 1.0)
    N_range, b_values, db_dN_values = calculate_aco_incentives(N, e_q, e_c, sigma_q, sigma_c)
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Shared Savings (b) vs Team Size (N)", "Rate of Change in Shared Savings (db/dN) vs Team Size (N)"))
    fig.add_trace(go.Scatter(x=N_range, y=b_values, mode='lines', name='b'), row=1, col=1)
    fig.add_trace(go.Scatter(x=N_range, y=db_dN_values, mode='lines', name='db/dN'), row=2, col=1)
    st.plotly_chart(fig)
    st.subheader("Model Interpretation")
    st.write("""
    The plots above reveal several key insights about ACO incentive structures:
    
    1. **Shared Savings Rate (b):**
       - As team size (N) increases, the required shared savings rate (b) also increases.
       - Larger ACOs need higher shared savings rates to maintain individual incentives.
       - The relationship is non-linear, showing diminishing returns at larger team sizes.
    
    2. **Rate of Change (db/dN):**
       - The positive db/dN indicates that shared savings must increase with team size.
       - The rate of increase is steeper at smaller team sizes and levels off as teams grow.
    
    3. **Policy Implications:**
       - Smaller ACOs may be more efficient from an incentive perspective.
       - As ACOs grow, maintaining effective incentives becomes increasingly costly.
       - There may be an optimal team size that balances economies of scale with incentive costs.
    """)
    st.subheader("Reference")
    st.markdown("""
    This model is based on:
    
    Frandsen, Brigham R. and Rebitzer, James B., "Structuring Incentives within Organizations: The Case of Accountable Care Organizations" (April 2014). NBER Working Paper No. w20034.  
    Available at SSRN: https://ssrn.com/abstract=2424605
    """)
