# Healthcare Cost Management and Savings Analysis Dashboard

## Overview
This interactive Streamlit dashboard analyzes healthcare cost management and savings projections, with a focus on ACO (Accountable Care Organization) incentive structures. The tool helps healthcare organizations model financial impacts, visualize cost-benefit scenarios, and assess risk through Monte Carlo simulations.

## Features
- **Cost-Benefit Analysis**: Calculate ROI and savings projections with configurable delay periods
- **Monte Carlo Simulation**: Risk analysis with customizable parameters and probability distributions
- **Implementation Cost vs. Savings**: Visual comparison of costs and projected savings over time
- **Implementation Timeline**: Phased approach visualization for project rollout
- **Cost and Savings Breakdown**: Detailed pie charts showing cost allocation and savings distribution
- **ACO Incentive Structure Model**: Analysis of shared savings based on team size and effort metrics

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies
```bash
pip install streamlit
pip install pandas
pip install numpy
pip install plotly
pip install matplotlib
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/healthcare-cost-management.git
cd healthcare-cost-management
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

### Global Parameters
- **Initial Investment**: Set the initial investment amount (in millions)
- **Annual Projected Savings**: Define expected annual savings (in billions)
- **Project Duration**: Set the total project timeline
- **Delay Before Savings**: Configure when savings begin to materialize

### ACO Parameters
- **Team Size**: Adjust the number of participating providers
- **Quality Effort**: Set the expected quality improvement effort
- **Cost Control Effort**: Define cost control measures
- **Standard Deviations**: Adjust risk parameters for quality and cost

## Features Detail

### Cost-Benefit Analysis
- Calculates ROI based on investment and projected savings
- Accounts for delayed savings implementation
- Shows both base and ACO-adjusted metrics

### Monte Carlo Simulation
- Runs multiple scenarios to assess risk
- Provides confidence intervals and probability distributions
- Visualizes potential outcomes and their likelihood

### Implementation Timeline
- Breaks down the project into phases
- Shows key objectives and milestones
- Aligns with the configured project duration

### Cost and Savings Breakdown
- Visual representation of cost allocation
- Distribution of savings across different initiatives
- Detailed metrics for financial planning

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Based on ACO incentive structure research by Frandsen & Rebitzer (2014)
- Inspired by healthcare cost management best practices
- Developed with input from healthcare financial analysts

## Version History
- v1.0.0 (2024-02-14): Initial release
- v1.1.0 (2024-02-14): Added configurable savings delay period
