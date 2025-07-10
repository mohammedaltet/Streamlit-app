# Superstore Sales Dashboard & Analytics Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-app-ffh2ba8zksxbn65vxoowsd.streamlit.app/)

A comprehensive business intelligence dashboard built with Streamlit that provides deep insights into superstore sales data through interactive visualizations, statistical analysis, and predictive modeling.

## 🚀 Live Demo

**[Try the app here](https://app-app-ffh2ba8zksxbn65vxoowsd.streamlit.app/)**

## 📊 Features

### Interactive Analytics Dashboard
- **Real-time Filtering**: Filter by date range, category, region, and customer segment
- **Dynamic Visualizations**: Interactive charts and graphs that update based on selected filters
- **Multi-tab Interface**: Organized analysis sections for better user experience

### Statistical Analysis
- **Descriptive Statistics**: Complete statistical summary of sales data
- **Outlier Detection**: Automated identification using IQR methodology
- **Distribution Analysis**: Sales distribution with outlier highlighting
- **Correlation Studies**: Relationship analysis between different variables

### Time Series Analysis
- **Trend Analysis**: Monthly and quarterly sales trends
- **Seasonal Decomposition**: Breakdown of trend, seasonal, and residual components
- **Pattern Recognition**: Identification of business cycles and seasonal effects

### Predictive Modeling
- **Sales Forecasting**: Up to 24-month ahead predictions using Random Forest Regression
- **Growth Metrics**: Calculated growth rates and trend projections
- **Category-specific Forecasts**: Individual predictions for different product categories

### Business Intelligence
- **KPI Tracking**: Total sales, orders, average sale value, shipping metrics
- **Regional Analysis**: Performance breakdown by geographic regions
- **Category Performance**: Product category analysis and profitability assessment
- **Shipping Analytics**: Analysis of shipping modes and delivery performance

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning (Random Forest Regression)
- **Matplotlib & Seaborn**: Statistical plotting
- **Statsmodels**: Time series analysis

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
plotly>=5.15.0
```

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/superstore-sales-dashboard.git
cd superstore-sales-dashboard
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

## 📁 Data Structure

The application expects a CSV file named `Superstore Sales Dataset.csv` with the following columns:
- Order ID, Order Date, Ship Date, Ship Mode
- Customer ID, Customer Name, Segment
- City, State, Postal Code, Region
- Product ID, Category, Sub-Category, Product Name
- Sales

If the dataset is not found, the app automatically generates demo data for demonstration purposes.

## 🎯 Use Cases

### Business Analysts
- Track sales performance across different dimensions
- Identify seasonal trends and business patterns
- Monitor regional and category performance

### Sales Managers
- Forecast future sales for planning purposes
- Analyze shipping performance and customer segments
- Identify top-performing products and regions

### Data Scientists
- Explore time series decomposition techniques
- Implement and test forecasting models
- Analyze statistical distributions and outliers

## 📈 Key Metrics & Insights

The dashboard provides insights into:
- **Sales Performance**: Total sales, average order value, growth trends
- **Geographic Analysis**: Regional performance, top-performing states
- **Product Analysis**: Category performance, sub-category breakdown
- **Operational Metrics**: Shipping performance, delivery times
- **Predictive Analytics**: Future sales forecasts with confidence intervals

## 🔄 Data Processing Pipeline

1. **Data Loading**: Automatic CSV detection with fallback to demo data
2. **Data Cleaning**: Handle missing values, data type conversions
3. **Feature Engineering**: Create time-based features (year, month, quarter)
4. **Statistical Analysis**: Calculate outliers, descriptive statistics
5. **Visualization**: Generate interactive charts and graphs
6. **Modeling**: Train Random Forest model for forecasting

## 🎨 Dashboard Sections

### 1. Sales Overview
- Key performance indicators
- Sales distribution analysis
- Top products by revenue

### 2. Category Analysis
- Sales breakdown by category and sub-category
- Interactive treemap visualization
- Order volume analysis

### 3. Time Series Analysis
- Monthly and quarterly trends
- Seasonal pattern identification
- Multi-category trend comparison

### 4. Regional Analysis
- Geographic performance breakdown
- State-wise sales analysis
- Regional comparison charts

### 5. Shipping Analysis
- Delivery performance metrics
- Shipping mode distribution
- Average delivery times

### 6. Forecast
- Predictive sales modeling
- Growth rate calculations
- Time series decomposition

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset: Superstore Sales Dataset
- Built with Streamlit's amazing framework
- Visualization powered by Plotly

---

**⭐ If you found this project helpful, please give it a star!**
