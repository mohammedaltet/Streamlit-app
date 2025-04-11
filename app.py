import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E7D32;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Superstore Sales Dashboard</h1>", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to read the CSV file
        try:
            df = pd.read_csv("Superstore Sales Dataset.csv")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return create_demo_data()
        
        # Basic data cleaning
        try:
            # Handle Postal Code - if present
            if 'Postal Code' in df.columns:
                df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)
            
            # Lowercase text columns if they exist
            text_columns = ['Ship Mode', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category', 'Product Name']
            text_columns = [col for col in text_columns if col in df.columns]
            if text_columns:
                df[text_columns] = df[text_columns].apply(lambda x: x.str.lower())
            
            # Convert date columns
            if 'Order Date' in df.columns:
                df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
            if 'Ship Date' in df.columns:
                df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
            
            # Calculate additional fields
            if 'Order Date' in df.columns and 'Ship Date' in df.columns:
                df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
            else:
                df['Shipping Days'] = 0
                
            if 'Order Date' in df.columns:
                # First ensure the dates are properly parsed
                df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
                
                # Handle NaN/Missing values properly before conversion to int
                # For Year
                df['Order Year'] = df['Order Date'].dt.year
                # Fill NaN values with a reasonable default before converting to int
                df['Order Year'] = df['Order Year'].fillna(-1).astype('Int64')  # Int64 is pandas nullable integer type
                
                # For Month
                df['Order Month'] = df['Order Date'].dt.month
                df['Order Month'] = df['Order Month'].fillna(-1).astype('Int64')
                
                # For Quarter
                df['Order Quarter'] = df['Order Date'].dt.quarter
                df['Order Quarter'] = df['Order Quarter'].fillna(-1).astype('Int64')
                
                # Create YearMonth string safely
                df['Order YearMonth'] = df['Order Date'].dt.strftime('%Y-%m')
                
                # Create YearQuarter string safely - handle potential NaN values
                valid_mask = (df['Order Year'].notna()) & (df['Order Quarter'].notna())
                df['Order YearQuarter'] = None  # Initialize with None
                
                # Only create YearQuarter for rows with valid Year and Quarter
                year_int = df.loc[valid_mask, 'Order Year'].astype(int)
                quarter_int = df.loc[valid_mask, 'Order Quarter'].astype(int)
                df.loc[valid_mask, 'Order YearQuarter'] = year_int.astype(str) + '-Q' + quarter_int.astype(str)
            
            # Drop unnecessary columns
            if 'Country' in df.columns and 'Row ID' in df.columns:
                df.drop(["Country", "Row ID"], axis=1, inplace=True)
            
            # Convert categorical columns
            categorical_col = ["Segment", "State", "Region", "Category", "Sub-Category"]
            categorical_col = [col for col in categorical_col if col in df.columns]
            if categorical_col:
                df[categorical_col] = df[categorical_col].astype("category")
            
            # Calculate outliers
            if 'Sales' in df.columns:
                Q1 = df['Sales'].quantile(0.25)
                Q3 = df['Sales'].quantile(0.75)
                IQR = Q3 - Q1
                outlier_threshold_low = Q1 - 1.5 * IQR
                outlier_threshold_high = Q3 + 1.5 * IQR
                df['Outlier'] = (df['Sales'] < outlier_threshold_low) | (df['Sales'] > outlier_threshold_high)
            else:
                df['Outlier'] = False
                
        except Exception as e:
            st.error(f"Error during data preprocessing: {e}")
            return create_demo_data()
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create a sample dataframe for demonstration purposes
        st.info("Loading demo data instead...")
        return create_demo_data()

def create_demo_data():
    """Creates demo data if the actual file isn't available"""
    # Create dates from 2015 to 2018
    start_date = pd.Timestamp('2015-01-01')
    end_date = pd.Timestamp('2018-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Sample size
    sample_size = 9800
    
    # Create a dataframe with random data
    np.random.seed(42)
    df = pd.DataFrame({
        'Order ID': [f'ORD-{i:05d}' for i in range(1, sample_size + 1)],
        'Order Date': np.random.choice(dates, size=sample_size),
        'Ship Date': None,  # Will fill in after
        'Ship Mode': np.random.choice(['standard class', 'second class', 'first class', 'same day'], size=sample_size, p=[0.6, 0.2, 0.15, 0.05]),
        'Customer ID': [f'CUS-{i:04d}' for i in np.random.randint(1, 800, size=sample_size)],
        'Customer Name': [f'Customer {i}' for i in np.random.randint(1, 800, size=sample_size)],
        'Segment': np.random.choice(['consumer', 'corporate', 'home office'], size=sample_size, p=[0.5, 0.3, 0.2]),
        'City': np.random.choice([f'City {i}' for i in range(1, 530)], size=sample_size),
        'State': np.random.choice([f'State {i}' for i in range(1, 50)], size=sample_size),
        'Postal Code': np.random.randint(10000, 99999, size=sample_size),
        'Region': np.random.choice(['west', 'east', 'central', 'south'], size=sample_size, p=[0.32, 0.28, 0.23, 0.17]),
        'Product ID': [f'PRD-{i:04d}' for i in np.random.randint(1, 1850, size=sample_size)],
        'Category': np.random.choice(['office supplies', 'furniture', 'technology'], size=sample_size, p=[0.6, 0.21, 0.19]),
        'Sub-Category': np.random.choice(['chairs', 'phones', 'storage', 'tables', 'supplies', 'bookcases', 'machines', 'accessories'], size=sample_size),
        'Product Name': [f'Product {i}' for i in np.random.randint(1, 1850, size=sample_size)],
        'Sales': np.random.gamma(2, 100, size=sample_size)  # Gamma distribution for sales
    })
    
    # Adjust Ship Date to be a few days after Order Date
    df['Ship Date'] = df['Order Date'] + pd.to_timedelta(np.random.randint(1, 10, size=sample_size), unit='D')
    
    # Calculate additional fields
    df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Quarter'] = df['Order Date'].dt.quarter
    df['Order YearMonth'] = df['Order Date'].dt.strftime('%Y-%m')
    df['Order YearQuarter'] = df['Order Year'].astype(int).astype(str) + '-Q' + df['Order Quarter'].astype(int).astype(str)
    
    # Calculate outliers
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_low = Q1 - 1.5 * IQR
    outlier_threshold_high = Q3 + 1.5 * IQR
    df['Outlier'] = (df['Sales'] < outlier_threshold_low) | (df['Sales'] > outlier_threshold_high)
    
    return df

# Load the data
df = load_data()

# Sidebar
st.sidebar.markdown("## Dashboard Controls")
st.sidebar.markdown("### Filters")

# Date range filter
min_date = df['Order Date'].min().date()
max_date = df['Order Date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['Order Date'].dt.date >= start_date) & (df['Order Date'].dt.date <= end_date)]
else:
    filtered_df = df.copy()

# Category filter
categories = ["All"] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Select Category", categories)

if selected_category != "All":
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]

# Region filter
regions = ["All"] + sorted(df['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)

if selected_region != "All":
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

# Segment filter
segments = ["All"] + sorted(df['Segment'].unique().tolist())
selected_segment = st.sidebar.selectbox("Select Segment", segments)

if selected_segment != "All":
    filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]

# Analysis options
st.sidebar.markdown("### Analysis Options")
analysis_options = st.sidebar.multiselect(
    "Select Analysis Views",
    ["Sales Overview", "Category Analysis", "Time Series Analysis", "Regional Analysis", "Shipping Analysis", "Forecast"],
    default=["Sales Overview"]
)

# Main content
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Total Sales</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${filtered_df['Sales'].sum():,.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Total Orders</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{filtered_df['Order ID'].nunique():,}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Average Sale</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>${filtered_df['Sales'].mean():,.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Avg. Shipping Days</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{filtered_df['Shipping Days'].mean():.1f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis Sections
    if "Sales Overview" in analysis_options:
        st.markdown("<h2 class='sub-header'>Sales Overview</h2>", unsafe_allow_html=True)
        
        # Sales distribution
        fig = px.histogram(
            filtered_df,
            x="Sales",
            color="Outlier",
            marginal="box",
            title="Sales Distribution (with Outliers Highlighted)",
            color_discrete_map={True: "red", False: "blue"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sales Statistics")
            stats_df = pd.DataFrame({
                "Metric": ["Total Sales", "Average Sale", "Median Sale", "Min Sale", "Max Sale"],
                "Value": [
                    f"${filtered_df['Sales'].sum():,.2f}",
                    f"${filtered_df['Sales'].mean():,.2f}",
                    f"${filtered_df['Sales'].median():,.2f}",
                    f"${filtered_df['Sales'].min():,.2f}",
                    f"${filtered_df['Sales'].max():,.2f}"
                ]
            })
            st.table(stats_df)
        
        with col2:
            # Top products
            st.markdown("### Top 5 Products by Sales")
            top_products = filtered_df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(5)
            
            fig = px.bar(
                x=top_products.values,
                y=top_products.index,
                orientation='h',
                labels={'x': 'Sales ($)', 'y': 'Product Name'}
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    if "Category Analysis" in analysis_options:
        st.markdown("<h2 class='sub-header'>Category Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by category
            category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
            fig = px.pie(
                category_sales, 
                values='Sales', 
                names='Category',
                title='Sales by Category',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Number of orders by category
            orders_by_category = filtered_df.groupby('Category')['Order ID'].nunique().reset_index()
            orders_by_category.columns = ['Category', 'Number of Orders']
            
            fig = px.bar(
                orders_by_category,
                x='Category',
                y='Number of Orders',
                title='Number of Orders by Category',
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sub-category analysis
        st.markdown("### Sub-Category Analysis")
        
        subcategory_sales = filtered_df.groupby(['Category', 'Sub-Category'])['Sales'].sum().reset_index()
        subcategory_orders = filtered_df.groupby(['Category', 'Sub-Category'])['Order ID'].nunique().reset_index()
        subcategory_orders.columns = ['Category', 'Sub-Category', 'Number of Orders']
        
        subcategory_analysis = subcategory_sales.merge(subcategory_orders, on=['Category', 'Sub-Category'])
        subcategory_analysis['Average Sale per Order'] = subcategory_analysis['Sales'] / subcategory_analysis['Number of Orders']
        
        fig = px.treemap(
            subcategory_analysis,
            path=[px.Constant("All Categories"), 'Category', 'Sub-Category'],
            values='Sales',
            color='Average Sale per Order',
            color_continuous_scale='RdBu',
            title='Sales by Category and Sub-Category'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if "Time Series Analysis" in analysis_options:
        st.markdown("<h2 class='sub-header'>Time Series Analysis</h2>", unsafe_allow_html=True)
        
        # Monthly sales trend
        monthly_sales = filtered_df.groupby('Order YearMonth')['Sales'].sum().reset_index()
        monthly_sales['Order YearMonth'] = pd.to_datetime(monthly_sales['Order YearMonth'])
        monthly_sales = monthly_sales.sort_values('Order YearMonth')
        
        fig = px.line(
            monthly_sales,
            x='Order YearMonth',
            y='Sales',
            title='Monthly Sales Trend',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quarterly sales trend
        quarterly_sales = filtered_df.groupby('Order YearQuarter')['Sales'].sum().reset_index()
        # Sort quarters chronologically - handle various possible formats
        try:
            # First try the expected format
            quarterly_sales['Year'] = quarterly_sales['Order YearQuarter'].str.split('-').str[0].astype(int)
            quarterly_sales['Quarter'] = quarterly_sales['Order YearQuarter'].str.split('-Q').str[1].astype(int)
        except ValueError:
            # If that fails, try to handle potential floating point values
            quarterly_sales['Year'] = quarterly_sales['Order YearQuarter'].str.split('-').str[0].astype(float).astype(int)
            # Handle possible floating point in quarter as well
            quarterly_sales['Quarter'] = pd.to_numeric(quarterly_sales['Order YearQuarter'].str.split('-Q').str[1], errors='coerce').fillna(1).astype(int)
        except Exception as e:
            # If all else fails, create simple sequential numbering
            st.warning(f"Could not parse year-quarter format. Using sequential ordering instead: {str(e)}")
            quarterly_sales['Order_Seq'] = range(len(quarterly_sales))
            quarterly_sales = quarterly_sales.sort_values('Order_Seq')
        else:
            quarterly_sales = quarterly_sales.sort_values(['Year', 'Quarter'])
        
        fig = px.bar(
            quarterly_sales,
            x='Order YearQuarter',
            y='Sales',
            title='Quarterly Sales Trend',
            color='Year',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly sales by category
        if selected_category == "All":
            monthly_category = filtered_df.groupby(['Order YearMonth', 'Category'])['Sales'].sum().reset_index()
            monthly_category['Order YearMonth'] = pd.to_datetime(monthly_category['Order YearMonth'])
            monthly_category = monthly_category.sort_values('Order YearMonth')
            
            fig = px.line(
                monthly_category,
                x='Order YearMonth',
                y='Sales',
                color='Category',
                title='Monthly Sales Trend by Category',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    if "Regional Analysis" in analysis_options:
        st.markdown("<h2 class='sub-header'>Regional Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by region
            region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
            
            fig = px.pie(
                region_sales,
                values='Sales',
                names='Region',
                title='Sales by Region',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Orders by region
            region_orders = filtered_df.groupby('Region')['Order ID'].nunique().reset_index()
            region_orders.columns = ['Region', 'Number of Orders']
            
            fig = px.bar(
                region_orders,
                x='Region',
                y='Number of Orders',
                title='Number of Orders by Region',
                color='Region',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top 10 states by sales
        state_sales = filtered_df.groupby('State')['Sales'].sum().sort_values(ascending=False).reset_index()
        top_10_states = state_sales.head(10)
        
        fig = px.bar(
            top_10_states,
            x='State',
            y='Sales',
            title='Top 10 States by Sales',
            color='Sales',
            color_continuous_scale=px.colors.sequential.Blugrn
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if "Shipping Analysis" in analysis_options:
        st.markdown("<h2 class='sub-header'>Shipping Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Shipping mode distribution
            shipping_count = filtered_df.groupby('Ship Mode').size().reset_index(name='Count')
            
            fig = px.pie(
                shipping_count,
                values='Count',
                names='Ship Mode',
                title='Distribution of Shipping Modes',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average shipping days by mode
            shipping_days = filtered_df.groupby('Ship Mode')['Shipping Days'].mean().reset_index()
            
            fig = px.bar(
                shipping_days,
                x='Ship Mode',
                y='Shipping Days',
                title='Average Shipping Days by Mode',
                color='Ship Mode',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Shipping days distribution
        fig = px.histogram(
            filtered_df,
            x='Shipping Days',
            color='Ship Mode',
            title='Distribution of Shipping Days',
            marginal='box',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if "Forecast" in analysis_options:
        st.markdown("<h2 class='sub-header'>Sales Forecast</h2>", unsafe_allow_html=True)
        
        # Only allow forecasting if we have enough data
        unique_dates = filtered_df['Order Date'].dt.strftime('%Y-%m').nunique()
        
        if unique_dates < 12:
            st.warning("Insufficient data for forecasting. Please select a date range spanning at least 12 months.")
        else:
            # Helper function for forecasting
            def generate_forecast(data, periods=12):
                try:
                    # Group data by month
                    monthly_data = data.groupby('Order YearMonth').agg({
                        'Sales': 'sum',
                        'Order Date': 'min'
                    }).reset_index()
                    
                    monthly_data = monthly_data.sort_values('Order Date')
                    
                    # Create features
                    monthly_data['Month'] = monthly_data['Order Date'].dt.month  # Seasonal component
                    monthly_data['Month_Num'] = range(1, len(monthly_data) + 1)  # Trend component
                    
                    # Split data
                    X = monthly_data[['Month', 'Month_Num']]
                    y = monthly_data['Sales']
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    # Create future dates
                    last_date = monthly_data['Order Date'].max()
                    last_month_num = monthly_data['Month_Num'].max()
                    future_dates = pd.date_range(
                        start=pd.Timestamp(last_date) + pd.DateOffset(months=1),
                        periods=periods,
                        freq='MS'
                    )
                    
                    # Create future features
                    future = pd.DataFrame({
                        'Order YearMonth': future_dates.strftime('%Y-%m'),
                        'Order Date': future_dates,
                        'Month': future_dates.month,
                        'Month_Num': range(last_month_num + 1, last_month_num + periods + 1)
                    })
                    
                    # Make predictions
                    future['Predicted_Sales'] = model.predict(future[['Month', 'Month_Num']])
                    
                    # Combine historical and future data for visualization
                    historical = monthly_data[['Order YearMonth', 'Order Date', 'Sales']].copy()
                    historical['Data_Type'] = 'Historical'
                    historical = historical.rename(columns={'Sales': 'Value'})
                    
                    forecast = future[['Order YearMonth', 'Order Date', 'Predicted_Sales']].copy()
                    forecast['Data_Type'] = 'Forecast'
                    forecast = forecast.rename(columns={'Predicted_Sales': 'Value'})
                    
                    combined = pd.concat([
                        historical[['Order YearMonth', 'Order Date', 'Value', 'Data_Type']],
                        forecast[['Order YearMonth', 'Order Date', 'Value', 'Data_Type']]
                    ])
                    
                    return combined, future
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    # Create empty dataframes with required structure for graceful failure
                    combined = pd.DataFrame(columns=['Order YearMonth', 'Order Date', 'Value', 'Data_Type'])
                    future = pd.DataFrame(columns=['Order YearMonth', 'Order Date', 'Predicted_Sales'])
                    return combined, future
            
            # Let user select forecast periods
            forecast_periods = st.slider(
                "Forecast Months",
                min_value=3,
                max_value=24,
                value=12,
                step=3,
                help="Select the number of months to forecast into the future"
            )
            
            # Generate forecasts
            if selected_category == "All":
                # Create tabs for overall forecast and category-specific forecasts
                tab1, tab2 = st.tabs(["Overall Forecast", "Category Forecasts"])
                
                with tab1:
                    # Overall forecast
                    combined_data, future_data = generate_forecast(filtered_df, forecast_periods)
                    
                    fig = px.line(
                        combined_data,
                        x='Order Date',
                        y='Value',
                        color='Data_Type',
                        title=f'Sales Forecast - Next {forecast_periods} Months',
                        color_discrete_map={
                            'Historical': 'blue',
                            'Forecast': 'red'
                        },
                        markers=True
                    )
                    
                    # Add a vertical line at the forecast start - safely
                    if 'combined_data' in locals() and not combined_data.empty and 'Data_Type' in combined_data.columns:
                        forecast_data = combined_data[combined_data['Data_Type'] == 'Forecast']
                        if not forecast_data.empty:
                            try:
                                forecast_start = forecast_data['Order Date'].min()
                                
                                # Convert to string format for plotly
                                if isinstance(forecast_start, pd.Timestamp):
                                    forecast_start_str = forecast_start.strftime('%Y-%m-%d')
                                    fig.add_vline(
                                        x=forecast_start_str,
                                        line_dash="dash",
                                        line_color="gray"
                                    )
                                    # Add the annotation through another method
                                    fig.add_annotation(
                                        x=forecast_start_str,
                                        y=0.5,
                                        yref="paper",
                                        text="Forecast Start",
                                        showarrow=False,
                                        textangle=-90
                                    )
                            except Exception as e:
                                st.warning(f"Could not add forecast line: {e}")
                                # Continue without the line
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show forecast data
                    st.markdown("### Forecast Data")
                    forecast_table = future_data[['Order YearMonth', 'Predicted_Sales']].copy()
                    forecast_table.columns = ['Month', 'Forecasted Sales']
                    forecast_table['Forecasted Sales'] = forecast_table['Forecasted Sales'].map('${:,.2f}'.format)
                    st.table(forecast_table)
                
                with tab2:
                    # Category forecasts
                    categories = filtered_df['Category'].unique()
                    
                    # Create a plot with forecasts for each category
                    fig = go.Figure()
                    
                    for i, category in enumerate(categories):
                        category_data = filtered_df[filtered_df['Category'] == category]
                        combined_data, _ = generate_forecast(category_data, forecast_periods)
                        
                        if not combined_data.empty:
                            # Historical data
                            historical = combined_data[combined_data['Data_Type'] == 'Historical']
                            if not historical.empty:
                                fig.add_trace(go.Scatter(
                                    x=historical['Order Date'],
                                    y=historical['Value'],
                                    mode='lines+markers',
                                    name=f'{category} (Historical)',
                                    line=dict(width=2)
                                ))
                            
                            # Forecast data
                            forecast = combined_data[combined_data['Data_Type'] == 'Forecast']
                            if not forecast.empty:
                                fig.add_trace(go.Scatter(
                                    x=forecast['Order Date'],
                                    y=forecast['Value'],
                                    mode='lines+markers',
                                    line=dict(dash='dash', width=2),
                                    name=f'{category} (Forecast)'
                                ))
                    
                    # Add a vertical line at the forecast start
                    if 'combined_data' in locals() and not combined_data.empty and 'Data_Type' in combined_data.columns:
                        forecast_data = combined_data[combined_data['Data_Type'] == 'Forecast']
                        if not forecast_data.empty:
                            try:
                                forecast_start = forecast_data['Order Date'].min()
                                
                                # Convert to string format for plotly
                                if isinstance(forecast_start, pd.Timestamp):
                                    forecast_start_str = forecast_start.strftime('%Y-%m-%d')
                                    fig.add_vline(
                                        x=forecast_start_str,
                                        line_dash="dash",
                                        line_color="gray"
                                    )
                                    # Add the annotation through another method
                                    fig.add_annotation(
                                        x=forecast_start_str,
                                        y=0.5,
                                        yref="paper",
                                        text="Forecast Start",
                                        showarrow=False,
                                        textangle=-90
                                    )
                            except Exception as e:
                                st.warning(f"Could not add forecast line: {e}")
                                # Continue without the line
                    
                    fig.update_layout(
                        title=f'Category Sales Forecast - Next {forecast_periods} Months',
                        xaxis_title='Date',
                        yaxis_title='Sales ($)',
                        legend_title='Category',
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Category-specific forecast
                combined_data, future_data = generate_forecast(filtered_df, forecast_periods)
                
                fig = px.line(
                    combined_data,
                    x='Order Date',
                    y='Value',
                    color='Data_Type',
                    title=f'{selected_category} Sales Forecast - Next {forecast_periods} Months',
                    color_discrete_map={
                        'Historical': 'blue',
                        'Forecast': 'red'
                    },
                    markers=True
                )
                
                # Add a vertical line at the forecast start
                if 'combined_data' in locals() and not combined_data.empty and 'Data_Type' in combined_data.columns:
                    forecast_data = combined_data[combined_data['Data_Type'] == 'Forecast']
                    if not forecast_data.empty:
                        try:
                            forecast_start = forecast_data['Order Date'].min()
                            
                            # Convert to string format for plotly
                            if isinstance(forecast_start, pd.Timestamp):
                                forecast_start_str = forecast_start.strftime('%Y-%m-%d')
                                fig.add_vline(
                                    x=forecast_start_str,
                                    line_dash="dash",
                                    line_color="gray"
                                )
                                # Add the annotation through another method
                                fig.add_annotation(
                                    x=forecast_start_str,
                                    y=0.5,
                                    yref="paper",
                                    text="Forecast Start",
                                    showarrow=False,
                                    textangle=-90
                                )
                        except Exception as e:
                            st.warning(f"Could not add forecast line: {e}")
                            # Continue without the line
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast data
                st.markdown("### Forecast Data")
                forecast_table = future_data[['Order YearMonth', 'Predicted_Sales']].copy()
                forecast_table.columns = ['Month', 'Forecasted Sales']
                forecast_table['Forecasted Sales'] = forecast_table['Forecasted Sales'].map('${:,.2f}'.format)
                st.table(forecast_table)
            
            # Calculate expected growth
            try:
                if not combined_data.empty and 'Data_Type' in combined_data.columns:
                    historical_data = combined_data[combined_data['Data_Type'] == 'Historical']
                    forecast_data = combined_data[combined_data['Data_Type'] == 'Forecast']
                    
                    if not historical_data.empty and not forecast_data.empty:
                        last_historical_value = historical_data['Value'].iloc[-1]
                        last_forecast_value = forecast_data['Value'].iloc[-1]
                        growth_amount = last_forecast_value - last_historical_value
                        growth_percent = (growth_amount / last_historical_value) * 100 if last_historical_value != 0 else 0
                    else:
                        growth_amount = 0
                        growth_percent = 0
                else:
                    growth_amount = 0
                    growth_percent = 0
            except Exception as e:
                st.warning(f"Could not calculate growth metrics: {str(e)}")
                growth_amount = 0
                growth_percent = 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-label'>Forecasted Growth Amount</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>${growth_amount:,.2f}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-label'>Forecasted Growth Rate</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{growth_percent:.2f}%</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Time series decomposition section (if enough data)
            st.markdown("### Sales Trend Decomposition")
            st.write("Analyzing the components of the sales time series to understand underlying patterns.")
            
            # Monthly data for decomposition
            monthly_data = filtered_df.groupby('Order YearMonth').agg({
                'Sales': 'sum',
                'Order Date': 'min'
            }).reset_index()
            monthly_data = monthly_data.sort_values('Order Date')
            monthly_ts = monthly_data.set_index('Order Date')['Sales']
            
            if len(monthly_ts) >= 24:  # Need at least 2 years of data for good decomposition
                try:
                    # Perform decomposition
                    try:
                        decomposition = seasonal_decompose(monthly_ts, model='multiplicative', period=12)
                    except:
                        # Fall back to additive model if multiplicative fails
                        decomposition = seasonal_decompose(monthly_ts, model='additive', period=12)
                    
                    # Create the figure with subplots
                    fig = make_subplots(
                        rows=4, 
                        cols=1,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                        vertical_spacing=0.1
                    )
                    
                    # Add traces for each component
                    fig.add_trace(
                        go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name="Observed"),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name="Trend"),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name="Seasonal"),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name="Residual"),
                        row=4, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=800,
                        title_text="Time Series Decomposition of Sales",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("#### Interpretation")
                    st.markdown("""
                    - **Trend**: Shows the long-term progression of sales over time
                    - **Seasonal**: Reveals regular patterns (e.g., holiday effects, quarterly business cycles)
                    - **Residual**: Represents the irregular component (random fluctuations)
                    """)
                except Exception as e:
                    st.warning(f"Could not perform time series decomposition: {e}")
            else:
                st.warning("Need at least 24 months of data for meaningful time series decomposition. Please select a wider date range.")
                
    # Add a download section at the bottom
    st.markdown("<h2 class='sub-header'>Download Data</h2>", unsafe_allow_html=True)
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = convert_df(filtered_df)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name='superstore_filtered_data.csv',
            mime='text/csv',
        )
    
    with col2:
        if 'future_data' in locals():
            forecast_csv = convert_df(future_data)
            st.download_button(
                label="Download Forecast Data as CSV",
                data=forecast_csv,
                file_name='superstore_forecast_data.csv',
                mime='text/csv',
            )

# Footer
st.markdown("---")
st.markdown("Superstore Sales Dashboard | Created with Streamlit")