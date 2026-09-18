import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json
import logging

logger = logging.getLogger(__name__)

# Set a beautiful modern dark template as default for Plotly charts
pio.templates.default = "plotly_dark"

def calculate_kpis(df):
    """Calculate core KPI metrics from clean transactions DataFrame."""
    if df.empty:
        return {
            'total_revenue': 0.0,
            'total_orders': 0,
            'total_customers': 0,
            'avg_order_value': 0.0,
            'avg_revenue_per_customer': 0.0,
            'rev_growth_mom': 0.0,
            'cust_growth_mom': 0.0
        }
    
    total_revenue = float(df['Revenue'].sum())
    total_orders = int(df['InvoiceNo'].nunique())
    
    # Filter customers with non-null ID
    valid_customers = df[df['CustomerID'].notna()]
    total_customers = int(valid_customers['CustomerID'].nunique())
    
    avg_order_value = float(total_revenue / total_orders) if total_orders > 0 else 0.0
    avg_rev_per_customer = float(total_revenue / total_customers) if total_customers > 0 else 0.0
    
    # Compute Month-over-Month Revenue Growth
    # Group by YearMonth
    monthly_summary = df.groupby('YearMonth').agg({
        'Revenue': 'sum',
        'CustomerID': 'nunique'
    }).sort_index()
    
    rev_growth_mom = 0.0
    cust_growth_mom = 0.0
    
    if len(monthly_summary) >= 2:
        prev_month_rev = monthly_summary['Revenue'].iloc[-2]
        curr_month_rev = monthly_summary['Revenue'].iloc[-1]
        if prev_month_rev > 0:
            rev_growth_mom = float(((curr_month_rev - prev_month_rev) / prev_month_rev) * 100)
            
        prev_month_cust = monthly_summary['CustomerID'].iloc[-2]
        curr_month_cust = monthly_summary['CustomerID'].iloc[-1]
        if prev_month_cust > 0:
            cust_growth_mom = float(((curr_month_cust - prev_month_cust) / prev_month_cust) * 100)
            
    return {
        'total_revenue': round(total_revenue, 2),
        'total_orders': total_orders,
        'total_customers': total_customers,
        'avg_order_value': round(avg_order_value, 2),
        'avg_revenue_per_customer': round(avg_rev_per_customer, 2),
        'rev_growth_mom': round(rev_growth_mom, 2),
        'cust_growth_mom': round(cust_growth_mom, 2)
    }

def get_top_bottom_products(df, n=10):
    """Retrieve top N and bottom N products by Revenue."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    prod_summary = df.groupby(['StockCode', 'Description']).agg({
        'Quantity': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    top_products = prod_summary.sort_values(by='Revenue', ascending=False).head(n)
    bottom_products = prod_summary[prod_summary['Revenue'] > 0].sort_values(by='Revenue', ascending=True).head(n)
    
    return top_products, bottom_products

def get_plotly_json(fig):
    """Helper to convert Plotly figure to JSON structure."""
    # Custom layout overrides for stunning UI integration
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_family="Outfit, Inter, sans-serif",
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=16, color='#f8f9fa', family="Outfit"),
        hoverlabel=dict(bgcolor='#1f2937', font_size=13, font_family="Inter")
    )
    return pio.to_json(fig)


# ==========================================
# CHART GENERATORS
# ==========================================

def build_revenue_trend_chart(df):
    """Generate interactive Line chart for monthly revenue trends."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return get_plotly_json(fig)
        
    monthly_rev = df.groupby('YearMonth')['Revenue'].sum().reset_index()
    monthly_rev = monthly_rev.sort_values('YearMonth')
    
    fig = px.line(
        monthly_rev, 
        x='YearMonth', 
        y='Revenue',
        title='Monthly Revenue Growth & Performance',
        labels={'YearMonth': 'Month', 'Revenue': 'Revenue ($)'},
        markers=True
    )
    fig.update_traces(
        line=dict(color='#3b82f6', width=3.5),
        marker=dict(size=8, color='#10b981', symbol='circle')
    )
    return get_plotly_json(fig)

def build_revenue_by_country_chart(df):
    """Generate Treemap for revenue share by country."""
    if df.empty:
        fig = go.Figure()
        return get_plotly_json(fig)
        
    country_rev = df.groupby('Country')['Revenue'].sum().reset_index()
    # Filter for representation
    country_rev = country_rev.sort_values(by='Revenue', ascending=False)
    
    fig = px.treemap(
        country_rev,
        path=['Country'],
        values='Revenue',
        title='Revenue Contribution by Country',
        color='Revenue',
        color_continuous_scale='Viridis'
    )
    return get_plotly_json(fig)

def build_top_products_chart(df, n=10):
    """Generate Horizontal Bar Chart for top products."""
    if df.empty:
        fig = go.Figure()
        return get_plotly_json(fig)
        
    top_p, _ = get_top_bottom_products(df, n)
    
    # Take first 30 chars of description to keep clean
    top_p['Product'] = top_p['Description'].str.slice(0, 30)
    top_p = top_p.sort_values(by='Revenue', ascending=True)
    
    fig = px.bar(
        top_p,
        x='Revenue',
        y='Product',
        orientation='h',
        title=f'Top {n} Products by Revenue',
        labels={'Revenue': 'Revenue ($)', 'Product': 'Product Description'},
        color='Revenue',
        color_continuous_scale='Tealgrn'
    )
    return get_plotly_json(fig)

def build_quarterly_revenue_chart(df):
    """Generate Bar chart for Quarterly Revenue."""
    if df.empty:
        fig = go.Figure()
        return get_plotly_json(fig)
        
    df['QuarterLabel'] = df['Year'].astype(str) + " Q" + df['Quarter'].astype(str)
    q_rev = df.groupby('QuarterLabel')['Revenue'].sum().reset_index().sort_values('QuarterLabel')
    
    fig = px.bar(
        q_rev,
        x='QuarterLabel',
        y='Revenue',
        title='Quarterly Revenue Performance',
        labels={'QuarterLabel': 'Quarter', 'Revenue': 'Revenue ($)'},
        color='Revenue',
        color_continuous_scale='Plotly3'
    )
    return get_plotly_json(fig)

def build_day_hour_heatmap(df):
    """Generate Heatmap of transactions by Day of Week vs Hour."""
    if df.empty:
        fig = go.Figure()
        return get_plotly_json(fig)
        
    # Standardize Hour feature
    df_heatmap = df.copy()
    if 'Hour' not in df_heatmap.columns:
        df_heatmap['Hour'] = df_heatmap['InvoiceDate'].dt.hour
        
    # Day Names
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_heatmap['DayName'] = df_heatmap['DayOfWeek'].apply(lambda x: days[x] if 0 <= x < 7 else 'Unknown')
    
    # Filter hours to business hours (e.g. 6 to 20)
    df_heatmap = df_heatmap[(df_heatmap['Hour'] >= 6) & (df_heatmap['Hour'] <= 20)]
    
    pivot_table = df_heatmap.pivot_table(
        index='DayName', 
        columns='Hour', 
        values='InvoiceNo', 
        aggfunc='nunique'
    ).reindex(days).fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Magma',
        hoverongaps=False,
        hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Orders: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Order Velocity Heatmap (Day vs Hour)',
        xaxis_title='Hour of Day (24h format)',
        yaxis_title='Day of Week'
    )
    return get_plotly_json(fig)

def build_customer_scatter_plot(rfm):
    """Generate 3D Scatter plot or 2D Scatter plot for Recency, Frequency, Monetary."""
    if rfm.empty:
        fig = go.Figure()
        return get_plotly_json(fig)
        
    # We clip values at 99th percentile for clean visualization (removing outliers)
    r_limit = rfm['Recency'].quantile(0.99)
    f_limit = rfm['Frequency'].quantile(0.99)
    m_limit = rfm['Monetary'].quantile(0.99)
    
    rfm_filtered = rfm[
        (rfm['Recency'] <= r_limit) &
        (rfm['Frequency'] <= f_limit) &
        (rfm['Monetary'] <= m_limit)
    ].copy()
    
    color_col = 'segment' if 'segment' in rfm.columns else 'Monetary'
    hover_cols = [c for c in ['CustomerID', 'clv', 'CLV', 'avg_basket', 'AverageBasketSize'] if c in rfm_filtered.columns]
    
    fig = px.scatter(
        rfm_filtered,
        x='Recency',
        y='Frequency',
        size='Monetary',
        color=color_col,
        hover_data=hover_cols,
        title='Customer Transaction Profile (Recency vs Frequency vs Monetary)',
        labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (Orders)', 'segment': 'Segment'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='white')))
    return get_plotly_json(fig)
