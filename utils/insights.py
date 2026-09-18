import pandas as pd
import numpy as np
from utils.database import execute_db, get_db_connection
import logging

logger = logging.getLogger(__name__)

def generate_and_save_insights(df_clean, rfm_df):
    """
    Analyze transactional and RFM customer data to generate dynamic,
    actionable business insights and save them to the database.
    """
    logger.info("Generating dynamic business insights...")
    insights = []
    
    if df_clean.empty or rfm_df.empty:
        return
        
    # --- 1. Revenue Growth / Performance Insights ---
    monthly_rev = df_clean.groupby('YearMonth')['Revenue'].sum().sort_index()
    if len(monthly_rev) >= 2:
        last_month = monthly_rev.index[-1]
        prev_month = monthly_rev.index[-2]
        last_val = monthly_rev.iloc[-1]
        prev_val = monthly_rev.iloc[-2]
        
        pct_change = ((last_val - prev_val) / prev_val) * 100
        direction = "increased" if pct_change > 0 else "decreased"
        importance = "High" if abs(pct_change) > 10 else "Medium"
        
        insights.append({
            'title': f"Month-over-Month Revenue Trend ({last_month})",
            'description': f"Monthly revenue {direction} by {abs(pct_change):.1f}% from ${prev_val:,.2f} in {prev_month} to ${last_val:,.2f} in {last_month}.",
            'category': 'Revenue Analysis',
            'importance': importance
        })
        
    # --- 2. Customer Segment Insights ---
    if 'segment' in rfm_df.columns:
        seg_summary = rfm_df.groupby('segment').agg({
            'CustomerID': 'count',
            'Monetary': 'sum'
        }).reset_index()
        
        total_rev = rfm_df['Monetary'].sum()
        total_cust = len(rfm_df)
        
        # Look for VIPs
        vip_row = seg_summary[seg_summary['segment'] == 'VIP Customers']
        if not vip_row.empty:
            vip_count = vip_row['CustomerID'].values[0]
            vip_rev = vip_row['Monetary'].values[0]
            vip_cust_pct = (vip_count / total_cust) * 100
            vip_rev_pct = (vip_rev / total_rev) * 100
            
            insights.append({
                'title': "VIP Customer Leverage",
                'description': f"VIP Customers represent just {vip_cust_pct:.1f}% of your customer base, yet they contribute {vip_rev_pct:.1f}% of your total monetary value (${vip_rev:,.2f}). Suggest implementing a high-touch customer success or rewards program.",
                'category': 'Customer Segmentation',
                'importance': 'High'
            })
            
        # Look for At Risk / Lost
        at_risk_row = seg_summary[seg_summary['segment'].isin(['At Risk Customers', 'Lost Customers'])]
        if not at_risk_row.empty:
            at_risk_count = at_risk_row['CustomerID'].sum()
            at_risk_pct = (at_risk_count / total_cust) * 100
            
            insights.append({
                'title': "Churn Risk Warning",
                'description': f"A total of {at_risk_count} customers ({at_risk_pct:.1f}% of your database) are classified as 'At Risk' or 'Lost'. A win-back marketing campaign with targeted email discounts is recommended to re-engage them.",
                'category': 'Customer Segmentation',
                'importance': 'High' if at_risk_pct > 25 else 'Medium'
            })
            
    # --- 3. Top Product Insights ---
    prod_rev = df_clean.groupby('Description')['Revenue'].sum().reset_index()
    prod_rev = prod_rev.sort_values(by='Revenue', ascending=False)
    
    if not prod_rev.empty:
        top_prod = prod_rev.iloc[0]
        top_prod_name = top_prod['Description']
        top_prod_rev = top_prod['Revenue']
        pct_of_total_rev = (top_prod_rev / df_clean['Revenue'].sum()) * 100
        
        insights.append({
            'title': f"Top Sales Driver: {top_prod_name}",
            'description': f"The product '{top_prod_name}' is your largest revenue source, generating ${top_prod_rev:,.2f} ({pct_of_total_rev:.2f}% of total sales). Ensure this item remains in stock and optimize its pricing page.",
            'category': 'Inventory Optimization',
            'importance': 'Medium'
        })
        
    # --- 4. Region Performance Insights ---
    country_rev = df_clean.groupby('Country')['Revenue'].sum().reset_index()
    country_rev = country_rev.sort_values(by='Revenue', ascending=False)
    
    if len(country_rev) >= 2:
        top_country = country_rev.iloc[0]
        second_country = country_rev.iloc[1]
        top_country_share = (top_country['Revenue'] / df_clean['Revenue'].sum()) * 100
        
        insights.append({
            'title': f"Geographic Sales Dominance: {top_country['Country']}",
            'description': f"{top_country['Country']} dominates your retail market with a {top_country_share:.1f}% share of revenue (${top_country['Revenue']:,.2f}). The next largest market is {second_country['Country']} representing ${second_country['Revenue']:,.2f}.",
            'category': 'Market Performance',
            'importance': 'High' if top_country_share > 75 else 'Medium'
        })
        
    # --- 5. Peak Transaction Hour Insights ---
    # Day-vs-hour analysis
    df_clean_insight = df_clean.copy()
    if 'Hour' not in df_clean_insight.columns:
        df_clean_insight['Hour'] = df_clean_insight['InvoiceDate'].dt.hour
        
    hour_counts = df_clean_insight.groupby('Hour')['InvoiceNo'].nunique()
    if not hour_counts.empty:
        peak_hour = hour_counts.idxmax()
        peak_count = hour_counts.max()
        
        insights.append({
            'title': f"Optimal Order Window ({peak_hour}:00)",
            'description': f"Transaction volume peaks between {peak_hour}:00 and {peak_hour+1}:00 with a total of {peak_count} unique orders. Schedule promotional emails, customer support staff, and server capacity during this hour.",
            'category': 'Marketing Operations',
            'importance': 'Low'
        })
        
    # --- 6. Low Value / Low Quantity Item Warnings ---
    prod_qty = df_clean.groupby(['Description', 'StockCode']).agg({
        'Quantity': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    
    low_performers = prod_qty[prod_qty['Revenue'] > 0].sort_values(by='Revenue', ascending=True).head(5)
    if not low_performers.empty:
        low_items_str = ", ".join([f"'{row['Description']}'" for _, row in low_performers.iterrows()])
        insights.append({
            'title': "Low-Velocity Stock Alert",
            'description': f"The following items generated the lowest revenues: {low_items_str}. Suggest bundling these with top selling items to liquidate slow-moving stock.",
            'category': 'Inventory Optimization',
            'importance': 'Low'
        })

    # Clear old insights and save new ones
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM business_insights')
    
    for ins in insights:
        cursor.execute(
            'INSERT INTO business_insights (title, description, category, importance) VALUES (?, ?, ?, ?)',
            (ins['title'], ins['description'], ins['category'], ins['importance'])
        )
    conn.commit()
    conn.close()
    
    logger.info("Saved %d generated insights to the SQLite database.", len(insights))
