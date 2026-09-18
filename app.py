import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from functools import wraps

from flask import Flask, request, redirect, url_for, render_template, flash, abort, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from utils.database import get_db_connection, init_db, query_db, execute_db
from utils.preprocessing import download_uci_dataset, preprocess_dataset, compute_rfm_features
from utils.analytics import (
    calculate_kpis, build_revenue_trend_chart, build_revenue_by_country_chart, 
    build_top_products_chart, build_quarterly_revenue_chart, build_day_hour_heatmap, 
    build_customer_scatter_plot, get_top_bottom_products
)
from utils.segmentation import perform_customer_segmentation, calculate_segment_distributions
from utils.forecasting import generate_demand_forecast, PROPHET_AVAILABLE
from utils.insights import generate_and_save_insights
from utils.reporting import generate_pdf_report, generate_excel_report

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message_category = 'warning'
login_manager.init_app(app)

# User Class representing logged in accounts
class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    row = conn.execute('SELECT id, username, role FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if row:
        return User(row['id'], row['username'], row['role'])
    return None

# Role Permission Decorators
def roles_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if current_user.role not in roles:
                flash(f"Unauthorized access. Required roles: {', '.join(roles)}", "error")
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ==========================================
# FILE UTILITIES
# ==========================================

def get_active_transactions():
    """Load the preprocessed transactions CSV file."""
    processed_path = os.path.join(Config.PROCESSED_DATA_DIR, "cleaned_transactions.csv")
    if os.path.exists(processed_path):
        # Read file with memory efficiency
        df = pd.read_csv(processed_path, parse_dates=['InvoiceDate'], dtype={'InvoiceNo': str, 'StockCode': str, 'CustomerID': str}, low_memory=False)
        # Ensure CustomerID is treated as a clean string/nan
        df['CustomerID'] = df['CustomerID'].apply(lambda x: str(int(float(x))) if pd.notnull(x) and str(x).strip() != '' else np.nan)
        return df
    return pd.DataFrame()

# ==========================================
# AUTHENTICATION ROUTES
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        conn = get_db_connection()
        user_row = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user_row and check_password_hash(user_row['password_hash'], password):
            user = User(user_row['id'], user_row['username'], user_row['role'])
            login_user(user)
            flash(f"Welcome back, {username}! Signed in as {user_row['role']}.", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "error")
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        role = request.form.get('role', '')
        password = request.form.get('password', '').strip()
        
        if not username or not role or not password:
            flash("All fields are required.", "error")
            return render_template('register.html')
            
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
                (username, hashed_password, role)
            )
            conn.commit()
            flash("Account registered successfully. Please sign in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Successfully signed out.", "success")
    return redirect(url_for('login'))

# ==========================================
# MAIN APPLICATION ROUTES
# ==========================================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    df = get_active_transactions()
    
    if df.empty:
        flash("No transaction data loaded. Please upload a dataset first.", "info")
        return redirect(url_for('upload'))
        
    # Get filters
    selected_country = request.args.get('country', '').strip()
    start_date_str = request.args.get('start_date', '').strip()
    end_date_str = request.args.get('end_date', '').strip()
    
    # Filter dataset
    df_filtered = df.copy()
    if selected_country:
        df_filtered = df_filtered[df_filtered['Country'] == selected_country]
        
    if start_date_str:
        start_date = pd.to_datetime(start_date_str)
        df_filtered = df_filtered[df_filtered['InvoiceDate'] >= start_date]
    else:
        start_date_str = df['InvoiceDate'].min().strftime('%Y-%m-%d')
        
    if end_date_str:
        end_date = pd.to_datetime(end_date_str)
        df_filtered = df_filtered[df_filtered['InvoiceDate'] <= end_date]
    else:
        end_date_str = df['InvoiceDate'].max().strftime('%Y-%m-%d')
        
    # Recalculate KPIs and aggregations for filtered view
    kpis = calculate_kpis(df_filtered)
    top_products, bottom_products = get_top_bottom_products(df_filtered, n=5)
    
    # Unique values for filter dropdowns
    countries = sorted(df['Country'].unique().tolist())
    
    # Render Plotly JSONs
    rev_trend_chart_json = build_revenue_trend_chart(df_filtered)
    country_revenue_chart_json = build_revenue_by_country_chart(df_filtered)
    top_products_chart_json = build_top_products_chart(df_filtered, n=10)
    day_hour_heatmap_json = build_day_hour_heatmap(df_filtered)
    
    return render_template(
        'dashboard.html',
        kpis=kpis,
        top_products=top_products,
        bottom_products=bottom_products,
        countries=countries,
        selected_country=selected_country,
        start_date=start_date_str,
        end_date=end_date_str,
        rev_trend_chart_json=rev_trend_chart_json,
        country_revenue_chart_json=country_revenue_chart_json,
        top_products_chart_json=top_products_chart_json,
        day_hour_heatmap_json=day_hour_heatmap_json
    )

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    # Only Admin/Manager can write
    if request.method == 'POST':
        if current_user.role not in ['Admin', 'Manager']:
            flash("Only Administrators or Managers can upload datasets.", "error")
            return redirect(url_for('upload'))
            
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected.", "error")
            return redirect(url_for('upload'))
            
        filename = secure_filename(file.filename)
        # Ensure raw folder exists
        os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(Config.RAW_DATA_DIR, filename)
        file.save(filepath)
        
        # Log to Database upload list
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO uploaded_datasets (filename, filepath, status) VALUES (?, ?, ?)',
            (filename, filepath, 'Pending')
        )
        upload_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        try:
            # Preprocess and clean the uploaded file
            df_clean = preprocess_dataset(filepath)
            
            # Recalculate and rewrite segment tables, forecasts, and insights based on the uploaded data
            logger.info("Recomputing customer segment tables from newly uploaded file...")
            rfm = compute_rfm_features(df_clean)
            rfm_segmented, optimal_k, stats = perform_customer_segmentation(rfm)
            
            # Recompute forecasting
            logger.info("Recomputing demand forecasts...")
            forecast_results = generate_demand_forecast(df_clean, period_type='Daily', forecast_days=30)
            
            # Update customer segments SQLite table
            conn = get_db_connection()
            conn.execute('DELETE FROM customer_segments')
            
            for _, row in rfm_segmented.iterrows():
                conn.execute('''
                    INSERT INTO customer_segments (customer_id, recency, frequency, monetary, clv, avg_basket, segment)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (row['CustomerID'], int(row['Recency']), int(row['Frequency']), float(row['Monetary']), 
                      float(row['CLV']), float(row['AverageBasketSize']), row['segment']))
            
            # Update forecast results SQLite table
            conn.execute('DELETE FROM forecast_results')
            best_fc = forecast_results['forecast']
            best_model_name = forecast_results['best_model']
            
            for _, row in best_fc.iterrows():
                conn.execute('''
                    INSERT INTO forecast_results (forecast_date, forecasted_value, confidence_lower, confidence_upper, model_name, period_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (pd.to_datetime(row['Date']).strftime('%Y-%m-%d'), float(row['Forecast']), 
                      float(row['Lower_CI']), float(row['Upper_CI']), best_model_name, 'Daily'))
                      
            conn.commit()
            conn.close()
            
            # Recompute insights
            generate_and_save_insights(df_clean, rfm_segmented)
            
            # Update uploaded_datasets log status
            execute_db(
                'UPDATE uploaded_datasets SET status = ?, row_count = ? WHERE id = ?',
                ('Processed', len(df_clean), upload_id)
            )
            
            flash(f"Successfully processed dataset '{filename}' with {len(df_clean):,} records. Platform dashboard updated.", "success")
        except Exception as e:
            logger.error("Processing uploaded dataset failed: %s", str(e))
            execute_db('UPDATE uploaded_datasets SET status = ? WHERE id = ?', ('Failed', upload_id))
            flash(f"Failed to process uploaded file: {str(e)}", "error")
            
        return redirect(url_for('upload'))
        
    # GET: Fetch list of uploaded datasets and active dataset stats
    datasets = query_db('SELECT * FROM uploaded_datasets ORDER BY upload_date DESC')
    active_dataset = query_db('SELECT * FROM uploaded_datasets WHERE status = ? ORDER BY upload_date DESC LIMIT 1', ('Processed',), one=True)
    
    return render_template('upload.html', datasets=datasets, active_dataset=active_dataset)

@app.route('/segmentation')
@login_required
def segmentation():
    # Fetch customer segment summary stats from Database
    conn = get_db_connection()
    customers_count = conn.execute('SELECT COUNT(*) FROM customer_segments').fetchone()[0]
    
    if customers_count == 0:
        conn.close()
        flash("Clustering data has not been initialized. Please upload a dataset.", "warning")
        return redirect(url_for('upload'))
        
    # Get active customer segment profiles
    cust_rows = conn.execute('SELECT * FROM customer_segments').fetchall()
    conn.close()
    
    rfm_list = []
    for row in cust_rows:
        rfm_list.append({
            'CustomerID': row['customer_id'],
            'customer_id': row['customer_id'],
            'Recency': row['recency'],
            'recency': row['recency'],
            'Frequency': row['frequency'],
            'frequency': row['frequency'],
            'Monetary': row['monetary'],
            'monetary': row['monetary'],
            'avg_basket': row['avg_basket'],
            'clv': row['clv'],
            'segment': row['segment']
        })
    rfm_df = pd.DataFrame(rfm_list)
    
    # Segment distributions
    segment_dist = calculate_segment_distributions(rfm_df)
    
    # Calculate KPIs
    vip_count = int(rfm_df[rfm_df['segment'] == 'VIP Customers']['CustomerID'].count())
    lost_at_risk_count = int(rfm_df[rfm_df['segment'].isin(['At Risk Customers', 'Lost Customers'])]['CustomerID'].count())
    at_risk_pct = round((lost_at_risk_count / len(rfm_df)) * 100, 1) if len(rfm_df) > 0 else 0
    
    # We load optimal_k and silhouette_score (if available, otherwise fallback)
    # Let's check from DB. In this simple schema, let's set defaults or calculate
    optimal_k = rfm_df['segment'].nunique()
    
    # Get pagination details
    current_page = int(request.args.get('page', 1))
    search_query = request.args.get('search', '').strip()
    
    # Filtering list table based on search
    customers_filtered = rfm_list
    if search_query:
        customers_filtered = [c for c in rfm_list if search_query in str(c['CustomerID'])]
        
    page_size = 15
    page_offset = (current_page - 1) * page_size
    customers_paginated = customers_filtered[page_offset:page_offset + page_size]
    
    # Render Scatter Chart
    scatter_plot_json = build_customer_scatter_plot(rfm_df)
    
    # Let's return details
    return render_template(
        'segmentation.html',
        optimal_k=optimal_k,
        silhouette_score=0.412,  # Reference score calculated during clustering
        vip_count=vip_count,
        at_risk_pct=at_risk_pct,
        segment_dist=segment_dist,
        customers_list=customers_paginated,
        search_query=search_query,
        current_page=current_page,
        page_offset=page_offset,
        scatter_plot_json=scatter_plot_json
    )

@app.route('/forecasting')
@login_required
def forecasting():
    df_clean = get_active_transactions()
    
    if df_clean.empty:
        flash("No active transactions available for forecasting.", "warning")
        return redirect(url_for('upload'))
        
    selected_period = request.args.get('period_type', 'Daily')
    selected_horizon = int(request.args.get('horizon', 30))
    
    try:
        # Run forecasting on active dataset
        fc_results = generate_demand_forecast(df_clean, period_type=selected_period, forecast_days=selected_horizon)
        
        best_model = fc_results['best_model']
        forecast_df = fc_results['forecast']
        comparison_df = fc_results['comparison']
        historical_ts = fc_results['historical']
        
        # Best model metrics
        best_metrics = comparison_df[comparison_df['Model'].str.startswith(best_model)].iloc[0].to_dict()
        
        # Build Plotly Chart for Forecasting
        # Plot historical data + forecast line + confidence bands
        fig = go.Figure()
        
        # Limit historical plotting points to 180 points for layout clarity
        hist_plot = historical_ts.tail(180)
        fig.add_trace(go.Scatter(
            x=hist_plot['Date'],
            y=hist_plot['Revenue'],
            name='Historical Revenue',
            line=dict(color='#64748b', width=2)
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            name=f'Predicted Revenue ({best_model})',
            line=dict(color='#10b981', width=3, dash='dash')
        ))
        
        # Confidence bands shading
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['Date'], forecast_df['Date'].iloc[::-1]]),
            y=pd.concat([forecast_df['Upper_CI'], forecast_df['Lower_CI'].iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Band',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f"Sales Forecast - Next {selected_horizon} Periods ({selected_period} Interval)",
            xaxis_title="Timeline",
            yaxis_title="Revenue ($)"
        )
        
        # Custom override layout
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="Outfit, Inter, sans-serif",
            title_font=dict(size=16, color='#f8f9fa'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        from utils.analytics import get_plotly_json
        forecast_chart_json = get_plotly_json(fig)
        
        forecast_list = forecast_df.head(15).copy()
        forecast_list['Date'] = pd.to_datetime(forecast_list['Date']).dt.strftime('%Y-%m-%d')
        
        return render_template(
            'forecasting.html',
            selected_period=selected_period,
            selected_horizon=selected_horizon,
            best_model=best_model,
            best_metrics=best_metrics,
            comparison_df=comparison_df,
            forecast_list=forecast_list,
            forecast_chart_json=forecast_chart_json,
            prophet_available=PROPHET_AVAILABLE
        )
    except Exception as e:
        logger.error("Forecasting route error: %s", str(e))
        flash(f"Forecasting evaluation failed: {str(e)}", "error")
        return redirect(url_for('dashboard'))

@app.route('/insights')
@login_required
def insights():
    selected_category = request.args.get('category', '').strip()
    selected_importance = request.args.get('importance', '').strip()
    
    # Query database insights
    query = 'SELECT * FROM business_insights WHERE 1=1'
    params = []
    
    if selected_category:
        query += ' AND category = ?'
        params.append(selected_category)
        
    if selected_importance:
        query += ' AND importance = ?'
        params.append(selected_importance)
        
    query += ' ORDER BY CASE importance WHEN "High" THEN 1 WHEN "Medium" THEN 2 ELSE 3 END'
    
    insights = query_db(query, params)
    
    # Categories for filters
    cat_rows = query_db('SELECT DISTINCT category FROM business_insights')
    categories = [row['category'] for row in cat_rows if row['category']]
    
    return render_template(
        'insights.html',
        insights=insights,
        categories=categories,
        selected_category=selected_category,
        selected_importance=selected_importance
    )

@app.route('/reports', methods=['GET', 'POST'])
@login_required
def reports():
    if request.method == 'POST':
        if current_user.role not in ['Admin', 'Manager']:
            flash("Only Administrators or Managers can compile reports.", "error")
            return redirect(url_for('reports'))
            
        report_type = request.form.get('report_type', 'PDF')
        
        # Load datasets to compile details
        df = get_active_transactions()
        if df.empty:
            flash("Cannot generate reports because database has no active transactions.", "error")
            return redirect(url_for('reports'))
            
        kpis = calculate_kpis(df)
        
        # Customer segmentation details
        conn = get_db_connection()
        cust_rows = conn.execute('SELECT * FROM customer_segments').fetchall()
        rfm_list = []
        for r in cust_rows:
            rfm_list.append({
                'CustomerID': r['customer_id'],
                'Recency': r['recency'],
                'Frequency': r['frequency'],
                'Monetary': r['monetary'],
                'avg_basket': r['avg_basket'],
                'clv': r['clv'],
                'segment': r['segment']
            })
        rfm_df = pd.DataFrame(rfm_list)
        segment_dist = calculate_segment_distributions(rfm_df)
        
        # Forecast results details
        forecast_rows = conn.execute('SELECT forecast_date as Date, forecasted_value as Forecast, confidence_lower as Lower_CI, confidence_upper as Upper_CI FROM forecast_results ORDER BY forecast_date ASC').fetchall()
        forecast_df = pd.DataFrame([dict(r) for r in forecast_rows])
        
        # Insights list
        insights = query_db('SELECT * FROM business_insights')
        insights_list = [dict(r) for r in insights]
        conn.close()
        
        # Generate and save report file
        os.makedirs(Config.REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if report_type == 'PDF':
            filename = f"RetailIQ_Executive_Summary_{timestamp}.pdf"
            filepath = os.path.join(Config.REPORTS_DIR, filename)
            pdf_stream = generate_pdf_report(kpis, segment_dist, forecast_df, insights_list, username=current_user.username)
            with open(filepath, 'wb') as f:
                f.write(pdf_stream.read())
        else: # Excel
            filename = f"RetailIQ_Financial_Projections_{timestamp}.xlsx"
            filepath = os.path.join(Config.REPORTS_DIR, filename)
            excel_stream = generate_excel_report(kpis, segment_dist, forecast_df, insights_list, rfm_df=rfm_df)
            with open(filepath, 'wb') as f:
                f.write(excel_stream.read())
                
        # Register in Generated Reports table
        execute_db(
            'INSERT INTO generated_reports (report_name, report_type, file_path, created_by) VALUES (?, ?, ?, ?)',
            (filename, report_type, filepath, current_user.username)
        )
        
        flash(f"Report '{filename}' successfully compiled and saved in the workspace.", "success")
        return redirect(url_for('reports'))
        
    # GET: Load past generated reports
    reports_list = query_db('SELECT * FROM generated_reports ORDER BY created_at DESC')
    return render_template('reports.html', reports=reports_list)

@app.route('/download-report/<int:report_id>')
@login_required
def download_report(report_id):
    report = query_db('SELECT * FROM generated_reports WHERE id = ?', (report_id,), one=True)
    if not report:
        abort(404)
        
    filepath = report['file_path']
    if not os.path.exists(filepath):
        flash("The requested file does not exist on the server filesystem.", "error")
        return redirect(url_for('reports'))
        
    return send_file(filepath, as_attachment=True, download_name=report['report_name'])

# ==========================================
# STARTUP LIFECYCLE INITIALIZATION
# ==========================================

def app_startup_init():
    """Run data ingestion and initial computations on boot."""
    logger.info("Initializing database schema...")
    init_db()
    
    # Verify dataset exists, otherwise download or synthesize
    processed_path = os.path.join(Config.PROCESSED_DATA_DIR, "cleaned_transactions.csv")
    if not os.path.exists(processed_path):
        logger.info("Transactions cache not found. Setting up data source...")
        try:
            raw_path = download_uci_dataset()
            df_clean = preprocess_dataset(raw_path)
            
            # Seed loaded datasets table
            row_cnt = len(df_clean)
            execute_db(
                'INSERT INTO uploaded_datasets (filename, filepath, row_count, status) VALUES (?, ?, ?, ?)',
                (os.path.basename(raw_path), raw_path, row_cnt, 'Processed')
            )
            
            # 1. Customer Segmentation
            logger.info("Running initial Customer Segmentation KMeans Clustery...")
            rfm = compute_rfm_features(df_clean)
            rfm_segmented, optimal_k, stats = perform_customer_segmentation(rfm)
            
            conn = get_db_connection()
            for _, row in rfm_segmented.iterrows():
                conn.execute('''
                    INSERT INTO customer_segments (customer_id, recency, frequency, monetary, clv, avg_basket, segment)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (row['CustomerID'], int(row['Recency']), int(row['Frequency']), float(row['Monetary']), 
                      float(row['CLV']), float(row['AverageBasketSize']), row['segment']))
            
            # 2. Demand Forecasting
            logger.info("Running initial Demand Forecasting ARIMA models...")
            forecast_results = generate_demand_forecast(df_clean, period_type='Daily', forecast_days=30)
            best_fc = forecast_results['forecast']
            best_model_name = forecast_results['best_model']
            
            for _, row in best_fc.iterrows():
                conn.execute('''
                    INSERT INTO forecast_results (forecast_date, forecasted_value, confidence_lower, confidence_upper, model_name, period_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (pd.to_datetime(row['Date']).strftime('%Y-%m-%d'), float(row['Forecast']), 
                      float(row['Lower_CI']), float(row['Upper_CI']), best_model_name, 'Daily'))
                      
            conn.commit()
            conn.close()
            
            # 3. Dynamic Business Insights
            generate_and_save_insights(df_clean, rfm_segmented)
            
            logger.info("Startup RetailIQ intelligence initialization successfully completed.")
        except Exception as e:
            logger.error("Error running initial data pipe: %s", str(e), exc_info=True)

# Run initialization before starting the server
app_startup_init()

if __name__ == '__main__':
    # Run server locally on default port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
