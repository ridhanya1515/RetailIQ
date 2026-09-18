import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Adjust path to import root modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from utils.database import init_db, get_db_connection, query_db
from utils.preprocessing import download_uci_dataset, preprocess_dataset, compute_rfm_features
from utils.segmentation import perform_customer_segmentation, calculate_segment_distributions
from utils.forecasting import generate_demand_forecast
from utils.insights import generate_and_save_insights
from utils.reporting import generate_pdf_report, generate_excel_report
from app import app, User

class TestRetailIQPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()
        
        # Run startup config and ensure database exists
        Config.init_app()
        init_db()

    def test_1_database_initialization(self):
        """Test database connection and tables exist."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify tables list
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = [
            'users', 'uploaded_datasets', 'customer_segments', 
            'forecast_results', 'generated_reports', 'business_insights'
        ]
        for table in required_tables:
            self.assertIn(table, tables, f"Database table '{table}' is missing.")
            
        # Verify default users seeded
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        self.assertGreaterEqual(user_count, 3, "Seed users were not created successfully.")
        
        conn.close()
        print("SUCCESS: Database initialization verified.")

    def test_2_preprocessing_pipeline(self):
        """Test data cleaning and feature engineering."""
        # Ensure we have data
        raw_path = os.path.join(Config.RAW_DATA_DIR, "OnlineRetail.xlsx")
        
        # If it doesn't exist, this function will download or generate synthetic data
        download_uci_dataset()
        self.assertTrue(os.path.exists(raw_path), "Raw dataset file was not created/downloaded.")
        
        # Process data
        df_clean = preprocess_dataset(raw_path)
        self.assertFalse(df_clean.empty, "Cleaned dataset should not be empty.")
        
        required_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                            'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country', 
                            'Revenue', 'Year', 'Quarter', 'Month', 'Week', 'Day']
        for col in required_columns:
            self.assertIn(col, df_clean.columns, f"Cleaned DataFrame missing column '{col}'.")
            
        # Verify clean filtering (no negative quantities/unit prices)
        self.assertTrue((df_clean['Quantity'] > 0).all(), "Negative quantities found in cleaned data.")
        self.assertTrue((df_clean['UnitPrice'] > 0).all(), "Negative unit prices found in cleaned data.")
        
        # Test RFM features computation
        rfm = compute_rfm_features(df_clean)
        self.assertFalse(rfm.empty, "RFM features DataFrame should not be empty.")
        self.assertIn('Recency', rfm.columns)
        self.assertIn('Frequency', rfm.columns)
        self.assertIn('Monetary', rfm.columns)
        self.assertIn('CLV', rfm.columns)
        self.assertIn('AverageBasketSize', rfm.columns)
        
        print("SUCCESS: Preprocessing and feature engineering pipelines verified.")

    def test_3_customer_segmentation(self):
        """Test KMeans customer segmentation clustering."""
        processed_path = os.path.join(Config.PROCESSED_DATA_DIR, "cleaned_transactions.csv")
        df_clean = pd.read_csv(processed_path, dtype={'InvoiceNo': str, 'StockCode': str, 'CustomerID': str}, low_memory=False)
        
        rfm = compute_rfm_features(df_clean)
        rfm_segmented, optimal_k, stats = perform_customer_segmentation(rfm)
        
        self.assertIn('segment', rfm_segmented.columns)
        self.assertGreaterEqual(optimal_k, 3, "Optimal cluster count should be at least 3.")
        self.assertIn('silhouette_score', stats)
        
        dist = calculate_segment_distributions(rfm_segmented)
        self.assertFalse(dist.empty)
        self.assertIn('customer_count', dist.columns)
        self.assertIn('revenue_share_pct', dist.columns)
        
        print(f"SUCCESS: KMeans clustering verified. Optimal K determined: {optimal_k}")

    def test_4_demand_forecasting(self):
        """Test ARIMA/SARIMAX demand forecasting models."""
        processed_path = os.path.join(Config.PROCESSED_DATA_DIR, "cleaned_transactions.csv")
        df_clean = pd.read_csv(processed_path, dtype={'InvoiceNo': str, 'StockCode': str, 'CustomerID': str}, low_memory=False)
        
        # Run forecast
        results = generate_demand_forecast(df_clean, period_type='Daily', forecast_days=10)
        
        self.assertIn('best_model', results)
        self.assertIn('forecast', results)
        self.assertIn('comparison', results)
        
        forecast_df = results['forecast']
        self.assertEqual(len(forecast_df), 10, "Forecast length does not match specified horizon.")
        self.assertIn('Forecast', forecast_df.columns)
        self.assertIn('Lower_CI', forecast_df.columns)
        self.assertIn('Upper_CI', forecast_df.columns)
        
        comparison = results['comparison']
        self.assertGreaterEqual(len(comparison), 2, "Comparison table should compare at least 2 models.")
        
        print(f"SUCCESS: Demand forecasting verified. Best model auto-selected: {results['best_model']}")

    def test_5_insights_generation(self):
        """Test rule-based insights generation."""
        processed_path = os.path.join(Config.PROCESSED_DATA_DIR, "cleaned_transactions.csv")
        df_clean = pd.read_csv(processed_path, parse_dates=['InvoiceDate'], dtype={'InvoiceNo': str, 'StockCode': str, 'CustomerID': str}, low_memory=False)
        
        rfm = compute_rfm_features(df_clean)
        rfm_segmented, _, _ = perform_customer_segmentation(rfm)
        
        # Save insights
        generate_and_save_insights(df_clean, rfm_segmented)
        
        # Read from db
        insights = query_db('SELECT * FROM business_insights')
        self.assertGreater(len(insights), 0, "No business insights were written to the database.")
        
        print(f"SUCCESS: Business insights generation verified. Created {len(insights)} insights.")

    def test_6_reporting_module(self):
        """Test ReportLab PDF and Excel report compilation."""
        # Setup mock report data
        kpis = {
            'total_revenue': 150000.0,
            'total_orders': 450,
            'total_customers': 380,
            'avg_order_value': 333.33,
            'avg_revenue_per_customer': 394.74,
            'rev_growth_mom': 12.4,
            'cust_growth_mom': 5.2
        }
        
        segment_dist = pd.DataFrame([
            {'segment': 'VIP Customers', 'customer_count': 10, 'total_revenue': 50000.0, 
             'average_spend': 5000.0, 'average_recency': 5.0, 'average_frequency': 15.0,
             'customer_share_pct': 2.6, 'revenue_share_pct': 33.3},
            {'segment': 'Regular Customers', 'customer_count': 90, 'total_revenue': 100000.0, 
             'average_spend': 1111.11, 'average_recency': 45.0, 'average_frequency': 4.0,
             'customer_share_pct': 23.6, 'revenue_share_pct': 66.7}
        ])
        
        forecast_df = pd.DataFrame([
            {'Date': '2026-06-18', 'Forecast': 5000.0, 'Lower_CI': 4500.0, 'Upper_CI': 5500.0},
            {'Date': '2026-06-19', 'Forecast': 5200.0, 'Lower_CI': 4700.0, 'Upper_CI': 5700.0}
        ])
        
        insights = [
            {'title': 'VIP Sales Growth', 'description': 'VIP revenue grew by 15% this quarter.', 
             'category': 'Customer Segmentation', 'importance': 'High'}
        ]
        
        # PDF Exporter
        pdf_stream = generate_pdf_report(kpis, segment_dist, forecast_df, insights)
        self.assertIsNotNone(pdf_stream, "PDF generation stream was null.")
        self.assertGreater(len(pdf_stream.getvalue()), 1000, "PDF file size is too small; check compilation.")
        
        # Excel Exporter
        excel_stream = generate_excel_report(kpis, segment_dist, forecast_df, insights)
        self.assertIsNotNone(excel_stream, "Excel generation stream was null.")
        self.assertGreater(len(excel_stream.getvalue()), 1000, "Excel file size is too small.")
        
        print("SUCCESS: Reporting system (PDF & Excel engines) verified.")

    def test_7_flask_routes(self):
        """Test Flask application endpoints return HTTP OK or redirects."""
        # 1. Login page accessible
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200, "Login route should be directly accessible.")
        
        # 2. Register page accessible
        response = self.client.get('/register')
        self.assertEqual(response.status_code, 200, "Register route should be accessible.")
        
        # 3. Protected pages redirect to login
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 302, "Dashboard should redirect unauthenticated request.")
        self.assertIn('/login', response.headers.get('Location', ''), "Redirect target should be login.")
        
        response = self.client.get('/segmentation')
        self.assertEqual(response.status_code, 302, "Segmentation page should redirect unauthenticated request.")
        
        print("SUCCESS: Flask application routing security verified.")

if __name__ == '__main__':
    unittest.main()
