import sqlite3
import os
from werkzeug.security import generate_password_hash
from config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Enable Write-Ahead Logging (WAL) for better concurrent performance on Windows
    try:
        conn.execute('PRAGMA journal_mode=WAL;')
    except sqlite3.Error:
        pass
    return conn

def init_db():
    # Ensure database folder exists
    Config.init_app()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Admin', 'Manager', 'Analyst')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. uploaded_datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            row_count INTEGER,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Pending'
        )
    ''')
    
    # 3. customer_segments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT UNIQUE NOT NULL,
            recency INTEGER,
            frequency INTEGER,
            monetary REAL,
            clv REAL,
            avg_basket REAL,
            segment TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 4. forecast_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecast_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_date TEXT NOT NULL,
            forecasted_value REAL NOT NULL,
            confidence_lower REAL,
            confidence_upper REAL,
            model_name TEXT,
            period_type TEXT CHECK(period_type IN ('Daily', 'Weekly', 'Monthly')),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 5. generated_reports table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_name TEXT NOT NULL,
            report_type TEXT NOT NULL CHECK(report_type IN ('PDF', 'Excel', 'CSV')),
            file_path TEXT NOT NULL,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 6. business_insights table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS business_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT,
            importance TEXT CHECK(importance IN ('High', 'Medium', 'Low')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Seed default users if table is empty
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        default_users = [
            ('admin', generate_password_hash('admin123'), 'Admin'),
            ('manager', generate_password_hash('manager123'), 'Manager'),
            ('analyst', generate_password_hash('analyst123'), 'Analyst')
        ]
        cursor.executemany(
            'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
            default_users
        )
    
    conn.commit()
    conn.close()

def query_db(query, args=(), one=False):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, args)
    rv = cursor.fetchall()
    conn.close()
    return (rv[0] if rv else None) if one else rv

def execute_db(query, args=()):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, args)
    conn.commit()
    conn.close()
