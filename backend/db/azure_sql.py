"""
azure_sql.py
────────────
Azure SQL Database connection, table creation, and helper functions.
All credentials are loaded from environment variables via .env

Tables created:
  - sales_transactions   raw uploaded transactions
  - daily_forecasts      XGBoost prediction outputs
  - anomaly_records      Isolation Forest flagged records
"""

import logging
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ── Try to import pyodbc (optional - graceful fallback if not installed) ──────
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False
    log.warning("pyodbc not installed. Azure SQL features will be disabled.")

# ── Connection string from .env ───────────────────────────────────────────────
AZURE_SQL_CONN = os.getenv("AZURE_SQL_CONNECTION_STRING", "")

# ── SQL: table creation ───────────────────────────────────────────────────────
CREATE_TRANSACTIONS_TABLE = """
IF NOT EXISTS (
    SELECT * FROM sysobjects WHERE name='sales_transactions' AND xtype='U'
)
CREATE TABLE sales_transactions (
    id                INT IDENTITY(1,1) PRIMARY KEY,
    transaction_id    NVARCHAR(50),
    date              DATE,
    customer_id       NVARCHAR(50),
    gender            NVARCHAR(10),
    age               INT,
    product_category  NVARCHAR(50),
    quantity          INT,
    price_per_unit    FLOAT,
    total_amount      FLOAT,
    uploaded_at       DATETIME DEFAULT GETDATE()
);
"""

CREATE_FORECASTS_TABLE = """
IF NOT EXISTS (
    SELECT * FROM sysobjects WHERE name='daily_forecasts' AND xtype='U'
)
CREATE TABLE daily_forecasts (
    id                INT IDENTITY(1,1) PRIMARY KEY,
    forecast_date     DATE,
    product_category  NVARCHAR(50),
    predicted_revenue FLOAT,
    model_version     NVARCHAR(20),
    created_at        DATETIME DEFAULT GETDATE()
);
"""

CREATE_ANOMALIES_TABLE = """
IF NOT EXISTS (
    SELECT * FROM sysobjects WHERE name='anomaly_records' AND xtype='U'
)
CREATE TABLE anomaly_records (
    id                INT IDENTITY(1,1) PRIMARY KEY,
    record_date       DATE,
    product_category  NVARCHAR(50),
    daily_revenue     FLOAT,
    anomaly_score     FLOAT,
    anomaly_flag      INT,
    revenue_z_score   FLOAT,
    created_at        DATETIME DEFAULT GETDATE()
);
"""


def get_connection():
    """Return a pyodbc connection or None if not configured."""
    if not PYODBC_AVAILABLE:
        log.warning("pyodbc not available — skipping SQL connection.")
        return None
    if not AZURE_SQL_CONN:
        log.warning("AZURE_SQL_CONNECTION_STRING not set in .env — skipping SQL.")
        return None
    try:
        conn = pyodbc.connect(AZURE_SQL_CONN, timeout=10)
        return conn
    except Exception as e:
        log.error(f"Azure SQL connection failed: {e}")
        return None


def init_tables():
    """Create tables if they don't exist. Called on app startup."""
    conn = get_connection()
    if conn is None:
        log.info("Azure SQL not configured — tables not created (safe to skip in local dev).")
        return False
    try:
        cursor = conn.cursor()
        for sql in [CREATE_TRANSACTIONS_TABLE, CREATE_FORECASTS_TABLE, CREATE_ANOMALIES_TABLE]:
            cursor.execute(sql)
        conn.commit()
        cursor.close()
        conn.close()
        log.info("Azure SQL tables initialised.")
        return True
    except Exception as e:
        log.error(f"Table init failed: {e}")
        return False


def insert_transactions(df: pd.DataFrame) -> int:
    """
    Bulk insert a DataFrame of transactions into sales_transactions.
    Returns number of rows inserted, or 0 on failure.
    """
    conn = get_connection()
    if conn is None:
        return 0

    rows_inserted = 0
    try:
        cursor = conn.cursor()
        sql = """
            INSERT INTO sales_transactions
                (transaction_id, date, customer_id, gender, age,
                 product_category, quantity, price_per_unit, total_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for _, row in df.iterrows():
            cursor.execute(sql, (
                str(row.get("Transaction ID", "")),
                str(row.get("Date", "")),
                str(row.get("Customer ID", "")),
                str(row.get("Gender", "")),
                int(row.get("Age", 0)),
                str(row.get("Product Category", "")),
                int(row.get("Quantity", 0)),
                float(row.get("Price per Unit", 0)),
                float(row.get("Total Amount", 0)),
            ))
            rows_inserted += 1

        conn.commit()
        cursor.close()
        conn.close()
        log.info(f"Inserted {rows_inserted} rows into sales_transactions.")
    except Exception as e:
        log.error(f"insert_transactions failed: {e}")

    return rows_inserted


def insert_forecast(date: str, category: str, predicted_revenue: float) -> bool:
    """Insert one forecast record."""
    conn = get_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO daily_forecasts (forecast_date, product_category, predicted_revenue, model_version)
            VALUES (?, ?, ?, ?)
            """,
            (date, category, predicted_revenue, "xgb_v1"),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        log.error(f"insert_forecast failed: {e}")
        return False


def insert_anomalies(df: pd.DataFrame) -> int:
    """Bulk insert anomaly records."""
    conn = get_connection()
    if conn is None:
        return 0
    rows = 0
    try:
        cursor = conn.cursor()
        sql = """
            INSERT INTO anomaly_records
                (record_date, product_category, daily_revenue,
                 anomaly_score, anomaly_flag, revenue_z_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        for _, row in df.iterrows():
            cursor.execute(sql, (
                str(row.get("Date", "")),
                str(row.get("Product Category", "")),
                float(row.get("daily_revenue", 0)),
                float(row.get("anomaly_score", 0)),
                int(row.get("anomaly_flag", 0)),
                float(row.get("revenue_z_score", 0)),
            ))
            rows += 1
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        log.error(f"insert_anomalies failed: {e}")
    return rows