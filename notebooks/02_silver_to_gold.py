# Databricks notebook source
# MAGIC %md
# MAGIC # Silver → Gold: Retail Analytics Aggregations
# MAGIC **Smart Retail Assistant — Data Pipeline**
# MAGIC
# MAGIC This notebook:
# MAGIC - Reads cleaned silver Delta table
# MAGIC - Creates 4 gold aggregation tables for Power BI
# MAGIC - Joins with XGBoost forecast outputs and anomaly scores

# COMMAND ----------

SILVER_DB = "retail_silver"
GOLD_DB   = "retail_gold"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {GOLD_DB}")

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## 1. Load Silver Table

# COMMAND ----------

df = spark.table(f"{SILVER_DB}.transactions")
print(f"Silver records: {df.count():,}")
display(df.limit(3))

# COMMAND ----------

# MAGIC %md ## 2. Gold Table 1: Daily Revenue by Category
# MAGIC *Used by Power BI: Demand Forecast page, Revenue Trends page*

# COMMAND ----------

daily_revenue = df.groupBy("date", "product_category", "year", "month", "day_of_week", "quarter", "is_weekend") \
    .agg(
        F.sum("total_amount").alias("daily_revenue"),
        F.sum("quantity").alias("daily_quantity"),
        F.count("transaction_id").alias("transaction_count"),
        F.avg("price_per_unit").alias("avg_price"),
        F.avg("age").alias("avg_customer_age"),
        F.avg(F.when(F.col("gender") == "F", 1).otherwise(0)).alias("female_ratio"),
    ) \
    .orderBy("date", "product_category")

daily_revenue.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{GOLD_DB}.daily_revenue")

print(f"Gold: daily_revenue  →  {spark.table(f'{GOLD_DB}.daily_revenue').count():,} rows")
display(spark.table(f"{GOLD_DB}.daily_revenue").limit(5))

# COMMAND ----------

# MAGIC %md ## 3. Gold Table 2: Monthly Summary
# MAGIC *Used by Power BI: Monthly Performance page*

# COMMAND ----------

monthly_summary = df.groupBy("year", "month", "product_category") \
    .agg(
        F.sum("total_amount").alias("monthly_revenue"),
        F.sum("quantity").alias("monthly_quantity"),
        F.count("transaction_id").alias("monthly_transactions"),
        F.avg("total_amount").alias("avg_transaction_value"),
        F.countDistinct("customer_id").alias("unique_customers"),
    ) \
    .withColumn("year_month", F.concat_ws("-", F.col("year"), F.lpad(F.col("month"), 2, "0"))) \
    .orderBy("year", "month", "product_category")

monthly_summary.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{GOLD_DB}.monthly_summary")

print(f"Gold: monthly_summary  →  {spark.table(f'{GOLD_DB}.monthly_summary').count():,} rows")
display(spark.table(f"{GOLD_DB}.monthly_summary").limit(5))

# COMMAND ----------

# MAGIC %md ## 4. Gold Table 3: Customer Segments
# MAGIC *Used by Power BI: Customer Insights page*

# COMMAND ----------

# Age brackets
df_seg = df.withColumn(
    "age_group",
    F.when(F.col("age") < 25, "18-24")
     .when(F.col("age") < 35, "25-34")
     .when(F.col("age") < 45, "35-44")
     .when(F.col("age") < 55, "45-54")
     .otherwise("55+")
)

customer_segments = df_seg.groupBy("product_category", "gender", "age_group") \
    .agg(
        F.count("transaction_id").alias("transaction_count"),
        F.sum("total_amount").alias("total_revenue"),
        F.avg("total_amount").alias("avg_spend"),
        F.avg("quantity").alias("avg_quantity"),
    ) \
    .orderBy("product_category", "gender", "age_group")

customer_segments.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{GOLD_DB}.customer_segments")

print(f"Gold: customer_segments  →  {spark.table(f'{GOLD_DB}.customer_segments').count():,} rows")
display(spark.table(f"{GOLD_DB}.customer_segments").limit(8))

# COMMAND ----------

# MAGIC %md ## 5. Gold Table 4: Anomaly Summary
# MAGIC *Used by Power BI: Anomaly Alerts page*

# COMMAND ----------

# Read anomaly_scores.csv from blob or silver
STORAGE_ACCOUNT = "yourstorageaccount"
BLOB_BASE = f"wasbs://bronze@{STORAGE_ACCOUNT}.blob.core.windows.net"

df_anomaly = spark.read.csv(
    f"{BLOB_BASE}/processed/anomaly_scores.csv",
    header=True,
    inferSchema=True,
)
df_anomaly = df_anomaly.withColumn("Date", F.to_date("Date", "yyyy-MM-dd"))

# Flag high-risk anomalies
df_anomaly = df_anomaly.withColumn(
    "risk_level",
    F.when(F.col("anomaly_score") >= 0.65, "HIGH")
     .when(F.col("anomaly_score") >= 0.55, "MEDIUM")
     .otherwise("LOW")
)

anomaly_gold = df_anomaly.select(
    F.col("Date").alias("date"),
    F.col("Product Category").alias("product_category"),
    "daily_revenue",
    "daily_quantity",
    "transaction_count",
    "revenue_z_score",
    "pct_from_rolling_mean",
    "rolling_mean_7",
    "anomaly_flag",
    "anomaly_score",
    "risk_level",
)

anomaly_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{GOLD_DB}.anomaly_records")

flagged = anomaly_gold.filter(F.col("anomaly_flag") == 1).count()
print(f"Gold: anomaly_records  →  {anomaly_gold.count():,} rows  ({flagged} anomalies flagged)")
display(anomaly_gold.filter(F.col("anomaly_flag") == 1).orderBy(F.desc("anomaly_score")).limit(10))

# COMMAND ----------

# MAGIC %md ## 6. Gold Summary Report

# COMMAND ----------

print("=" * 55)
print("  GOLD LAYER — COMPLETE")
print("=" * 55)

tables = ["daily_revenue", "monthly_summary", "customer_segments", "anomaly_records"]
for t in tables:
    count = spark.table(f"{GOLD_DB}.{t}").count()
    print(f"  {t:<25} {count:>6,} rows  ✅")

print("\n  Connect Power BI to Databricks SQL Endpoint:")
print("  Server:   <your-workspace>.azuredatabricks.net")
print("  HTTP Path: /sql/1.0/warehouses/<warehouse-id>")
print("  Database: retail_gold")
print("\n  Silver → Gold complete ✅")