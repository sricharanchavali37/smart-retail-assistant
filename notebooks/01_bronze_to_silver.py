# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze → Silver: Retail Sales Data Cleaning
# MAGIC **Smart Retail Assistant — Data Pipeline**
# MAGIC
# MAGIC This notebook:
# MAGIC - Reads raw retail_sales.csv from Azure Blob (bronze layer)
# MAGIC - Cleans and validates the data
# MAGIC - Engineers basic features
# MAGIC - Writes cleaned data as Delta table (silver layer)
# MAGIC
# MAGIC Run after uploading data via: `python scripts/upload_to_blob.py`

# COMMAND ----------

# MAGIC %md ## 1. Configure Azure Blob Storage Access
# MAGIC Set these in Databricks → Cluster → Environment Variables
# MAGIC or use the spark.conf below with your storage account details

# COMMAND ----------

# Replace with your values
STORAGE_ACCOUNT = "yourstorageaccount"   # e.g. smartretailstorage
CONTAINER_NAME  = "bronze"
STORAGE_KEY     = dbutils.secrets.get(scope="retail-scope", key="storage-key")
# OR hardcode for demo: STORAGE_KEY = "your-storage-account-key"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.blob.core.windows.net",
    STORAGE_KEY
)

BLOB_BASE = f"wasbs://{CONTAINER_NAME}@{STORAGE_ACCOUNT}.blob.core.windows.net"
SILVER_DB = "retail_silver"

print(f"Blob base path: {BLOB_BASE}")

# COMMAND ----------

# MAGIC %md ## 2. Read Raw CSV (Bronze Layer)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, DoubleType, DateType
)

RAW_SCHEMA = StructType([
    StructField("Transaction_ID",    IntegerType(),  True),
    StructField("Date",              StringType(),   True),
    StructField("Customer_ID",       StringType(),   True),
    StructField("Gender",            StringType(),   True),
    StructField("Age",               IntegerType(),  True),
    StructField("Product_Category",  StringType(),   True),
    StructField("Quantity",          IntegerType(),  True),
    StructField("Price_per_Unit",    DoubleType(),   True),
    StructField("Total_Amount",      DoubleType(),   True),
])

raw_path = f"{BLOB_BASE}/raw/retail_sales.csv"
print(f"Reading from: {raw_path}")

df_raw = spark.read.csv(
    raw_path,
    header=True,
    schema=RAW_SCHEMA,
    mode="PERMISSIVE",
)

print(f"Raw records loaded: {df_raw.count():,}")
df_raw.printSchema()
display(df_raw.limit(5))

# COMMAND ----------

# MAGIC %md ## 3. Clean & Validate

# COMMAND ----------

# Rename columns to snake_case
df_clean = df_raw.withColumnRenamed("Transaction_ID",   "transaction_id") \
                 .withColumnRenamed("Customer_ID",      "customer_id") \
                 .withColumnRenamed("Product_Category", "product_category") \
                 .withColumnRenamed("Price_per_Unit",   "price_per_unit") \
                 .withColumnRenamed("Total_Amount",     "total_amount")

# Parse date
df_clean = df_clean.withColumn("date", F.to_date("Date", "yyyy-MM-dd")) \
                   .drop("Date")

# Remove nulls in critical columns
before = df_clean.count()
df_clean = df_clean.dropna(subset=["date", "total_amount", "product_category", "quantity"])
after = df_clean.count()
print(f"Dropped {before - after:,} rows with nulls")

# Remove zero/negative amounts
df_clean = df_clean.filter(F.col("total_amount") > 0)
df_clean = df_clean.filter(F.col("quantity") > 0)
df_clean = df_clean.filter(F.col("age") > 0)

# Standardise gender
df_clean = df_clean.withColumn(
    "gender",
    F.when(F.col("gender") == "Female", "F")
     .when(F.col("gender") == "Male",   "M")
     .otherwise("Unknown")
)

# Encode product category
df_clean = df_clean.withColumn(
    "category_code",
    F.when(F.col("product_category") == "Beauty",      0)
     .when(F.col("product_category") == "Clothing",    1)
     .when(F.col("product_category") == "Electronics", 2)
     .otherwise(-1)
)

print(f"\nClean records: {df_clean.count():,}")

# COMMAND ----------

# MAGIC %md ## 4. Add Date Features

# COMMAND ----------

df_silver = df_clean \
    .withColumn("year",         F.year("date")) \
    .withColumn("month",        F.month("date")) \
    .withColumn("day",          F.dayofmonth("date")) \
    .withColumn("day_of_week",  F.dayofweek("date")) \
    .withColumn("week_of_year", F.weekofyear("date")) \
    .withColumn("quarter",      F.quarter("date")) \
    .withColumn("is_weekend",   (F.dayofweek("date").isin([1, 7])).cast("int")) \
    .withColumn("is_month_end", (F.dayofmonth("date") >= 25).cast("int"))

# Recalculate total_amount for consistency
df_silver = df_silver.withColumn(
    "total_amount_calc",
    F.col("quantity") * F.col("price_per_unit")
)

print("Silver schema:")
df_silver.printSchema()
display(df_silver.limit(5))

# COMMAND ----------

# MAGIC %md ## 5. Data Quality Report

# COMMAND ----------

print("=" * 50)
print("  SILVER LAYER — DATA QUALITY REPORT")
print("=" * 50)
print(f"  Total records      : {df_silver.count():,}")
print(f"  Date range         : {df_silver.agg(F.min('date')).first()[0]} → {df_silver.agg(F.max('date')).first()[0]}")
print(f"  Unique customers   : {df_silver.select('customer_id').distinct().count():,}")
print(f"  Categories         : {df_silver.select('product_category').distinct().count()}")
print(f"  Null check (date)  : {df_silver.filter(F.col('date').isNull()).count()}")
print(f"  Null check (amount): {df_silver.filter(F.col('total_amount').isNull()).count()}")

print("\n  Revenue by Category:")
df_silver.groupBy("product_category") \
         .agg(
             F.sum("total_amount").alias("total_revenue"),
             F.avg("total_amount").alias("avg_per_tx"),
             F.count("transaction_id").alias("transactions"),
         ) \
         .orderBy(F.desc("total_revenue")) \
         .show()

# COMMAND ----------

# MAGIC %md ## 6. Write to Delta Table (Silver Layer)

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_DB}")

df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{SILVER_DB}.transactions")

print(f"Written to: {SILVER_DB}.transactions")
print(f"Row count: {spark.table(f'{SILVER_DB}.transactions').count():,}")

# COMMAND ----------

# Verify
print("Silver table preview:")
display(spark.table(f"{SILVER_DB}.transactions").limit(10))
print("\nBronze → Silver complete ✅")