\# Azure Data Factory — Setup Guide

\## Smart Retail Assistant



\---



\## Prerequisites

\- Azure Blob Storage account created

\- Azure Databricks workspace created

\- `python scripts/upload\_to\_blob.py` already run



\---



\## Step 1 — Create Azure Data Factory



1\. Go to \*\*portal.azure.com\*\*

2\. Click \*\*Create a resource\*\* → search \*\*Data Factory\*\* → Create

3\. Fill in:

&#x20;  - Name: `smart-retail-adf`

&#x20;  - Region: Same as your storage account

&#x20;  - Version: V2

4\. Click \*\*Review + Create\*\* → \*\*Create\*\*

5\. Once deployed → click \*\*Launch Studio\*\*



\---



\## Step 2 — Create Linked Service: Azure Blob Storage



In ADF Studio:

1\. Click \*\*Manage\*\* (toolbox icon on left)

2\. \*\*Linked Services\*\* → \*\*+ New\*\*

3\. Choose \*\*Azure Blob Storage\*\* → Continue

4\. Fill in:

&#x20;  - Name: `LS\_AzureBlobStorage`

&#x20;  - Auth method: \*\*Connection string\*\*

&#x20;  - Connection string: paste from `.env` → `AZURE\_BLOB\_CONNECTION\_STRING`

5\. Click \*\*Test connection\*\* → should show green ✅

6\. Click \*\*Create\*\*



\---



\## Step 3 — Create Linked Service: Azure Databricks



In ADF Studio:

1\. \*\*Linked Services\*\* → \*\*+ New\*\*

2\. Choose \*\*Azure Databricks\*\* → Continue

3\. Fill in:

&#x20;  - Name: `LS\_AzureDatabricks`

&#x20;  - Databricks workspace: select your workspace

&#x20;  - Cluster: \*\*Existing interactive cluster\*\* → select your running cluster

&#x20;  - Access token: from Databricks → User Settings → Access Tokens → Generate

4\. Click \*\*Test connection\*\* → green ✅

5\. Click \*\*Create\*\*



\---



\## Step 4 — Create Datasets



\*\*Dataset 1: Bronze Raw CSV (source)\*\*

1\. \*\*Author\*\* → \*\*Datasets\*\* → \*\*+ New\*\*

2\. Choose \*\*Azure Blob Storage\*\* → \*\*DelimitedText\*\* → Continue

3\. Name: `DS\_BronzeRawCSV`

4\. Linked service: `LS\_AzureBlobStorage`

5\. File path: `bronze` / `raw` / `retail\_sales.csv`

6\. First row as header: ✅

7\. Click \*\*OK\*\*



\*\*Dataset 2: Bronze Output (sink)\*\*

1\. \*\*Datasets\*\* → \*\*+ New\*\* → \*\*Azure Blob Storage\*\* → \*\*DelimitedText\*\*

2\. Name: `DS\_BronzeOutput`

3\. Linked service: `LS\_AzureBlobStorage`

4\. File path: `bronze` / `processed` / `retail\_sales\_clean.csv`

5\. Click \*\*OK\*\*



\---



\## Step 5 — Import Pipeline



1\. In ADF Studio → \*\*Author\*\* → \*\*Pipelines\*\* → click `...` → \*\*Import from JSON\*\*

2\. Upload `adf/pipeline\_definition.json`

3\. The pipeline `RetailDataPipeline` will appear with 4 activities:

&#x20;  ```

&#x20;  CopyRawCSVToBlob → RunBronzeToSilver → RunSilverToGold → NotifySuccess

&#x20;  ```

4\. Click \*\*Validate All\*\* — fix any linked service references if flagged

5\. Click \*\*Publish All\*\*



\---



\## Step 6 — Upload Notebooks to Databricks



1\. Go to \*\*Databricks workspace\*\*

2\. \*\*Workspace\*\* → \*\*Shared\*\* → Create folder `SmartRetail`

3\. Import notebook `01\_bronze\_to\_silver.py`:

&#x20;  - Click \*\*Import\*\* → \*\*File\*\* → upload `notebooks/01\_bronze\_to\_silver.py`

&#x20;  - Rename to `01\_bronze\_to\_silver`

4\. Import notebook `02\_silver\_to\_gold.py`:

&#x20;  - Same steps → rename to `02\_silver\_to\_gold`

5\. In each notebook, update:

&#x20;  ```python

&#x20;  STORAGE\_ACCOUNT = "yourstorageaccount"  # your actual storage account name

&#x20;  STORAGE\_KEY = "your-storage-key"        # or use dbutils.secrets

&#x20;  ```



\---



\## Step 7 — Run the Pipeline



1\. In ADF Studio → \*\*Pipelines\*\* → `RetailDataPipeline`

2\. Click \*\*Debug\*\* (runs immediately without trigger)

3\. Watch the activity run status:

&#x20;  ```

&#x20;  CopyRawCSVToBlob  →  ✅ Succeeded

&#x20;  RunBronzeToSilver →  ✅ Succeeded  (\~2 min)

&#x20;  RunSilverToGold   →  ✅ Succeeded  (\~1 min)

&#x20;  NotifySuccess     →  ✅ Succeeded

&#x20;  ```

4\. Check Databricks → \*\*Data\*\* → verify `retail\_gold` database has 4 tables



\---



\## Step 8 — Set Pipeline Trigger (optional)



1\. ADF Studio → \*\*Manage\*\* → \*\*Triggers\*\* → \*\*+ New\*\*

2\. Type: \*\*Schedule\*\*

3\. Recurrence: \*\*Daily at 06:00 UTC\*\*

4\. Click \*\*OK\*\* → \*\*Publish All\*\*



\---



\## Verify Gold Tables in Databricks



Run this in a Databricks notebook cell:

```python

%sql

SHOW TABLES IN retail\_gold;

SELECT COUNT(\*) FROM retail\_gold.daily\_revenue;

SELECT COUNT(\*) FROM retail\_gold.monthly\_summary;

SELECT COUNT(\*) FROM retail\_gold.customer\_segments;

SELECT COUNT(\*) FROM retail\_gold.anomaly\_records;

```



Expected output:

```

daily\_revenue      1,056 rows

monthly\_summary      36 rows

customer\_segments    \~60 rows

anomaly\_records    1,056 rows (53 flagged)

```

