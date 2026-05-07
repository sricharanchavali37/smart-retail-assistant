\# Power BI Dashboard — Setup Guide

\## Smart Retail Assistant



\---



\## Connect Power BI to Databricks



1\. Open \*\*Power BI Desktop\*\*

2\. \*\*Get Data\*\* → search \*\*Azure Databricks\*\* → Connect

3\. Fill in:

&#x20;  - Server hostname: `<your-workspace>.azuredatabricks.net`

&#x20;  - HTTP Path: `/sql/1.0/warehouses/<warehouse-id>`

&#x20;  - (Find this in Databricks → SQL Warehouses → your warehouse → Connection details)

4\. Authentication: \*\*Personal Access Token\*\* → paste your Databricks token

5\. Navigator → expand \*\*retail\_gold\*\* → select all 4 tables:

&#x20;  - ✅ daily\_revenue

&#x20;  - ✅ monthly\_summary

&#x20;  - ✅ customer\_segments

&#x20;  - ✅ anomaly\_records

6\. Click \*\*Load\*\*



\---



\## Add Forecast Data (from FastAPI)



Since Power BI cannot call REST APIs natively in basic mode,

export forecast data to CSV first:



Run in CMD:

```cmd

python scripts/export\_forecasts.py

```

Then in Power BI → \*\*Get Data\*\* → \*\*Text/CSV\*\* → select `data/processed/forecast\_export.csv`



\---



\## Page 1 — Demand Forecast



\*\*Title:\*\* Demand Forecast — Next 7 Days



\*\*Visuals to add:\*\*



1\. \*\*Line Chart\*\* — Predicted Revenue by Day

&#x20;  - X-axis: `date`

&#x20;  - Y-axis: `daily\_revenue` (from `daily\_revenue` table)

&#x20;  - Legend: `product\_category`

&#x20;  - Title: "Daily Revenue Trend by Category"



2\. \*\*Card\*\* — Total Predicted Revenue

&#x20;  - Field: SUM of `daily\_revenue`

&#x20;  - Format: Currency



3\. \*\*Card\*\* — Average Daily Revenue

&#x20;  - Field: AVERAGE of `daily\_revenue`



4\. \*\*Clustered Bar Chart\*\* — Revenue by Category

&#x20;  - X-axis: `product\_category`

&#x20;  - Y-axis: SUM `daily\_revenue`

&#x20;  - Colors: Green=Beauty, Blue=Clothing, Orange=Electronics



5\. \*\*Slicer\*\* — Month filter

&#x20;  - Field: `month` from `daily\_revenue`



\---



\## Page 2 — Anomaly Alerts



\*\*Title:\*\* Anomaly Detection Dashboard



\*\*Visuals to add:\*\*



1\. \*\*Table\*\* — Top Anomalous Days

&#x20;  - Columns: `date`, `product\_category`, `daily\_revenue`, `anomaly\_score`, `risk\_level`

&#x20;  - Sort by: `anomaly\_score` descending

&#x20;  - Conditional formatting: RED if `risk\_level = HIGH`



2\. \*\*Clustered Column Chart\*\* — Anomalies by Category

&#x20;  - X-axis: `product\_category`

&#x20;  - Y-axis: COUNT of rows where `anomaly\_flag = 1`

&#x20;  - Title: "Anomaly Count by Category"



3\. \*\*Scatter Chart\*\* — Revenue vs Anomaly Score

&#x20;  - X-axis: `daily\_revenue`

&#x20;  - Y-axis: `anomaly\_score`

&#x20;  - Legend: `product\_category`

&#x20;  - Title: "Revenue vs Anomaly Risk"



4\. \*\*Card\*\* — Total Anomalies Detected

&#x20;  - Measure: COUNTROWS(FILTER(anomaly\_records, anomaly\_records\[anomaly\_flag] = 1))



5\. \*\*Line Chart\*\* — Anomaly Score Over Time

&#x20;  - X-axis: `date`

&#x20;  - Y-axis: `anomaly\_score`

&#x20;  - Legend: `product\_category`

&#x20;  - Add reference line at 0.55 (medium risk threshold)



\---



\## Page 3 — Category Performance



\*\*Title:\*\* Category Performance 2023



\*\*Visuals to add:\*\*



1\. \*\*Stacked Area Chart\*\* — Monthly Revenue by Category

&#x20;  - X-axis: `year\_month` (from `monthly\_summary`)

&#x20;  - Y-axis: `monthly\_revenue`

&#x20;  - Legend: `product\_category`



2\. \*\*Pie Chart\*\* — Revenue Share by Category

&#x20;  - Values: SUM `monthly\_revenue`

&#x20;  - Legend: `product\_category`



3\. \*\*Matrix Table\*\* — Monthly Breakdown

&#x20;  - Rows: `product\_category`

&#x20;  - Columns: month names

&#x20;  - Values: SUM `monthly\_revenue`

&#x20;  - Conditional formatting: heat map (green=high, red=low)



4\. \*\*KPI Cards\*\* (one per category)

&#x20;  - Beauty: Total Revenue, Avg Transaction

&#x20;  - Clothing: Total Revenue, Avg Transaction

&#x20;  - Electronics: Total Revenue, Avg Transaction



\---



\## Page 4 — Customer Insights



\*\*Title:\*\* Customer Segment Analysis



\*\*Visuals to add:\*\*



1\. \*\*Clustered Bar Chart\*\* — Revenue by Age Group

&#x20;  - X-axis: `age\_group` (from `customer\_segments`)

&#x20;  - Y-axis: SUM `total\_revenue`

&#x20;  - Legend: `product\_category`



2\. \*\*Donut Chart\*\* — Gender Split

&#x20;  - Values: SUM `transaction\_count`

&#x20;  - Legend: `gender`

&#x20;  - (one per category using page filter)



3\. \*\*Treemap\*\* — Spend by Category + Gender

&#x20;  - Group: `product\_category`

&#x20;  - Details: `gender`

&#x20;  - Values: SUM `total\_revenue`



4\. \*\*Bar Chart\*\* — Avg Spend by Age Group

&#x20;  - X-axis: `age\_group`

&#x20;  - Y-axis: AVERAGE `avg\_spend`



\---



\## DAX Measures to Add



In Power BI → \*\*Modeling\*\* → \*\*New Measure\*\*, add these:



```dax

Total Revenue = SUM(daily\_revenue\[daily\_revenue])



Anomaly Count = COUNTROWS(FILTER(anomaly\_records, anomaly\_records\[anomaly\_flag] = 1))



Anomaly Rate % = DIVIDE(\[Anomaly Count], COUNTROWS(anomaly\_records), 0) \* 100



Avg Daily Revenue = AVERAGE(daily\_revenue\[daily\_revenue])



Revenue Growth MoM = 

VAR CurrentMonth = MAX(monthly\_summary\[monthly\_revenue])

VAR PrevMonth = CALCULATE(

&#x20;   MAX(monthly\_summary\[monthly\_revenue]),

&#x20;   DATEADD(monthly\_summary\[year\_month], -1, MONTH)

)

RETURN DIVIDE(CurrentMonth - PrevMonth, PrevMonth, 0) \* 100

```



\---



\## Publish \& Share



1\. \*\*File\*\* → \*\*Publish\*\* → \*\*Publish to Power BI\*\*

2\. Sign in with your Microsoft account

3\. Select workspace → click \*\*Select\*\*

4\. Once published → go to \*\*app.powerbi.com\*\*

5\. Find your report → click \*\*Share\*\* → copy link



\*\*Include this link in your project submission as the Power BI dashboard URL.\*\*



\---



\## Screenshot Checklist for Submission



Take screenshots of:

\- \[ ] Page 1: Demand Forecast with line chart showing all 3 categories

\- \[ ] Page 2: Anomaly table showing HIGH risk days highlighted in red

\- \[ ] Page 3: Monthly stacked area chart (full year 2023)

\- \[ ] Page 4: Customer segments treemap

\- \[ ] ADF pipeline run showing all 4 activities green ✅

\- \[ ] Databricks gold tables (run `SHOW TABLES IN retail\_gold`)

