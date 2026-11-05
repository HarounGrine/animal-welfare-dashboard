# Animal Welfare Monitoring Platform
### Executive Dashboard – Proof of Concept (POC)

**Project:** Animal Welfare Monitoring and Root-Cause Analysis  
**Application:** Interactive Streamlit Dashboard  
**Status:** Initial Proof of Concept  
**Data:** Synthetic / simulated demonstration data

---

## 1. Project Overview

The Animal Welfare Monitoring Platform is a data-driven proof of concept designed to bring together different sources of information from slaughterhouse operations into one centralised monitoring and analysis environment.

The objective is to help slaughterhouse operators and management monitor animal welfare indicators, identify unusual patterns, investigate potential contributing factors and make better-informed operational decisions.

In many operational environments, relevant information is distributed across different systems, including video monitoring, acoustic sensors, production records, welfare reports and animal traceability systems.

This project explores how these different data sources could be integrated and analysed together to provide a more comprehensive picture of animal welfare and operational conditions.

The long-term vision is to move from fragmented monitoring and retrospective reporting towards a more integrated and proactive approach to animal welfare management.

## 2. Project Objectives

The main objectives are:

- **Centralised monitoring:** Combine information from different operational and monitoring systems in one dashboard.
- **Early identification of unusual patterns:** Highlight changes or unusual activity across monitoring streams.
- **Root-cause exploration:** Investigate possible relationships between operational conditions and recorded welfare events.
- **Operational transparency:** Provide management with an overview of important indicators across different zones and time periods.
- **Data quality assurance:** Monitor the completeness, consistency and reliability of incoming data.
- **Future AI integration:** Establish a foundation for automated anomaly detection, predictive analysis and AI-assisted monitoring.

## 3. Data Sources and Monitoring Streams

The prototype is structured around five main data streams.

### 3.1 Video Monitoring

Video monitoring data represents potential welfare-related events identified through video analysis.

The dashboard includes:

- Detection timestamps and event duration.
- Operational zone where the event was detected.
- Event classification.
- Model confidence score.
- Human review status.
- Review outcome: confirmed, rejected or uncertain.

The dashboard distinguishes between detected events and events that have been reviewed and confirmed.

This distinction is important because an automated detection should not automatically be interpreted as a verified welfare incident.

### 3.2 Acoustic Monitoring

Acoustic data provides information about the sound environment and animal vocalisation patterns.

The prototype includes:

- Noise levels measured in decibels (dB).
- Animal vocalisation rate.
- Timestamp of each observation.
- Monitoring zone.

These indicators can be explored alongside other monitoring streams to investigate whether unusual acoustic conditions coincide with operational or welfare-related observations.

### 3.3 Process and Operational Monitoring

Operational data describes conditions within the slaughterhouse process.

The prototype includes:

- Animal waiting time.
- Queue length.
- Production line speed.
- Timestamp of each observation.
- Operational zone.

These variables allow the exploration of potential relationships between operational pressure and recorded welfare observations.

For example, the analysis can investigate whether longer waiting times or increased queue lengths coincide with higher numbers of video detections or reported welfare events.

### 3.4 Welfare Events and Client Records

This stream represents welfare observations and incidents recorded by the client or operational personnel.

The data includes:

- Event timestamp.
- Operational zone.
- Welfare event category.
- Client-stated potential cause.

This information provides an additional reference for comparing recorded incidents with patterns identified in video, acoustic and process data.

### 3.5 Animal Traceability and Ear-Tag Records

The traceability component provides additional information about animal arrivals and transport conditions.

The prototype includes:

- Animal identifier.
- Batch identifier.
- Origin farm.
- Arrival timestamp.
- Transport duration.

This information can help explore transport-related patterns and provide additional context for animal arrivals.

**Current limitation:** The prototype does not yet contain a shared animal or batch identifier linking ear-tag records to video detections, acoustic observations or welfare events. Individual animal-level cross-stream analysis therefore requires further data integration.

## 4. Dashboard Functionality

The Streamlit application provides an interactive executive overview with six main analytical sections.

### 4.1 Executive KPIs

The dashboard displays key indicators, including:

- Number of unique animals and batches.
- Number of video detections.
- Percentage of video detections reviewed.
- Confirmation rate among reviewed detections.
- Number of recorded welfare events.
- Average waiting time and queue length.
- Average production line speed.
- Average acoustic noise level and vocalisation rate.
- Overall data validation rate.

These indicators provide a high-level overview of the selected monitoring period.

### 4.2 Unified Monitoring Timeline

Combines information from the different monitoring streams into a common time-based view.

It allows users to compare:

- Video detections and confirmed detections.
- Recorded welfare events.
- Average waiting time and queue length.
- Acoustic noise levels and vocalisation rates.

Users can switch between hourly and daily aggregation to examine patterns at different time resolutions.

### 4.3 Zone-Level Analysis

Provides comparisons between operational zones.

The dashboard includes:

- Zone-level operational indicators.
- Average waiting times and queue lengths.
- Acoustic measurements.
- Video detection counts.
- Confirmed detection counts.
- Recorded welfare events.
- A comparative heatmap.
- Correlation analysis across hourly observations.

The correlation analysis uses Spearman rank correlation to explore associations between selected variables.

**Important:** Correlation indicates statistical association, not causation. Further investigation and validation are required before drawing conclusions about the causes of welfare events.

### 4.4 Video Detection and Review Analysis

Provides an overview of the video detection and validation process.

It includes:

- Detection-to-review funnel.
- Number of detections reviewed.
- Number of confirmed detections.
- Event types and review outcomes.
- Distribution of model confidence scores.

This helps distinguish between automated detections and events validated through human review.

### 4.5 Welfare Events and Root-Cause Exploration

Provides an overview of recorded welfare events and their documented causes.

It includes:

- Welfare events by category and zone.
- Most frequently recorded client-stated causes.
- Comparisons between video detections and recorded welfare events.
- A category-to-cause breakdown.

This section is intended to support further investigation into recurring patterns and possible contributing factors.

### 4.6 Traceability, Data Quality and Coverage

The final sections provide additional operational and technical context.

**Traceability analysis:**
- Transport duration distribution.
- Animal arrivals by origin farm.
- Daily arrival volumes.
- Average transport duration.

**Data quality and coverage:**
- Valid records.
- Rejected records.
- Duplicate records.
- Record counts by dataset.
- Daily monitoring coverage across the main streams.

These indicators help assess the quality and availability of the underlying information.

## 5. Technical Architecture

The current proof of concept follows a simple data-processing and visualisation architecture.

```text
        DATA SOURCES
              |
    +---------+---------+
    |         |         |
  Video   Acoustic   Process
    |         |         |
    +---------+---------+
              |
       Welfare Events
              |
       Ear-Tag Records
              |
              v
     DATA PROCESSING
       Python / Pandas
              |
    Validation and Cleaning
              |
       Curated CSV Files
              |
              v
      STREAMLIT APP
       Data Loading
       Filtering
       Aggregation
       Statistical Analysis
              |
              v
    INTERACTIVE DASHBOARD
       KPIs and Charts
       Timelines
       Zone Comparisons
       Correlation Analysis
       Data Quality
```

### Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing, filtering and aggregation |
| NumPy | Numerical calculations |
| Plotly | Interactive charts and visualisations |
| Streamlit | Interactive web application |
| GitHub | Source code version control and hosting |
| Streamlit Community Cloud | Online dashboard deployment |

The dashboard reads curated CSV datasets and performs the required filtering, aggregation and visualisation.

It does not currently execute video recognition, acoustic classification or predictive machine-learning models itself. It visualises structured outputs representing those monitoring streams.

## 6. Project Structure

The repository is organised as follows:

```text
animal-welfare-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    └── curated/
        ├── video_flags.csv
        ├── acoustic_records.csv
        ├── process_records.csv
        ├── welfare_events.csv
        ├── ear_tag_records.csv
        └── data_quality_report.csv
```

### Main Components

- `app.py`: Main Streamlit application containing data loading, filtering, KPI calculations, analytical functions and visualisations.
- `requirements.txt`: Python dependencies required to run the application.
- `data/curated/`: Contains the structured datasets consumed by the dashboard.
- `README.md`: Project documentation.

## 7. Dataset Requirements

The dashboard expects six CSV files with the following required columns.

| Dataset | Required columns |
|---|---|
| video_flags.csv | ts_start, ts_end, zone, event_type, confidence, reviewed, review_outcome |
| acoustic_records.csv | ts, zone, noise_db, vocalisation_rate |
| process_records.csv | ts, zone, wait_minutes, queue_length, line_speed |
| welfare_events.csv | ts_logged, zone, category, stated_cause |
| ear_tag_records.csv | arrival_ts
