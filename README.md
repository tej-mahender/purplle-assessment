# Store Intelligence Analytics Platform

## Overview

Store Intelligence Analytics Platform is a retail analytics solution that processes customer movement events and transaction data to generate actionable business insights. The application provides analytics such as customer footfall, conversion funnels, store heatmaps, and anomaly detection through REST APIs.

The platform is built using FastAPI, SQLite, and Docker, and is designed to help retailers understand customer behavior and optimize store performance.

---

## Features

- Customer Footfall Analysis
- Conversion Funnel Analytics
- Store Heatmap Generation
- Store Performance Metrics
- Anomaly Detection
- RESTful APIs
- Interactive Swagger Documentation
- Dockerized Deployment

---

## Technology Stack

### Backend
- Python
- FastAPI
- Uvicorn

### Database
- SQLite

### Data Processing
- Pandas
- NumPy

### Deployment
- Docker
- Docker Compose
- Render

---

## Architecture

```text
Customer Events
       │
       ▼
Event Ingestion
       │
       ▼
Analytics Engine
       │
       ├── Footfall Analysis
       ├── Funnel Analysis
       ├── Heatmap Generation
       └── Anomaly Detection
       │
       ▼
REST APIs
       │
       ▼
Swagger Documentation
```

---

## Project Structure

```text
store-intelligence/
│
├── app/
├── pipeline/
├── dashboard/
├── data/
├── tests/
├── docs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites

Before running the application, ensure the following are installed:

- Docker Desktop
- Git
- Python 3.10+ (optional)

---

## Installation and Setup

### Clone Repository

```bash
git clone <repository-url>
cd store-intelligence
```

### Configure Environment

```bash
cp .env.example .env
```

### Build and Run

```bash
docker compose up --build
```

### Verify Application Health

Open:

```text
http://localhost:8000/health
```

Expected Response:

```json
{
  "status": "ok"
}
```

### Access Swagger Documentation

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Returns application status and service health.

### Ingest Events

```http
POST /events/ingest
```

Ingests customer event data into the system.

### Store Metrics

```http
GET /stores/{store_id}/metrics
```

Returns key store performance metrics.

### Conversion Funnel

```http
GET /stores/{store_id}/funnel
```

Returns customer conversion funnel analytics.

### Heatmap Analytics

```http
GET /stores/{store_id}/heatmap
```

Returns zone-level customer activity data.

### Anomaly Detection

```http
GET /stores/{store_id}/anomalies
```

Returns detected anomalies and operational alerts.

---

## Running Event Ingestion

After generating or obtaining an `events.jsonl` file:

```bash
python pipeline/ingest_events.py --events ./data/events.jsonl
```

This imports customer event data into the analytics platform.

---

## Deployment on Render

### Step 1

Push the project to GitHub.

### Step 2

Create a new Web Service on Render.

### Step 3

Connect the GitHub repository.

### Step 4

Select:

```text
Environment: Docker
```

### Step 5

Configure Health Check Path:

```text
/health
```

### Step 6

Deploy the application.

### Step 7

Access the deployed application:

```text
https://<your-render-url>/docs
```

---

## Sample Environment Variables

```env
DATABASE_URL=sqlite+aiosqlite:///./data/store_intelligence.db
STORE_LAYOUT_PATH=./data/store_layout.json
POS_TRANSACTIONS_PATH=./data/pos_transactions.csv
LOG_LEVEL=INFO
```

---

## Use Cases

- Retail Store Analytics
- Customer Behavior Tracking
- Footfall Measurement
- Conversion Monitoring
- Queue Management
- Store Layout Optimization
- Operational Anomaly Detection

---

## Future Enhancements

- Real-Time Dashboard
- Machine Learning Forecasting
- Multi-Store Analytics
- Cloud Database Integration
- Advanced Visualizations

---

## Author

Developed as part of the Apex Retail / Purplle Store Intelligence Challenge.

---

## License

This project is intended for educational, research, and assessment purposes.
