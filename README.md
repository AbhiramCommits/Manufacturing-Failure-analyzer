# Manufacturing Failure Analyzer

A Python-based tool for analyzing hardware test logs, detecting failure patterns via clustering, and generating AI-powered root-cause hypotheses.

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │            main.py (CLI)             │
                          │  ┌──────────┐  ┌──────────────────┐ │
                          │  │  serve   │  │     analyze      │ │
                          │  │ uvicorn  │  │ pipeline + LLM   │ │
                          │  └────┬─────┘  └────────┬─────────┘ │
                          └───────┼──────────────────┼──────────┘
                                  │                  │
          ┌───────────────────────┼──────────────────┼──────────────────┐
          │                  api/routes.py            │                  │
          │  GET /health  GET /failures  POST /analyze                  │
          │  GET /summary  GET /anomalies  GET /clusters/...            │
          └───────────────────────┬─────────────────────────────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
  ┌────▼─────┐  ┌──────────▼──────┐  ┌───────────▼──────┐  ┌▼──────────────┐
  │  loader  │  │    analyzer     │  │    llm_engine    │  │  visualizer   │
  │ CSV → DF │  │ Z-score +      │  │  OpenAI gpt-4o   │  │  PCA scatter  │
  │ validate │  │ K-Means/DBSCAN │  │  root-cause      │  │  + bar chart  │
  └────┬─────┘  └───────┬────────┘  └────────┬─────────┘  └──────┬────────┘
       │                │                    │                    │
       └────────────────┼────────────────────┼────────────────────┘
                        │                    │
                   ┌────▼────┐          ┌────▼─────┐
                   │ SQLite  │          │   .env    │
                   │  .db    │          │ API key   │
                   └─────────┘          └──────────┘
```

## Project Structure

```
.
├── api/
│   ├── __init__.py
│   └── routes.py              # FastAPI route handlers
├── data/                      # CSV logs, SQLite DB, chart PNG
│   └── sample_test_log.csv    # 500-row sample with 4 failure clusters
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV schema validation and loading
│   ├── analyzer.py            # Z-score anomaly detection, K-Means/DBSCAN
│   ├── llm_engine.py          # OpenAI gpt-4o root-cause analysis
│   └── visualizer.py          # Matplotlib cluster report generation
├── main.py                    # CLI entry point (serve / analyze)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

## Setup

### Option 1: Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your OpenAI API key (required for LLM root-cause analysis)
echo 'OPENAI_API_KEY=sk-...' > .env
```

### Option 2: Docker

```bash
# Build and run with Docker Compose
docker compose up --build

# Or build separately
docker build -t mfa .
docker run -p 8000:8000 -v ./data:/app/data -e OPENAI_API_KEY=sk-... mfa
```

## CLI Usage

```bash
# Start the API server
python main.py serve --host 0.0.0.0 --port 8000

# Run analysis on a CSV file (skip LLM if no API key)
python main.py analyze data/sample_test_log.csv --no-llm

# Full pipeline with output to JSON
python main.py analyze path/to/log.csv \
  --threshold 2.5 \
  --k 4 \
  --output-json results.json
```

## API Endpoints

Base URL: `http://localhost:8000`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check message |
| `GET` | `/api/health` | Service status + record count from SQLite |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/api/analyze` | Upload CSV (path or inline) → full pipeline |
| `GET` | `/api/failures` | Query persisted failures with filters |
| `GET` | `/api/summary` | Aggregate pass/fail/anomaly stats |
| `GET` | `/api/anomalies` | List anomaly records |
| `GET` | `/api/clusters/{kmeans\|dbscan}` | Per-cluster summaries |
| `GET` | `/api/clusters/{kmeans\|dbscan}/{id}/analyze` | LLM root-cause per cluster |

### Example curl Commands

**POST /api/analyze** (by file path)

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/sample_test_log.csv"}'
```

**POST /api/analyze** (inline CSV)

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"inline_csv": "test_id,component_id,timestamp,test_type,failure_code,error_message,duration_ms,temperature_c,voltage_v\nTST-0001,COMP-A-001,2025-06-01T08:00:00,thermal_stress,101,Thermal runaway,350,78.5,12.0"}'
```

**GET /api/health**

```bash
curl http://localhost:8000/api/health
# {"status":"healthy","record_count":500}
```

**GET /api/failures** (with filters)

```bash
curl "http://localhost:8000/api/failures?cluster_id=2&test_type=voltage_sweep&limit=5"
```

**GET /api/summary**

```bash
curl http://localhost:8000/api/summary
```

**GET /api/clusters/kmeans**

```bash
curl http://localhost:8000/api/clusters/kmeans
```

**GET /api/clusters/kmeans/2/analyze** (LLM root-cause)

```bash
curl http://localhost:8000/api/clusters/kmeans/2/analyze
```

## Sample Output

### POST /api/analyze Response

```json
{
  "total_records": 500,
  "anomalies_detected": 0,
  "kmeans_summary": [
    {
      "cluster": 0,
      "size": 151,
      "dominant_failure_code": 0,
      "mean_duration_ms": 1129.92,
      "anomaly_rate_pct": 0.0,
      "top_error_messages": [
        "Voltage rail dropout below minimum threshold"
      ]
    },
    {
      "cluster": 2,
      "size": 59,
      "dominant_failure_code": 202,
      "mean_duration_ms": 3041.96,
      "anomaly_rate_pct": 0.0,
      "top_error_messages": [
        "Ripple voltage exceeds specification",
        "Voltage rail dropout below minimum threshold",
        "Intermittent open circuit on signal path"
      ]
    }
  ],
  "dbscan_summary": [
    {"cluster": -1, "size": 156, "dominant_failure_code": 0, "mean_duration_ms": 2536.91, "anomaly_rate_pct": 0.0, "top_error_messages": ["Thermal runaway detected at junction", "Voltage rail dropout below minimum threshold"]},
    {"cluster": 0, "size": 325, "dominant_failure_code": 0, "mean_duration_ms": 2472.0, "anomaly_rate_pct": 0.0, "top_error_messages": []}
  ],
  "chart_path": "data/cluster_report.png",
  "root_cause_analysis": [
    {
      "hypotheses": [
        {
          "rank": 1,
          "root_cause": "Defective voltage regulator causing ripple on supply rail",
          "confidence": "high",
          "rationale": "Cluster shows dominant failure code 202 with voltage-related error messages...",
          "triage_action": "Replace voltage regulator module on affected batch and re-run sweep tests"
        },
        {
          "rank": 2,
          "root_cause": "PCB trace impedance mismatch on VCC plane",
          "confidence": "medium",
          "rationale": "Voltage dropout errors concentrated in voltage_sweep test type...",
          "triage_action": "Inspect PCB layout for VCC trace width and add decoupling capacitors"
        },
        {
          "rank": 3,
          "root_cause": "ESD damage to input protection diodes",
          "confidence": "low",
          "rationale": "Intermittent failures observed alongside voltage anomalies...",
          "triage_action": "Audit ESD handling procedures on assembly line"
        }
      ],
      "cluster": 2
    }
  ]
}
```

### GET /api/failures Response (truncated)

```json
[
  {
    "test_id": "TST-0004",
    "component_id": "COMP-D-001",
    "timestamp": "2025-06-01T08:09:27",
    "test_type": "continuity",
    "failure_code": 404,
    "error_message": "Resistance fluctuation beyond tolerance",
    "duration_ms": 1863.49,
    "temperature_c": 59.66,
    "voltage_v": 9.66,
    "cluster_kmeans": 3,
    "is_anomaly": false
  }
]
```

## Data Format

Input CSV must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `test_id` | str | Unique test identifier |
| `component_id` | str | Component under test |
| `timestamp` | str | ISO 8601 formatted timestamp |
| `test_type` | str | One of: thermal_stress, voltage_sweep, burn_in, continuity, functional |
| `failure_code` | int | 0 = pass, 101/202/303/404 = failure codes |
| `error_message` | str | Error description (empty string if passed) |
| `duration_ms` | float | Test duration in milliseconds |
| `temperature_c` | float | Ambient temperature during test |
| `voltage_v` | float | Supply voltage during test |
