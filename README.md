# Manufacturing Failure Analyzer

A Python-based tool for analyzing hardware test logs and detecting failure patterns in manufacturing data.

## Features

- Load and validate hardware test logs from CSV
- Analyze failure patterns using clustering (scikit-learn)
- REST API for querying failure data (FastAPI)
- SQLite-backed persistence for test results

## Project Structure

```
.
├── api/             # FastAPI route handlers
│   └── routes.py
├── data/            # Raw CSV log files
│   └── sample_test_log.csv
├── src/             # Core analysis logic
│   └── data_loader.py
├── main.py          # Application entry point
├── requirements.txt # Python dependencies
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the API server:

```bash
python main.py
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

## Data Format

Input CSV must contain the following columns:

| Column        | Type   | Description                      |
|---------------|--------|----------------------------------|
| test_id       | str    | Unique test identifier           |
| component_id  | str    | Component under test             |
| timestamp     | str    | ISO 8601 timestamp               |
| test_type     | str    | Type of test performed           |
| failure_code  | int    | Failure code (0 = pass)          |
| error_message | str    | Error description (NaN if pass)  |
| duration_ms   | float  | Test duration in milliseconds    |
| temperature_c | float  | Ambient temperature during test  |
| voltage_v     | float  | Supply voltage during test       |
