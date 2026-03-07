## ML Model Drift Monitor with Auto-Retrain

This project implements a fully local, production-style ML drift monitoring and auto-retraining system.

Key technologies:

- Prefect for orchestration
- MLflow for experiment tracking and simple model registry
- Evidently for feature and prediction drift detection
- Plotly for dashboard visualizations
- scikit-learn for the baseline and challenger models

### Environment setup

1. Create and activate a virtual environment (no conda):

```bash
python -m venv .venv

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# On Windows (cmd.exe)
.venv\Scripts\activate.bat

# On Unix-like shells
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Project layout

- `src/ml_drift_monitor/` – core package (data, models, monitoring, orchestration, tracking, db, utils, dashboard)
- `artifacts/` – local data, models, logs, drift reports, job state DB, MLflow store (git-ignored)
- `tests/` – pytest test suite (no manual setup; data generated programmatically in fixtures)

### Build and test (no manual steps)

From the project root:

1. `python -m venv .venv` then activate and `pip install -r requirements.txt`.
2. Run the full test suite in one go: `pytest`

All data is generated programmatically; no external services are required (MLflow/Prefect run in-process or file-backed).

### Dashboard (FastAPI, no Streamlit)

Serve the monitoring dashboard as minimal HTML/JS via FastAPI:

```bash
uvicorn ml_drift_monitor.dashboard.server:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000 in a browser. The dashboard reads only from persisted artifacts (drift reports, job state, event log).

### Running the pipeline

To process a month (e.g. month 4) and optionally trigger retrain:

```python
from ml_drift_monitor.orchestration.prefect_flows import monitor_and_retrain_flow
monitor_and_retrain_flow(4)
```

Job state is persisted in SQLite (`artifacts/job_state.db`); cost metadata (inference time, estimated compute cost) is attached to retrain results and persisted.
