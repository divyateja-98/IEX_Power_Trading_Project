# Backend

The backend layer exposes the existing FastAPI service through the enterprise
layout.

## Run

```powershell
uvicorn backend.app.main:app --reload
```

The legacy command still works:

```powershell
uvicorn api.main:app --reload
```

## Current Capability

- Health endpoint: `/health`
- Forecasting, risk, and recommendation APIs can be added here while reusing
  the existing modules in `src/`, `xgboost_pipeline.py`,
  `risk_analytics.py`, and `recommendation_engine.py`.
