# Deployment

Deployment assets describe how to package and run the backend and frontend.

## Local Services

Backend:

```powershell
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
streamlit run frontend/Home.py --server.port 8501
```

## Production Notes

- Package data/model artifacts through DVC remote storage.
- Pull DVC artifacts during release preparation or container startup.
- Expose FastAPI behind an ingress or API gateway.
- Keep Streamlit as an internal operator dashboard unless hardened for public
  access.
