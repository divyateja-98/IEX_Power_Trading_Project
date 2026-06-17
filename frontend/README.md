# IEX Power Forecasting React Frontend

React UI for the FastAPI forecasting service.

## Run locally

```powershell
cd frontend
npm install
npm run dev
```

The app reads the API base URL from `VITE_FASTAPI_BASE_URL`.

```powershell
Copy-Item .env.example .env
```

Default API:

```text
http://127.0.0.1:8002
```

## Build

```powershell
npm run build
npm run preview
```

## Docker

From the repository root:

```powershell
docker build -f frontend/Dockerfile -t iex-power-forecasting-frontend .
docker run --rm -p 8080:8080 iex-power-forecasting-frontend
```

The root Compose stack also includes `react-frontend` on port `8080`.
