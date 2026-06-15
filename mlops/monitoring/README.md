# Monitoring

Monitoring covers production data quality, forecasting quality, and API health.

## Signals

- Input schema drift for raw market and weather files
- Missing-value rate by feature group
- MCP target distribution drift
- Forecast error metrics when actual MCP becomes available
- Recommendation distribution by action
- Backend health endpoint availability

## Current Runtime Check

```powershell
python mlops/validation/smoke_check.py
```
