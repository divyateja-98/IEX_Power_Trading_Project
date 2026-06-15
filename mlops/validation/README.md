# Validation

Validation assets for productionizing the forecasting pipeline.

## Checks

- Raw input files exist and are readable.
- Required merge keys are present: `Date`, `Hour`.
- Cleaned data has no duplicate rows.
- Engineered data contains lag and rolling features.
- Model artifact exists after training.
- Recommendation output is one of `BUY NOW`, `WAIT`, or `SELL`.

Run the current smoke validator:

```powershell
python mlops/validation/smoke_check.py
```
