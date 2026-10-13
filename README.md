# NB Farm forecasting

Separate Python service used by Laravel (`nb_system`).

**Primary model:** `three_year_temperature`  
- Same calendar day across up to 3 years of harvest history  
- Tomorrow’s temperature shifts Small ↔ Large  
- Returns total + size breakdown  

**Legacy model:** `linear_regression` (old 7-day OLS), still available via `"model": "linear_regression"`.

This service does **not** connect to MySQL. Laravel sends history + temperatures.

## Run locally

```bash
cd forecasting
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m flask --app api.index run --port 8001
```

Health check: http://127.0.0.1:8001/health

## Endpoint

`POST /forecast` (also `/api/forecast`)

Header (if `FORECAST_API_KEY` is set):

```
X-Forecast-Key: change-me
```

Body:

```json
{
  "model": "three_year_temperature",
  "history": [
    {
      "date": "2023-10-05",
      "eggs": 6000,
      "small": 900,
      "medium": 2100,
      "large": 2400,
      "extra_large": 600,
      "cracked": 0
    }
  ],
  "predict_dates": ["2026-10-05"],
  "temperatures": { "2026-10-05": 26.5 },
  "history_years": 3,
  "neutral_temperature": 28,
  "temp_shift_per_degree": 0.015,
  "temp_shift_cap": 0.20,
  "eggs_per_tray": 30
}
```

## Tests

```bash
cd forecasting
python -m unittest tests.test_regression
```

## Vercel

Set the Vercel project root to `forecasting/`. Add `FORECAST_API_KEY` in Vercel env vars. Point Laravel `FORECAST_API_URL` at the deployment URL (no trailing slash).
