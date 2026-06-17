export const API_BASE_URL =
  import.meta.env.VITE_FASTAPI_BASE_URL || "http://127.0.0.1:8002";

export const DEFAULT_FEATURES = {
  demand: 6771.7,
  renewable_generation: 0,
  load_forecast: 5054.1,
  hour: 1,
  temperature: 16.335714285714285,
  humidity: 75.14285714285714,
  cloud_cover: 10.071428571428571,
  wind_speed: 8.285714285714285,
  solar_radiation: 0,
  weekday: 1,
  weekend_flag: 0,
  lag_1: 2938.66,
  lag_2: 3735.05,
  lag_24: 2543.58,
  lag_48: 2890.56,
  rolling_mean_24: 5818.813333333333,
  rolling_std_24: 3171.596210228571,
};

export const FEATURE_GROUPS = [
  {
    title: "Market",
    fields: [
      { name: "demand", label: "Demand", step: 100 },
      {
        name: "renewable_generation",
        label: "Renewable Generation",
        step: 100,
      },
      { name: "load_forecast", label: "Load Forecast", step: 100 },
      { name: "hour", label: "Hour", min: 0, max: 23, step: 1 },
    ],
  },
  {
    title: "Weather",
    fields: [
      { name: "temperature", label: "Temperature", step: 0.1 },
      { name: "humidity", label: "Humidity", min: 0, max: 100, step: 0.1 },
      { name: "cloud_cover", label: "Cloud Cover", min: 0, max: 100, step: 0.1 },
      { name: "wind_speed", label: "Wind Speed", min: 0, step: 0.1 },
      { name: "solar_radiation", label: "Solar Radiation", min: 0, step: 1 },
    ],
  },
  {
    title: "Calendar",
    fields: [
      { name: "weekday", label: "Weekday", min: 0, max: 6, step: 1 },
      { name: "weekend_flag", label: "Weekend Flag", min: 0, max: 1, step: 1 },
    ],
  },
  {
    title: "History",
    fields: [
      { name: "lag_1", label: "Lag 1", step: 10 },
      { name: "lag_2", label: "Lag 2", step: 10 },
      { name: "lag_24", label: "Lag 24", step: 10 },
      { name: "lag_48", label: "Lag 48", step: 10 },
      { name: "rolling_mean_24", label: "Rolling Mean 24", step: 10 },
      { name: "rolling_std_24", label: "Rolling Std 24", min: 0, step: 10 },
    ],
  },
];
