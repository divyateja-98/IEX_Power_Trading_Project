"""Feast feature definitions for IEX power trading forecasts."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


iex_power_source = FileSource(
    name="iex_power_feature_engineered_source",
    path="../data/processed/feature_engineered.parquet",
    timestamp_field="event_timestamp",
    file_format=ParquetFormat(),
)

timestamp = Entity(
    name="timestamp",
    join_keys=["timestamp"],
    value_type=ValueType.STRING,
    description="ISO timestamp key for one IEX hourly observation.",
)

date = Entity(
    name="date",
    join_keys=["date"],
    value_type=ValueType.STRING,
    description="Trading date key in YYYY-MM-DD format.",
)

hour = Entity(
    name="hour",
    join_keys=["hour"],
    value_type=ValueType.INT64,
    description="Trading hour key from the IEX source data.",
)

iex_power_features = FeatureView(
    name="iex_power_features",
    entities=[timestamp, date, hour],
    ttl=timedelta(days=365),
    schema=[
        Field(name="mcp", dtype=Float32),
        Field(name="demand", dtype=Float32),
        Field(name="renewable_generation", dtype=Float32),
        Field(name="load_forecast", dtype=Float32),
        Field(name="rolling_mean_24", dtype=Float32),
        Field(name="rolling_std_24", dtype=Float32),
        Field(name="lag_1", dtype=Float32),
        Field(name="lag_2", dtype=Float32),
        Field(name="lag_24", dtype=Float32),
        Field(name="lag_48", dtype=Float32),
        Field(name="temperature", dtype=Float32),
        Field(name="humidity", dtype=Float32),
        Field(name="cloud_cover", dtype=Float32),
        Field(name="wind_speed", dtype=Float32),
        Field(name="solar_radiation", dtype=Float32),
        Field(name="weekday", dtype=Int64),
        Field(name="weekend_flag", dtype=Int64),
        Field(name="timestamp", dtype=String),
        Field(name="date", dtype=String),
    ],
    source=iex_power_source,
    online=True,
    description="Canonical hourly IEX power trading features for forecasting.",
)
