"""IEX data ingestion utilities.

Stubs included for project scaffolding. Implement connectors to IEX Cloud
or other providers as needed.
"""

from typing import Any


def fetch_iex_prices(symbol: str, start: str, end: str) -> Any:
    """Fetch price/time-series data for `symbol` between `start` and `end`.

    Return a pandas DataFrame-like object. Replace with real implementation.
    """
    raise NotImplementedError("Implement IEX ingestion logic here")
