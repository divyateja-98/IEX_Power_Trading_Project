import { useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  Gauge,
  Loader2,
  RotateCcw,
  Server,
  Zap,
} from "lucide-react";
import { predictPowerPrice } from "./api";
import { API_BASE_URL, DEFAULT_FEATURES, FEATURE_GROUPS } from "./constants";

function App() {
  const [features, setFeatures] = useState(DEFAULT_FEATURES);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const confidencePercent = useMemo(() => {
    if (!prediction) return "0.0%";
    return `${(Number(prediction.confidence || 0) * 100).toFixed(1)}%`;
  }, [prediction]);

  async function handleSubmit(event) {
    event.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      const numericFeatures = Object.fromEntries(
        Object.entries(features).map(([key, value]) => [key, Number(value)]),
      );
      const result = await predictPowerPrice(numericFeatures);
      setPrediction(result);
    } catch (caughtError) {
      setError(caughtError.message);
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  }

  function updateFeature(name, value) {
    setFeatures((current) => ({
      ...current,
      [name]: value,
    }));
  }

  function resetFeatures() {
    setFeatures(DEFAULT_FEATURES);
    setPrediction(null);
    setError("");
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">IEX Power Trading</p>
            <h1>Forecast MCP</h1>
          </div>
          <div className="api-pill">
            <Server size={16} aria-hidden="true" />
            <span>{API_BASE_URL}</span>
          </div>
        </header>

        <div className="content-grid">
          <form className="forecast-form" onSubmit={handleSubmit}>
            <div className="form-header">
              <div>
                <h2>Prediction Inputs</h2>
                <p>{Object.keys(features).length} features</p>
              </div>
              <button
                className="icon-button"
                type="button"
                onClick={resetFeatures}
                title="Reset inputs"
                aria-label="Reset inputs"
              >
                <RotateCcw size={18} aria-hidden="true" />
              </button>
            </div>

            <div className="groups">
              {FEATURE_GROUPS.map((group) => (
                <fieldset className="field-group" key={group.title}>
                  <legend>{group.title}</legend>
                  <div className="field-grid">
                    {group.fields.map((field) => (
                      <label className="input-field" key={field.name}>
                        <span>{field.label}</span>
                        <input
                          type="number"
                          value={features[field.name]}
                          min={field.min}
                          max={field.max}
                          step={field.step ?? 1}
                          onChange={(event) =>
                            updateFeature(field.name, event.target.value)
                          }
                        />
                      </label>
                    ))}
                  </div>
                </fieldset>
              ))}
            </div>

            <div className="action-row">
              <button className="primary-button" type="submit" disabled={isLoading}>
                {isLoading ? (
                  <Loader2 className="spin" size={18} aria-hidden="true" />
                ) : (
                  <Zap size={18} aria-hidden="true" />
                )}
                <span>{isLoading ? "Running" : "Predict"}</span>
              </button>
            </div>
          </form>

          <aside className="results-panel">
            <div className="result-card primary-result">
              <div className="card-label">
                <Gauge size={18} aria-hidden="true" />
                <span>Prediction</span>
              </div>
              <strong>
                {prediction ? formatNumber(prediction.prediction) : "--"}
              </strong>
              <small>Rs/MWh</small>
            </div>

            <div className="result-card">
              <div className="card-label">
                <Activity size={18} aria-hidden="true" />
                <span>Confidence</span>
              </div>
              <strong>{prediction ? confidencePercent : "--"}</strong>
              <div className="confidence-track" aria-hidden="true">
                <div
                  style={{
                    width: prediction
                      ? `${Math.max(0, Math.min(100, prediction.confidence * 100))}%`
                      : "0%",
                  }}
                />
              </div>
            </div>

            <div className="model-panel">
              <h2>Model</h2>
              <dl>
                <div>
                  <dt>Version</dt>
                  <dd>{prediction ? modelVersion(prediction.model_uri) : "--"}</dd>
                </div>
                <div>
                  <dt>Source</dt>
                  <dd>{prediction?.model_source || "--"}</dd>
                </div>
                <div>
                  <dt>URI</dt>
                  <dd>{prediction?.model_uri || "--"}</dd>
                </div>
                <div>
                  <dt>Timestamp</dt>
                  <dd>
                    {prediction?.timestamp
                      ? new Date(prediction.timestamp).toLocaleString()
                      : "--"}
                  </dd>
                </div>
              </dl>
            </div>

            {error ? (
              <div className="error-panel" role="alert">
                <AlertCircle size={18} aria-hidden="true" />
                <span>{error}</span>
              </div>
            ) : null}
          </aside>
        </div>
      </section>
    </main>
  );
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  }).format(Number(value || 0));
}

function modelVersion(modelUri) {
  if (!modelUri) return "--";
  const parts = String(modelUri).split("/").filter(Boolean);
  return parts.at(-1) || modelUri;
}

export default App;
