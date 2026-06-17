import { API_BASE_URL } from "./constants";

export async function predictPowerPrice(features) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      request_id: `react-${new Date().toISOString()}`,
      features,
    }),
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = payload?.detail || response.statusText || "Prediction failed";
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }

  return payload;
}
