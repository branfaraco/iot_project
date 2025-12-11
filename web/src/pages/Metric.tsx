// src/pages/Metric.tsx
//
// Live view of MaskedMAE-Focal loss over recently streamed frames.
// Backend endpoint: GET /metrics/loss_history
// Response shape:
// {
//   "metric": "masked_mae_focal",
//   "history": [
//     { "frame_index": 0, "raw": <number|null>, "enriched": <number|null> },
//     ...
//   ]
// }

import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { AlertCircle } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface LossPoint {
  frame_index: number;
  raw: number | null;
  enriched: number | null;
}

interface LossHistoryResponse {
  metric: string;
  history: LossPoint[];
}

const Metric: React.FC = () => {
  const [history, setHistory] = useState<LossPoint[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const apiBase =
      (import.meta as any).env.VITE_BACKEND_API_URL ?? "http://localhost:8000";

    let cancelled = false;

    const fetchHistory = () => {
      fetch(`${apiBase}/metrics/loss_history`)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then((json: LossHistoryResponse | any) => {
          if (cancelled) return;

          if (Array.isArray(json?.history)) {
            setHistory(json.history as LossPoint[]);
            setError(null);
          } else if (json?.error) {
            setError(json.error as string);
          } else {
            setError("Unexpected response structure");
          }
        })
        .catch((err) => {
          console.error(err);
          if (!cancelled) setError("Failed to load loss history");
        });
    };

    // Initial fetch + polling
    fetchHistory();
    const id = setInterval(fetchHistory, 1000); // poll every 1 second

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const latest = history.length ? history[history.length - 1] : null;

  if (error && !history.length) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex items-center gap-2 text-muted-foreground">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (!history.length) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-muted-foreground">
          Waiting for streamed frames… open the Simulation and press Start.
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 flex flex-col gap-6 animate-fade-in">
      <h1 className="text-2xl font-semibold text-center">
        Masked MAE-Focal Loss (Live)
      </h1>

      {latest && (
        <div className="flex justify-center gap-8 text-sm text-muted-foreground">
          <div>
            <span className="font-medium text-foreground">Frame:</span>{" "}
            {latest.frame_index}
          </div>
          <div>
            <span className="font-medium text-foreground">Raw:</span>{" "}
            {latest.raw !== null ? latest.raw.toFixed(4) : "—"}
          </div>
          <div>
            <span className="font-medium text-foreground">Enriched:</span>{" "}
            {latest.enriched !== null ? latest.enriched.toFixed(4) : "—"}
          </div>
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-6">
        {/* Raw model chart */}
        <Card className="p-4">
          <h2 className="text-lg font-medium mb-2">Raw model loss</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <XAxis dataKey="frame_index" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="raw"
                  name="Raw loss"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Enriched model chart */}
        <Card className="p-4">
          <h2 className="text-lg font-medium mb-2">Enriched model loss</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <XAxis dataKey="frame_index" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="enriched"
                  name="Enriched loss"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <p className="text-xs text-muted-foreground text-center max-w-xl mx-auto">
        The curves update roughly once per second and reflect the most
        recently streamed frames from the Simulation. Lower is better.
      </p>
    </div>
  );
};

export default Metric;
