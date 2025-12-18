# Web application

The `web/` directory contains the frontend of the Traffic Prediction
Visualiser.  It is a single‑page application built with React and
TypeScript that connects to the backend’s WebSocket and HTTP
endpoints to display real‑time predictions and errors.  The frontend
does not perform any heavy computations itself; all model inference
and preprocessing happen on the server side.

## Communication with the backend

The frontend communicates with the backend via:

* A **WebSocket connection** to `/stream`.  Using the custom hook
  `useTrafficStream` (see `src/hooks/useTrafficStream.ts`), the
  application opens a WebSocket when the user presses “Play”.  It
  sends control messages (`start`, `pause`, `resume`, `goto`) and
  receives prediction messages from the backend.  Each message
  includes the timestamp, the aggregated prediction from both models and their respective losses.  The
  predictions are rendered by `MatrixVisualization.tsx`.
* An **HTTP request** to `/metrics/loss_history`.  The
  `Metric.tsx` page periodically polls this endpoint to retrieve the
  history of per‑frame losses and displays them in a chart.
