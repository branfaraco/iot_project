import { useEffect, useRef, useState } from "react";

/**
 * Hook for connecting to the backend WebSocket stream and returning
 * matrices for the baseline (raw) and enriched models. It also exposes
 * helpers for adjusting the streaming speed, pausing/resuming the stream
 * and restarting the connection.
 *
 * The backend must expose a `/stream` WebSocket endpoint as defined in
 * the FastAPI application. Messages emitted by the backend should be
 * JSON objects with `raw` and `enriched` keys containing nested lists of
 * shape (F x H x W). The hook extracts the first future step for each
 * model and converts it into a 2‑D array of numbers.
 */
export function useTrafficStream() {

  const [rawMatrix, setRawMatrix] = useState<number[][]>([]);
  const [enrichedMatrix, setEnrichedMatrix] = useState<number[][]>([]);

  // Reference to the active WebSocket.  We avoid connecting on mount so
  // that the consumer can decide when to start streaming.
  const socketRef = useRef<WebSocket | null>(null);

  /**
   * Open a new WebSocket connection to the backend and set up
   * handlers.  If a connection already exists it will be closed.
   */
const connect = (initialSpeed?: number) => {
  // Close any existing connection before opening a new one.
  if (socketRef.current) {
    socketRef.current.close();
  }
  const wsUrl =
    (import.meta as any).env.VITE_BACKEND_WS_URL ||
    (import.meta as any).env.VITE_WS_URL ||
    "ws://localhost:8000/stream";
  const ws = new WebSocket(wsUrl);
  socketRef.current = ws;

  ws.onopen = () => {
    console.log("[client] WebSocket opened");
    if (initialSpeed !== undefined) {
      console.log("[client] auto-starting stream with speed", initialSpeed);
      ws.send(JSON.stringify({ type: "start", speed: initialSpeed }));
    }
  };


    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // The backend should send classification matrices under
        // `raw_classes` and `enriched_classes`.  If these keys are
        // present, use them directly.  Otherwise, fall back to the
        // first future frame from `raw` and `enriched` arrays.
        if (data.raw_classes && Array.isArray(data.raw_classes)) {
          setRawMatrix(data.raw_classes);
        } else if (data.raw && Array.isArray(data.raw)) {
          const frame = data.raw[0];
          if (Array.isArray(frame)) {
            setRawMatrix(frame);
          }
        }
        if (data.enriched_classes && Array.isArray(data.enriched_classes)) {
          setEnrichedMatrix(data.enriched_classes);
        } else if (data.enriched && Array.isArray(data.enriched)) {
          const frame = data.enriched[0];
          if (Array.isArray(frame)) {
            setEnrichedMatrix(frame);
          }
        }
      } catch (error) {
        console.error("Failed to parse websocket message", error);
      }
    };

    ws.onclose = () => {
      // Clear matrices on disconnect to indicate no data is being
      // received.
      setRawMatrix([]);
      setEnrichedMatrix([]);
    };
  };

  /**
   * Start streaming.  This simply calls `connect`.  Provided as a
   * clearer alias for consumers.
   */
// AFTER: start now takes the current simulation speed
const start = (initialSpeed: number) => {
  console.log("[client] start() called with speed", initialSpeed);
  const ws = socketRef.current;
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.log("[client] WS already open, sending start");
    ws.send(JSON.stringify({ type: "start", speed: initialSpeed }));
  } else {
    console.log("[client] WS not open, connecting with auto-start");
    connect(initialSpeed);
  }
};


const setSpeed = (value: number) => {
  console.log("[client] setSpeed(", value, ") readyState =", socketRef.current?.readyState);
  if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
    console.log("[client] setSpeed: socket not open");
    return;
  }
  socketRef.current.send(JSON.stringify({ type: "set_speed", value }));
};

const pause = () => {
  console.log("[client] pause() called, readyState =", socketRef.current?.readyState);
  if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
    console.log("[client] pause: socket not open");
    return;
  }
  socketRef.current.send(JSON.stringify({ type: "pause" }));
};

const resume = () => {
  console.log("[client] resume() called, readyState =", socketRef.current?.readyState);
  if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
    console.log("[client] resume: socket not open");
    return;
  }
  socketRef.current.send(JSON.stringify({ type: "resume" }));
};


const restart = (initialSpeed: number) => {
  console.log("[client] restart() called with speed", initialSpeed);
  connect(initialSpeed);
};



  return {
    rawMatrix,
    enrichedMatrix,
    start,
    setSpeed,
    pause,
    resume,
    restart,
  };
}

export default useTrafficStream;