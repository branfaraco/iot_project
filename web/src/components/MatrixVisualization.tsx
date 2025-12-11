import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface MatrixVisualizationProps {
  title: string;
  data: number[][];
  className?: string;
}

const MatrixVisualization = ({
  title,
  data,
  className,
}: MatrixVisualizationProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Classification codes: 0–3
  const isClassCode = (value: number) =>
    Number.isInteger(value) && value >= 0 && value <= 3;

  // RGB colours for class codes:
  // 0 → white; 1 → yellow; 2 → green; 3 → red
  const classColors: Record<number, [number, number, number]> = {
    0: [255, 255, 255],
    1: [255, 255, 0],
    2: [0, 255, 0],
    3: [255, 0, 0],
  };

  const valueToRGB = (value: number): [number, number, number] => {
    if (isClassCode(value)) {
      return classColors[value] ?? [255, 255, 255];
    }

    // Continuous values: clamp to [0, 1] and map to a simple
    // white → cyan gradient for now.
    const v = Math.min(Math.max(value, 0), 1);
    const r = Math.round(255 * (1 - v));
    const g = Math.round(255 * v);
    const b = 255;
    return [r, g, b];
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!data || data.length === 0 || !data[0] || data[0].length === 0) {
      return;
    }

    const height = data.length;
    const width = data[0].length;

    // 1 pixel per matrix cell
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(width, height);
    const buf = imageData.data;

    let i = 0;
    for (let y = 0; y < height; y++) {
      const row = data[y];
      for (let x = 0; x < width; x++) {
        const [r, g, b] = valueToRGB(row[x]);
        buf[i++] = r;
        buf[i++] = g;
        buf[i++] = b;
        buf[i++] = 255; // alpha
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  const rows = data?.length || 0;
  const cols = rows > 0 ? data[0].length : 0;

  return (
    <div className={cn("matrix-container flex flex-col h-full", className)}>
      <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
        {title}
      </h3>

      <div className="flex-1 flex items-center justify-center overflow-auto">
        <canvas
          ref={canvasRef}
          className="w-full h-auto border border-border rounded-md"
          style={{ imageRendering: "pixelated" as const }}
        />
      </div>

      <div className="mt-2 text-xs text-muted-foreground font-mono">
        Dimensions: {rows} × {cols}
      </div>
    </div>
  );
};

export default MatrixVisualization;
