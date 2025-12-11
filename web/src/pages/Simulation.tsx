import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import MatrixVisualization from "@/components/MatrixVisualization";
import ControlPanel from "@/components/ControlPanel";

import useTrafficStream from "@/hooks/useTrafficStream";

const Simulation = () => {
  const navigate = useNavigate();

  const {
    rawMatrix,
    enrichedMatrix,
    start,
    setSpeed,
    pause,
    resume,
    restart,
  } = useTrafficStream();


  const [started, setStarted] = useState(false);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeedState] = useState(0.2); 

    const isWaitingForData =
    started &&
    rawMatrix.length === 0 &&
    enrichedMatrix.length === 0;


  const handleSpeedChange = (value: number) => {
    if (value > 0) {
      setSpeedState(value);
      setSpeed(value);
    }
  };

  // Toggle pause/resume.
  const togglePause = () => {
    if (paused) {
      resume();
      setPaused(false);
    } else {
      pause();
      setPaused(true);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-card/50 backdrop-blur-sm border-b border-border p-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/")}
              className="gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Experiment
            </Button>
            <h1 className="text-lg font-semibold">
              Traffic Prediction{" "}
              <span className="text-primary">
                {started ? (paused ? "Paused" : "Active") : "Idle"}
              </span>
            </h1>
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  !started
                    ? "bg-muted"
                    : paused
                    ? "bg-muted"
                    : "bg-accent animate-pulse"
                }`}
              />
              <span className="text-sm text-muted-foreground">
                {!started ? "Idle" : paused ? "Paused" : "Running"}
              </span>
            </div>
          </div>

          {/* Controls: speed, pause/resume, restart, start */}
          <div className="flex items-center gap-3">
            {!started && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  start(speed);
                  setStarted(true);
                  setPaused(false);
                }}
              >
                Start
              </Button>
            )}
            {started && (
              <>
                <label htmlFor="speed-input" className="text-xs text-muted-foreground">
                  Speed (s/frame)
                </label>
                <input
                  id="speed-input"
                  type="number"
                  min="0.05"
                  step="0.05"
                  value={speed}
                  onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
                  className="w-20 px-2 py-1 border border-border rounded-md bg-card text-sm"
                />
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={togglePause}
                  className="gap-1"
                >
                  {paused ? "Resume" : "Pause"}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    // Restart resets the backend iterator and clears the matrices.
                    restart(speed);
                    setPaused(false);
                    setSpeed(speed);
                  }}
                >
                  Restart
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Content - Split Screen */}
     
            {/* Main Content */}
      {isWaitingForData ? (
        <main className="flex-1 p-4 animate-fade-in flex items-center justify-center">
          <div className="text-muted-foreground text-sm">
            Waiting for backend data...
          </div>
        </main>
      ) : (
        <main className="flex-1 grid md:grid-cols-2 gap-4 p-4 animate-fade-in">
          <MatrixVisualization
            title="Raw Model Prediction"
            data={rawMatrix && rawMatrix.length > 0 ? rawMatrix : [[0]]}
            className="min-h-[400px]"
          />
          <MatrixVisualization
            title="Enriched Model Prediction"
            data={enrichedMatrix && enrichedMatrix.length > 0 ? enrichedMatrix : [[0]]}
            className="min-h-[400px]"
          />
        </main>
      )}

       {/*
      <main className="flex-1 p-4 animate-fade-in flex items-center justify-center">
  <div className="text-muted-foreground text-sm">
    Matrix visualization temporarily disabled for debugging.
  </div>
</main>
*/}
      {/* Legend */}
      <div className="px-4 pb-2">
        <div className="max-w-7xl mx-auto text-xs text-muted-foreground flex flex-wrap gap-4 items-center">
          <span className="font-semibold mr-2">Legend:</span>

          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-sm border border-border"
              style={{ backgroundColor: "rgb(0,255,0)" }} // code 2
            />
            <span>
              Green – correct prediction with ≤ 0.05 error (≈12.75 speed units)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-sm border border-border"
              style={{ backgroundColor: "rgb(255,255,0)" }} // code 1
            />
            <span>Yellow – correct but 0 prediction</span>
          </div>

          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-sm border border-border"
              style={{ backgroundColor: "rgb(255,0,0)" }} // code 3
            />
            <span>Red – incorrect prediction</span>
          </div>
        </div>
      </div>


      {/* Control Panel */}

      <ControlPanel
        onOpenLBCS={() => window.open("/lbcs-map", "_blank")}
        onOpenMaskedMAEFocalLoss={() => window.open("/metric", "_blank")}
      />

    </div>
  );
};

export default Simulation;