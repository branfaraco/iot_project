import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Play, FlaskConical, Cpu, Zap } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 animate-fade-in">
      <div className="max-w-3xl w-full text-center space-y-8">
        {/* Icon */}
        <div className="flex justify-center">
          <div className="relative">
            <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center border border-primary/30">
              <FlaskConical className="w-10 h-10 text-primary" />
            </div>
            <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-accent animate-pulse" />
          </div>
        </div>

        {/* Title */}
        <div className="space-y-4">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground tracking-tight">
            Traffic Prediction
            <span className="text-primary"> Experiment</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-xl mx-auto">
            This experiment streams real‑time  traffic predictions on a two‑dimensional grid. We compare a baseline U‑Net model against an enriched FiLM‑conditioned model that incorporates local weather data and land‑use (LBCS) context.
          </p>
        </div>

        {/* Description Card */}
        <div className="bg-card border border-border rounded-xl p-6 text-left space-y-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            Experiment Description
          </h2>
          <p className="text-muted-foreground leading-relaxed">
            We ingest a history of 12 traffic frames and produce four future predictions. The baseline model uses only traffic data, while the enriched model uses FiLM to condition the U‑Net on weather vectors and land‑use categories. This page summarizes the models you’ll explore in the simulation.
          </p>
          <div className="grid md:grid-cols-2 gap-4 pt-2">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                <span className="text-primary font-mono text-sm">R</span>
              </div>
              <div>
                <h3 className="font-medium text-sm">Raw Model</h3>
                <p className="text-xs text-muted-foreground">U‑Net trained on traffic history only.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center shrink-0">
                <span className="text-accent font-mono text-sm">E</span>
              </div>
              <div>
                <h3 className="font-medium text-sm">Enriched Model</h3>
                <p className="text-xs text-muted-foreground">FiLM‑conditioned U‑Net using weather and LBCS context.</p>
              </div>
            </div>
          </div>
        </div>


        {/* CTA Button */}
        <Button variant="simulation" size="xl" onClick={() => navigate("/simulation")} className="mt-4">
          <Play className="w-5 h-5" />
          Run Simulation
        </Button>
      </div>
    </div>
  );
};

export default Index;