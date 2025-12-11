import { Button } from "@/components/ui/button";
import { Grid3X3, TrendingUp } from "lucide-react";

interface ControlPanelProps {
  onOpenLBCS: () => void;
  onOpenMaskedMAEFocalLoss: () => void;
}

const ControlPanel = ({
  onOpenLBCS,
  onOpenMaskedMAEFocalLoss,
}: ControlPanelProps) => {
  return (
    <div className="control-panel p-4">
      <div className="max-w-4xl mx-auto flex flex-wrap items-center justify-center gap-4">
        <Button variant="control" onClick={onOpenLBCS} className="gap-2">
          <Grid3X3 className="w-4 h-4" />
          Visualize LBCS Grid
        </Button>
        <Button
          variant="control"
          onClick={onOpenMaskedMAEFocalLoss}
          className="gap-2"
        >
          <TrendingUp className="w-4 h-4" />
          MaskedMAEFocalLoss
        </Button>
      </div>
    </div>
  );
};

export default ControlPanel;
