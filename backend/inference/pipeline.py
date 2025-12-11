

from typing import Dict, Any

import torch

from backend.utils.preprocess import prepare_inputs_raw, prepare_inputs_enriched


class InferencePipeline:
    """
    Holds two models and runs inference on a single sample.

    The sample should be a dictionary with at least the key
    "x_raw" containing the baseline model input, and optionally
    "x_enriched" and "weather_vec" for the enriched model.
    """
    def __init__(self, raw_model: torch.nn.Module, enriched_model: torch.nn.Module, device: torch.device):
        self.raw_model = raw_model.to(device)
        self.enriched_model = enriched_model.to(device)
        self.device = device

        self.raw_model.eval()
        self.enriched_model.eval()

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run both models on the provided sample and return the
        predictions. The input dictionary is expected to already
        contain preformatted arrays under the keys "x_raw" and
        "x_enriched", and a weather vector under "weather_vec". If
        `x_enriched` or `weather_vec` are missing, the enriched model
        is skipped.
        """
        results: Dict[str, Any] = {}

        # Baseline prediction
        if "x_raw" in sample:
            x_raw = sample["x_raw"]
            x_raw_tensor = prepare_inputs_raw(x_raw).to(self.device)
            with torch.no_grad():
                #print("x_raw shape:", x_raw_tensor.shape)
                pred_raw = self.raw_model(x_raw_tensor)  # (1, F, H, W)
            results["raw"] = pred_raw.squeeze(0).cpu().numpy()
        else:
            results["raw"] = None

        # Enriched prediction
        if "x_enriched" in sample and "weather_vec" in sample:
            x_enr = sample["x_enriched"]
            w_vec = sample["weather_vec"]
            x_enr_tensor = prepare_inputs_enriched(x_enr).to(self.device)
            if isinstance(w_vec, torch.Tensor):
                w_tensor = w_vec.to(self.device).unsqueeze(0)
            else:
                w_tensor = torch.tensor(w_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                #print("x_enriched shape:", x_enr_tensor.shape)
                pred_enr = self.enriched_model(x_enr_tensor, w_tensor)
            results["enriched"] = pred_enr.squeeze(0).cpu().numpy()
        else:
            results["enriched"] = None

        return results