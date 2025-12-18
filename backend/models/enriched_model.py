

import os
import torch

from shared.models.unet_film import UNet2D_FiLM



def load_enriched_model(
    device: torch.device,
    weather_dim: int | None = None,
    parameters_dir: str | None = None,
    weights_name: str | None = None,
) -> UNet2D_FiLM:
    """
    Create and return the FiLM-conditioned U-Net model with pretrained weights.
    """
    if weather_dim is None:
        weather_dim = 12

    if parameters_dir is None:
        parameters_dir = os.environ.get("MODEL_PARAMETERS_DIR")
        if parameters_dir is None:
            parameters_dir = os.path.join(os.path.dirname(__file__), "..", "parameters")
    parameters_dir = os.path.abspath(parameters_dir)

    if weights_name is None:
        weights_name = os.environ.get("ENRICHED_MODEL_WEIGHTS", "enriched_model.pth")

    weight_path = os.path.join(parameters_dir, weights_name)

    history_steps = 12
    traffic_channels = 8
    lbcs_channels = 9
    future_steps = 4
    in_channels = history_steps * (traffic_channels + lbcs_channels)
    out_channels = future_steps
    base_channels = 16
    model = UNet2D_FiLM(
        in_channels=in_channels,
        out_channels=out_channels,
        weather_dim=weather_dim,
        base_ch=base_channels,
    )

    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(
            f"Warning: enriched model weights not found at {weight_path}. "
            "The model will be randomly initialised."
        )

    model = model.to(device)
    model.eval()
    return model
