

import os
import torch

try:
    # Import the model architecture from the shared package.
    from shared.models.unet_base import UNet2D
except ImportError as e:
    raise ImportError(
        "Unable to import UNet2D from shared.models. Make sure the shared "
        "package is on the Python path and that the model definitions "
        "have been moved out of your training scripts."
    ) from e


def load_raw_model(
    device: torch.device,
    parameters_dir: str | None = None,
    weights_name: str | None = None,
) -> UNet2D:
    """
    Create and return the baseline U-Net model with pretrained weights.

    Parameters
    ----------
    device : torch.device
        The device on which the model should reside (CPU or CUDA).
    parameters_dir : str, optional
        Path to the directory containing the weight file. If not provided,
        the environment variable MODEL_PARAMETERS_DIR or a default
        backend/parameters directory will be used.
    weights_name : str, optional
        Name of the weight file inside parameters_dir. If not provided,
        the environment variable RAW_MODEL_WEIGHTS or 'raw_model.pth'
        will be used.

    Returns
    -------
    UNet2D
        The loaded model in eval mode.
    """
    # Determine where to load weights from.
    if parameters_dir is None:
        parameters_dir = os.environ.get("MODEL_PARAMETERS_DIR")
        if parameters_dir is None:
            parameters_dir = os.path.join(os.path.dirname(__file__), "..", "parameters")
    parameters_dir = os.path.abspath(parameters_dir)

    # Decide which filename to use.
    if weights_name is None:
        weights_name = os.environ.get("RAW_MODEL_WEIGHTS", "raw_model.pth")

    weight_path = os.path.join(parameters_dir, weights_name)

    # Define the model. These channel counts must match the training script.
    history_steps = 12
    traffic_channels = 8
    future_steps = 4
    in_channels = history_steps * traffic_channels
    out_channels = future_steps
    base_channels = 16
    model = UNet2D(in_channels=in_channels, out_channels=out_channels, base_ch=base_channels)

    # Load weights if present.
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(
            f"Warning: raw model weights not found at {weight_path}. "
            "The model will be randomly initialised."
        )

    model = model.to(device)
    model.eval()
    return model
