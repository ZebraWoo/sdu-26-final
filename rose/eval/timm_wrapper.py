import torch
import timm
from safetensors.torch import load_file
import logging

logger = logging.getLogger("dinov3")

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': False, 'num_classes': 0}, 
                 pool: bool = True,
                 weights_path: str | None = None):
        super().__init__()

        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
                
        if weights_path is not None:
            logger.info(f"Loading weights from: {weights_path}")
            state_dict = load_file(weights_path)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

            missing, unexpected = self.model.load_state_dict(new_sd, strict=False)

        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out