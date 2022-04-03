import sys
import torch
import einops
import subprocess
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf

from pathlib import Path
from efficientnet_pytorch import EfficientNet

from ..utils.base_model import BaseModel

geoloc_path = Path(__file__).parent / "../../third_party/geoloc/"
sys.path.append(str(geoloc_path))


class Residual(nn.Module):

    def __init__(self, dims=1792, hidden_dims=4096):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, dims)

    def forward(self, x, labels=None):
        x_tr = self.fc2(F.relu(self.fc1(x)))
        x = x + x_tr
        x = self.norm(x)
        return x


class RetrievalHead(nn.Module):
    def __init__(self, dims=1792):
        super(RetrievalHead, self).__init__()
        self.dims = dims
        self.norm = nn.LayerNorm(dims)
        self.res = Residual(dims)

    def forward(self, x):
        x = self.norm(x)
        x = self.res(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GeoLoc(BaseModel):
    required_inputs = ['image']

    models_urls = {
        'efficientnet': 'https://mever.iti.gr/geoloc/efficientnet.pt',
        'rrm': 'https://mever.iti.gr/geoloc/rrm.pt'
    }

    def _init(self, conf):
        self.norm_rgb = tvf.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

        # Download the checkpoint.
        geoloc_path.mkdir(parents=True, exist_ok=True)
        if not (geoloc_path / 'efficientnet.pt').exists():
            link = self.models_urls['efficientnet']
            cmd = ['wget', link, '-O', str(geoloc_path / 'efficientnet.pt')]
            subprocess.run(cmd, check=True)
        if not (geoloc_path / 'rrm.pt').exists():
            link = self.models_urls['rrm']
            cmd = ['wget', link, '-O', str(geoloc_path / 'rrm.pt')]
            subprocess.run(cmd, check=True)

        self.model = EfficientNet.from_pretrained('efficientnet-b4', include_top=False)
        checkpoint = torch.load(str(geoloc_path / 'efficientnet.pt'), map_location='cpu')
        self.model.load_state_dict(checkpoint)

        if conf['rrm']:
            self.ret_head = RetrievalHead(1792)
            checkpoint = torch.load(str(geoloc_path / 'rrm.pt'), map_location='cpu')
            self.ret_head.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

    def _forward(self, data):
        img = self.norm_rgb(data['image'])
        desc = self.model.extract_features(img)
        desc = einops.rearrange(desc, 'b c h w -> b (h w) c')
        if hasattr(self, 'ret_head'):
            desc = self.ret_head(desc)
        desc = einops.reduce(desc, 'b n c -> b c', 'mean')
        desc = F.normalize(desc, p=2, dim=1)
        return {
            'global_descriptor': desc
        }
