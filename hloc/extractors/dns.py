import sys
from pathlib import Path
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel

dns_path = Path(__file__).parent / "../../third_party/distill-and-select"
sys.path.append(str(dns_path))
from model.feature_extractor import FeatureExtractor
from model.students import CoarseGrainedStudent


class DnS(BaseModel):
    required_inputs = ['image']

    def _init(self, conf):
        self.feat_ext = FeatureExtractor(dims=512).eval()
        self.cg_student = CoarseGrainedStudent(pretrained=True).eval()

    def _forward(self, data):
        img = data['image'].permute(0, 2, 3, 1) * 255
        desc = self.feat_ext(img).permute(1, 0, 2)
        desc = self.cg_student.index_video(desc)
        return {
            'global_descriptor': desc
        }
