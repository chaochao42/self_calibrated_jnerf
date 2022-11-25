import jittor as jt
import jittor.nn as nn

import numpy as np
# from jnerf.models.position_encoders.neus_encoder.embedder import get_embedder
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS


@NETWORKS.register_module()
class ResidualCamera(nn.Module):
    def __init__(self, number_of_images, pose_num=6):
         self.residual_camera = jt.Embedding(number_of_images, pose_num)

    def get_complete_pose_(self, current_pose):
        new_pose = self.residual_camera + current_pose