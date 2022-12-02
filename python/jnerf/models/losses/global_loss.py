import jittor as jt
from jittor import nn
from jnerf.utils.registry import LOSSES

@LOSSES.register_module()
class GlobalLoss(nn.Module):
    def __init__(self, delta=1):
        self.delta = delta
    def execute(self, poses):
        print(poses.shape)
        numbers = len(poses)
        pose_list = jt.misc.chunk(poses, numbers, 0)
