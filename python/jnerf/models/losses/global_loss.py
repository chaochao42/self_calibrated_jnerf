import jittor as jt
from jittor import nn
from jnerf.utils.registry import LOSSES

@LOSSES.register_module()
class GlobalLoss(nn.Module):
    def __init__(self, delta=1):
        self.delta = delta
    def execute(self, poses):

        numbers = len(poses)

        pose_list = jt.misc.chunk(poses, numbers, 0)
        print(f"========={poses.shape}======={numbers}=========Length {len(pose_list)}====index 0 shape{pose_list[0].shape}")