import jittor as jt
from jittor import nn
from jnerf.utils.registry import LOSSES
import random
import numpy as np


@LOSSES.register_module()
class GlobalLoss(nn.Module):
    def __init__(self, shot_number=4, pose_number=35):
        self.shot_number = shot_number
        self.pose_number = pose_number

    # def pose_to_angle(self, single_pose):

    def execute(self, poses):
        index_list = np.random.randint(0, self.pose_number, size=self.shot_number)
        pose_var = poses[index_list]
        pose_list = jt.misc.chunk(pose_var, self.shot_number, 0)
        print(f"============ pose_list type {type(pose_list)} and length {len(pose_list)}")
        print("========================================")
        print(pose_list[0])
        #relative_1 =