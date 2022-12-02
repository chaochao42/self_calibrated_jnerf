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

    def pose_to_angle(self, single_pose):
        print("=================== Single pose")
        # print(f"======Shape {single_pose.shape}")
        # print(single_pose)
        r32 = single_pose[:,2,1]
        r33 = single_pose[:,2,2]
        r31 = single_pose[:,2,0]
        r21 = single_pose[:,1,0]
        r11 = single_pose[:,0,0]
        theta_x = jt.misc.arctan2(r32, r33)
        theta_y = jt.misc.arctan2(-r31, jt.sqrt(r32**2 + r33**2))
        theta_z = jt.misc.arctan2(r21, r11)

        print(f"=={theta_x}=={theta_y}=={theta_z}==")
        # r32 = single_pose[]
    def execute(self, poses):
        index_list = np.random.randint(0, self.pose_number, size=self.shot_number)
        pose_var = poses[index_list]
        pose_list = jt.misc.chunk(pose_var, self.shot_number, 0)
        pose_0 = pose_list[0]
        self.pose_to_angle(pose_0)
        # print(f"============ pose_list type {type(pose_list)} and length {len(pose_list)}")
        # print("========================================")
        # print(pose_list[0])
        #relative_1 =