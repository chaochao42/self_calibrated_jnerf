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

    def pose_to_angle(self, pose_var):
        r32 = pose_var[:,2,1]
        r33 = pose_var[:,2,2]
        r31 = pose_var[:,2,0]
        r21 = pose_var[:,1,0]
        r11 = pose_var[:,0,0]
        theta_x = jt.misc.arctan2(r32, r33)
        theta_y = jt.misc.arctan2(-r31, jt.sqrt(r32**2 + r33**2))
        theta_z = jt.misc.arctan2(r21, r11)
        return theta_x, theta_y, theta_z


    def execute(self, poses):
        index_list = np.random.randint(0, self.pose_number, size=self.shot_number)
        pose_var = poses[index_list]

        theta_x, theta_y, theta_z = self.pose_to_angle(pose_var)

        index_x, value_x = jt.argsort(theta_x)
        index_y, value_y = jt.argsort(theta_y)
        index_z, value_z = jt.argsort(theta_z)
        print("===================================")
        print(f"Index x==============")
        print(index_x, value_x)
        print(f"Index y==============")
        print(index_y, value_y)
        print(f"Index z==============")
        print(index_z, value_z)
        #relative_1 =