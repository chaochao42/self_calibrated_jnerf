import jittor as jt
from jittor import nn
from jittor.contrib import getitem
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

        # index_x, value_x = jt.argsort(theta_x)
        # index_y, value_y = jt.argsort(theta_y)
        # index_z, value_z = jt.argsort(theta_z)
        #
        # value_x_moved = getitem(value_x, [3, 0, 1, 2])
        # value_y_moved = getitem(value_y, [3, 0, 1, 2])
        # value_z_moved = getitem(value_z, [3, 0, 1, 2])
        value_x = theta_x
        value_y = theta_y
        value_z = theta_z
        value_x_moved = getitem(theta_x, [3, 0, 1, 2])
        value_y_moved = getitem(theta_y, [3, 0, 1, 2])
        value_z_moved = getitem(theta_y, [3, 0, 1, 2])

        diff_x = (value_x - value_x_moved).unsqueeze(0)
        diff_y = (value_y - value_y_moved).unsqueeze(0)
        diff_z = (value_z - value_z_moved).unsqueeze(0)
        Diff_matrix = jt.concat([diff_x, diff_y, diff_z], dim=0)
        # target_1 = jt.sum(jt.abs(Diff_matrix[:, 0]))
        # target_2 = jt.sum(jt.abs(Diff_matrix[:, 1:]))

        target_1 = jt.abs(jt.sum(Diff_matrix[:, 0]))
        target_2 = jt.abs(jt.sum(Diff_matrix[:, 1:]))

        global_loss = (target_1 - target_2).abs()
        return global_loss

