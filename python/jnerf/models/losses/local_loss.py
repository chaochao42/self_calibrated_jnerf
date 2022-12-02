import jittor as jt
from jittor import nn
from jnerf.utils.registry import LOSSES

@LOSSES.register_module()
class LocalLoss(nn.Module):
    def __init__(self, delta):
        self.delta = delta