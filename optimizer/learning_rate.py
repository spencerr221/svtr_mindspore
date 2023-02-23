from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# class Cosine(object):
#     """
#     Cosine learning rate decay
#     lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
#     Args:
#         lr(float): initial learning rate
#         step_each_epoch(int): steps each epoch
#         epochs(int): total training epochs
#         last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
#     """
#
#     def __init__(self,
#                  learning_rate,
#                  step_each_epoch,
#                  epochs,
#                  warmup_epoch=0,
#                  last_epoch=-1,
#                  **kwargs):
#         super(Cosine, self).__init__()
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.total_step = step_each_epoch * epochs            #trainging steps
#         self.step_per_epoch=step_each_epoch
#         self.last_epoch = last_epoch
#         self.warmup_epoch=warmup_epoch
#         self.warmup_steps = round(warmup_epoch * step_each_epoch)    #warming up steps
#
#     def __call__(self):
#         learning_rate = nn.cosine_decay_lr(    #TODO: fix nn.cosine_decay_lr to lr.CosineAnnealingDecay
#             max_lr=self.learning_rate,
#             min_lr=0.0,
#             total_step=self.total_step,
#             step_per_epoch=self.step_per_epoch,
#             decay_epoch=self.epochs)
#             # T_max=self.T_max,
#             # last_epoch=self.last_epoch)
#         if self.warmup_epoch > 0:
#             if self.total_step > self.warmup_steps:
#                 learning_rate=learning_rate
#             else:
#                 # print("nn.warmup_lr")
#                 learning_rate = nn.warmup_lr(      #TODO: fix
#                     learning_rate=learning_rate,
#                     warmup_epoch=self.warmup_epoch,
#                     total_step=self.total_step,
#                     step_per_epoch=self.step_per_epoch
#                    )
#         return learning_rate

"""Cosine Decay with Warmup Learning Rate Scheduler"""
from mindspore import nn as nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
import math
import mindspore as ms
import mindspore.ops as ops

from mindspore import Tensor

def cosine_annealing_lr(t_max, eta_min, *, eta_max, steps_per_epoch, epochs):
    ## eta_max -> learning_rate
    steps = steps_per_epoch * epochs
    delta = 0.5 * (eta_max - eta_min)
    lrs = []
    import pdb;pdb.set_trace()
    for i in range(steps):
        t_cur = math.floor(i / steps_per_epoch)
        lrs.append(eta_min + delta * (1.0 + math.cos(math.pi * t_cur / t_max)))
    return lrs

class WarmupCosineDecayLR(LearningRateSchedule):
    """ CosineDecayLR with warmup
    The learning rate will increase from 0 to max_lr in `warmup_epochs` epochs, then decay to min_lr in `decay_epoches` epochs
    """

    def __init__(self,
                 min_lr,
                 max_lr,
                 warmup_epochs,
                 decay_epochs,
                 steps_per_epoch
                 ):
        super().__init__()
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.decay_steps = decay_epochs * steps_per_epoch
        if self.warmup_steps > 0:
            self.warmup_lr = nn.WarmUpLR(max_lr, self.warmup_steps)
        self.cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, self.decay_steps)
        self.zero = Tensor(0.0, dtype=ms.float32)

    def step_lr(self, global_step):
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                lr = self.cosine_decay_lr(global_step - self.warmup_steps)
            else:
                lr = self.warmup_lr(global_step)
        else:
            lr = self.cosine_decay_lr(global_step)

        lr = ops.clip_by_value(lr, clip_value_min=self.zero)
        return lr

    def construct(self, global_step):
        lr = self.step_lr(global_step)
        return lr

class LinearStepDecayLR(LearningRateSchedule):
    """ Multiple step learning rate
    The learning rate will decay once the number of step reaches one of the milestones.
    """

    def __init__(self, learning_rate, warmup_epochs, steps_per_epoch, epochs, **kwargs):
        super().__init__()
        self.warmup_steps = round(warmup_epochs * steps_per_epoch)
        self.t_max = steps_per_epoch * epochs
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_lr = 0.0
        if self.warmup_steps > 0:
            self.warmup_lr = nn.WarmUpLR(learning_rate, self.warmup_steps)

    def construct(self, global_step):
        learning_rate = cosine_annealing_lr(
            t_max=self.t_max,
            eta_min=0.0,
            eta_max=self.learning_rate,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs
        )
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                # lr = self.start_lr + (self.learning_rate - self.start_lr) * global_step / self.warmup_steps + self.start_lr
                lr = learning_rate
            else:
                lr = self.warmup_lr(global_step)
        else:
            print("warmup_steps not support 0")
        return lr
