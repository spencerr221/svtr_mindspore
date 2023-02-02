from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mindspore.nn as nn

class Cosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Cosine, self).__init__()
        self.learning_rate = learning_rate
        self.total_step = step_each_epoch * epochs            #trainging steps
        self.step_per_epoch=step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch=warmup_epoch
        self.warmup_steps = round(warmup_epoch * step_each_epoch)    #warming up steps

    def __call__(self):
        learning_rate = nn.cosine_decay_lr(    #TODO: fix nn.cosine_decay_lr to lr.CosineAnnealingDecay
            max_lr=self.learning_rate,
            min_lr=0,
            total_step=self.total_step,
            step_per_epoch=self.step_per_epoch,
            decay_epoch=epochs)
            # T_max=self.T_max,
            # last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            if self.total_step > self.warmup_steps:
                learning_rate=learning_rate
                else:
                    print("nn.warmup_lr")
                    learning_rate = nn.warmup_lr(      #TODO: fix
                        learning_rate=learning_rate,
                        warmup_epoch=self.warmup_epoch,
                        total_step=self.total_step,
                        step_per_epoch=self.step_per_epoch
                       )
        return learning_rate