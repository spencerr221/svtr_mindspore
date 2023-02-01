from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mindspore import nn

class AdamW(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.01,
                 multi_precision=False,
                 grad_clip=None,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False,
                 name=None,
                 lazy_mode=False,
                 **args):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = 0.01 if weight_decay is None else weight_decay
        self.grad_clip = grad_clip
        self.name = name
        self.lazy_mode = lazy_mode
        self.multi_precision = multi_precision
        self.no_weight_decay_name_list = no_weight_decay_name.split(
        ) if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model):
        parameters = [
            param for param in model.parameters() if param.trainable is True
        ]

        self.no_weight_decay_param_name_list = [
            p.name for n, p in model.named_parameters()
            if any(nd in n for nd in self.no_weight_decay_name_list)
        ]

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += [
                p.name for n, p in model.named_parameters() if len(p.shape) == 1
            ]

        opt = nn.AdamWeightDecay(
            params=parameters,
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.epsilon,
            weight_decay=self.weight_decay)     #multi_precision, grad_clip, name, lazy_mode, apply_decay_param_fun

        return opt

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list