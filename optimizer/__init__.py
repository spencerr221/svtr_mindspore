from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy

__all__ = ['build_optimizer']
from .learning_rate import WarmupCosineDecayLR
from .learning_rate import LinearStepDecayLR

def build_lr_scheduler(lr_config, epochs, step_each_epoch):
    from . import learning_rate
    lr_name = lr_config.pop('name').lower()
    lr_config.update({'steps_per_epoch': step_each_epoch, 'scheduler': lr_name, 'epochs': epochs})
    # lr_name = lr_config.pop('name').lower()
    # lr = getattr(learning_rate, lr_name)(**lr_config)()
    # return lr
    lr = create_scheduler(**lr_config)
    return lr

def create_scheduler(
        steps_per_epoch: int,
        scheduler: str,
        learning_rate: float,
        min_lr: float,
        warmup_epochs: int,
        epochs: int,
        decay_epochs: int,
        decay_rate: float = 0.9,
        milestones: list = None
):
    r"""Creates learning rate scheduler by name.
    Args:
        steps_per_epoch: number of steps per epoch.
        scheduler: scheduler name like 'constant', 'warmup_cosine_decay', 'step_decay',
            'exponential_decay', 'polynomial_decay', 'multi_step_decay'. Default: 'constant'.
        lr: learning rate value. Default: 0.01.
        min_lr: lower lr bound for cyclic/cosine/polynomial schedulers. Default: 1e-6.
        warmup_epochs: epochs to warmup LR, if scheduler supports. Default: 3.
        decay_epochs: epochs to decay LR to min_lr for cyclic and polynomial schedulers.
            decay LR by a factor of decay_rate every `decay_epochs` for exponential scheduler and step LR scheduler.
            Default: 10.
        decay_rate: LR decay rate (default: 0.9)
        milestones: list of epoch milestones for multi_step_decay scheduler. Must be increasing.
    Returns:
        Cell object for computing LR with input of current global steps
    """

    if milestones is None:
        milestones = []
    # import pdb;pdb.set_trace()
    if scheduler == 'warmup_cosine_decay':
        lr_scheduler = WarmupCosineDecayLR(min_lr=min_lr,
                                           max_lr=learning_rate,
                                           warmup_epochs=warmup_epochs,
                                           decay_epochs=decay_epochs,
                                           steps_per_epoch=steps_per_epoch
                                           )
    elif scheduler == 'linear_step_decay_lr':
        lr_scheduler = LinearStepDecayLR(
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )
    elif scheduler == 'constant':
        lr_scheduler = learning_rate
    else:
        raise ValueError(f'Invalid scheduler: {scheduler}')

    return lr_scheduler


def build_optimizer(config, epochs, step_each_epoch, model):
    from . import regularizer, optimizer
    config = copy.deepcopy(config)
    # step1 build lr
    lr = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch)

    # step2 build regularization
    if 'regularizer' in config and config['regularizer'] is not None:
        reg_config = config.pop('regularizer')
        reg_name = reg_config.pop('name')
        if not hasattr(regularizer, reg_name):
            reg_name += 'Decay'
        reg = getattr(regularizer, reg_name)(**reg_config)()
    elif 'weight_decay' in config:
        reg = config.pop('weight_decay')
    else:
        reg = None

    # step3 build optimizer
    optim_name = config.pop('name')
    # if 'clip_norm' in config:
    #     clip_norm = config.pop('clip_norm')
    #     grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    # elif 'clip_norm_global' in config:
    #     clip_norm = config.pop('clip_norm_global')
    #     grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    # else:
    #     grad_clip = None
    grad_clip = None
    optim = getattr(optimizer, optim_name)(learning_rate=lr,
                                           weight_decay=reg,
                                           grad_clip=grad_clip,
                                           **config)
    return optim(model), lr