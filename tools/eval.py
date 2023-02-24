
import os
import argparse

from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from svtr_mindspore.losses import build_loss
from svtr_mindspore.optimizer import build_optimizer
from svtr_mindspore.postprocess import build_post_process
from svtr_mindspore.metrics import build_metric
from svtr_mindspore.utils import load_config
from svtr_mindspore.data import build_dataloader
from svtr_mindspore.modeling.architectures import build_model

def eval(args):
    config_path = args.config_path
    config = load_config(config_path)
