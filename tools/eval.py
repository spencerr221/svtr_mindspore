'''
Model evaluation 
'''
import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import yaml
import argparse
from addict import Dict

import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size

from mindocr.data import build_dataset
from mindocr.models import build_model
from mindocr.postprocess import build_postprocess
from mindocr.metrics import build_metric
from mindocr.utils.callbacks import Evaluator

_data_list = {
    "CUTE80": "CUTE80",
    "IC03_860": "IC03_860",
    "IC03_867": "IC03_867",
    "IC13_857": "IC13_857",
    "IC13_1015": "IC13_1015",
    "IC15_1811": "IC15_1811",
    "IC15_2077": "IC15_2077",
    "IIIT5k_3000": "IIIT5k_3000",
    "SVT": "SVT",
    "SVTP": "SVTP"
}

def main(cfg):
    # env init
    ms.set_context(mode=cfg.system.mode)
    ms.set_context(device_id=0)
    if cfg.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True,
                                     )
    else:
        device_num = None
        rank_id = None

    is_main_device = rank_id in [None, 0]

    # load dataset
    loader_eval = build_dataset(
            cfg.eval.dataset,
            cfg.eval.loader,
            num_shards=device_num,
            shard_id=rank_id,
            is_train=False,
            refine_batch_size=True,
            )
    num_batches = loader_eval.get_dataset_size()

    # model
    cfg.model.backbone.pretrained = False
    network = build_model(cfg.model, ckpt_load_path=cfg.eval.ckpt_load_path)
    network.set_train(False)
    # import pdb; pdb.set_trace()

    if cfg.system.amp_level != 'O0':
        print('INFO: Evaluation will run in full-precision(fp32)')

    # TODO: check float type conversion in official Model.eval
    # ms.amp.auto_mixed_precision(network, amp_level='O0')

    # postprocess, metric
    postprocessor = build_postprocess(cfg.postprocess)
    # postprocess network prediction
    metric = build_metric(cfg.metric)

    net_evaluator = Evaluator(network, None, postprocessor, [metric])
    data_dir = cfg.eval.dataset.data_dir
    for k, v in _data_list.items():
        data_folder = os.path.join(data_dir, v)
        cfg.eval.dataset.data_dir = data_folder
        loader_eval = build_dataset(
                cfg.eval.dataset,
                cfg.eval.loader,
                num_shards=None,
                shard_id=None,
                is_train=False)
        num_batches = loader_eval.get_dataset_size()

        # log
        print('='*40)
        print(
            f'Num batches: {num_batches}\n'
            )
        if 'name' in cfg.model:
            print(f'Model: {cfg.model.name}')
        else:
            print(f'Model: {cfg.model.backbone.name}-{cfg.model.neck.name}-{cfg.model.head.name}')
        print('='*40)

        measures = net_evaluator.eval(loader_eval)
        print(f"{v} dataset performance:", measures)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation Config', add_help=False)
    parser.add_argument('-c', '--config', type=str, default='',
                        help='YAML config file specifying default arguments (default='')')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # argpaser
    args = parse_args()
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)

    #print(config)
    
    main(config)
