from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import argparse

__dir__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from svtr_mindspore.data import build_dataloader
from svtr_mindspore.modeling.architectures import build_model
from svtr_mindspore.modeling.wrapper.wrapper import with_loss_cell
from svtr_mindspore.losses import build_loss
from svtr_mindspore.optimizer import build_optimizer
from svtr_mindspore.postprocess import build_post_process
from svtr_mindspore.metrics import build_metric
from svtr_mindspore.utils import load_config
from svtr_mindspore.utils import EvalCallback

import mindspore as ms
from mindspore import FixedLossScaleManager, Model, CheckpointConfig, ModelCheckpoint, DynamicLossScaleManager
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore import Profiler
ms.set_seed(42)


def apply_eval(eval_param):
    evaluation_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = evaluation_model.eval(eval_ds, dataset_sink_mode=False)
    return res[metrics_name]

def train(args):
    # set up mindspore runing mode
    config_path = args.config_path
    config = load_config(config_path)
    mode = config['Global']['mode']
    enable_graph_kernel = config['Global']['enable_graph_kernel']
    ms.set_context(mode=mode)
    if mode == 0:
        ms.set_context(enable_graph_kernel=enable_graph_kernel)

    # set up distribution mode
    if config['Global']['distributed']:
        ms.communication.init()
        device_num = ms.communication.get_group_size()
        rank_id = ms.communication.get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )

        if "DEVICE_ID" in os.environ:
            ms.set_context(device_id=int(os.environ["DEVICE_ID"]))
        print("rank_id,device_num", rank_id, device_num)
    else:
        device_num = None
        rank_id = None

    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train',num_shards=device_num)
    # for data in train_dataloader:
    #     print("DDDDDDDDDDDDD", data)
    # exit(0)
# TODO: eval
    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval')
    else:
        valid_dataloader = None

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)


    # build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                if config['Architecture']['Models'][key]['Head'][
                        'name'] == 'MultiHead':  # for multi head
                    if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    # update SARLoss params
                    assert list(config['Loss']['loss_config_list'][-1].keys())[
                        0] == 'DistillationSARLoss'
                    config['Loss']['loss_config_list'][-1][
                        'DistillationSARLoss']['ignore_index'] = char_num + 1
                    out_channels_list = {}
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
        elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # for multi head
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            # update SARLoss params
            assert list(config['Loss']['loss_config_list'][1].keys())[
                0] == 'SARLoss'
            if config['Loss']['loss_config_list'][1]['SARLoss'] is None:
                config['Loss']['loss_config_list'][1]['SARLoss'] = {
                    'ignore_index': char_num + 1
                }
            else:
                config['Loss']['loss_config_list'][1]['SARLoss'][
                    'ignore_index'] = char_num + 1
            out_channels_list = {}
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num

        if config['PostProcess']['name'] == 'SARLabelDecode':  # for SAR model
            config['Loss']['ignore_index'] = char_num - 1
    model_no_loss = build_model(config['Architecture'])


    # build loss
    loss_class = build_loss(config['Loss'])

    network = with_loss_cell(model_no_loss, loss_class)

    # build optim with lr
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=train_dataloader.get_dataset_size(),
        model=model_no_loss)


    # build metric
    eval_class = build_metric(config['Metric'], decoder=post_process_class)

    if config["Global"].get("use_ema", True):
        print("under_developing")
    else:
        if config["Optimizer"]["dynamic_loss_scale"]:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=config["Global"]["loss_scale"], scale_factor=2,
                                                         scale_window=1000)
        else:
            print("using fixed loss scale")
            loss_scale_manager = FixedLossScaleManager(loss_scale=config["Global"]["loss_scale"], drop_overflow_update=False)
            # model=Model(network=model,loss_fn=loss_class,optimizer=optimizer,metrics={'RecMetric':eval_class},amp_level=config["Global"]["amp_level"],loss_scale_manager=loss_scale_manager)
        model = Model(network=network, optimizer=optimizer, amp_level=config["Global"]["amp_level"], loss_scale_manager= loss_scale_manager)


# callbacks:   #TODO:eval and infer
    step_each_epoch = train_dataloader.get_dataset_size()
    epochs = config['Global']['epoch_num']
    if config["Train"]["save_checkpoint"]:
        save_ckpt_path = os.path.join(config["Global"]["save_model_dir"],'ckpt')

    if valid_dataloader is not None and valid_dataloader.get_dataset_size() > 0:
        # eval_model = Model(network=network.set_train(False),  optimizer=optimizer, loss_fn=loss_class, metrics={'SVTRMetric': eval_class}, amp_level=config["Global"]["amp_level"], loss_scale_manager=loss_scale_manager)
        eval_model = Model(network=model_no_loss.set_train(False), loss_fn=loss_class,
                           metrics={'SVTRAccuracy': eval_class})
        eval_param_dict = {
            "model": eval_model,
            "dataset": valid_dataloader,
            "metrics_name": "SVTRAccuracy"
        }
        eval_callback = EvalCallback(apply_eval, eval_param_dict, rank_id, interval=config['Global']["eval_interval"],
                                     eval_start_epoch=config['Global']["eval_start_epoch"], save_best_ckpt=True,
                                     ckpt_directory=save_ckpt_path, best_ckpt_name="best_acc.ckpt",
                                     eval_all_saved_ckpts=config['Global']["eval_all_saved_ckpts"], metrics_name="acc")
        callbacks = [eval_callback]
    else:
        callbacks = [LossMonitor(per_print_times=config["Global"]["per_print_time"]),
               TimeMonitor(data_size=step_each_epoch)]
# #save checkpoints
#     if config["Train"]["save_checkpoint"]:
#         save_ckpt_path = os.path.join(config["Global"]["save_model_dir"],'ckpt')
#         config_ck = CheckpointConfig(save_checkpoint_steps=config["Global"]["save_checkpoint_steps"],keep_checkpoint_max=config["Global"]["keep_checkpoint_max"])
#         ckpt_cb = ModelCheckpoint(prefix="svtr",directory=save_ckpt_path,config=config_ck)
#         callbacks.append(ckpt_cb)
# start train
    dataset_sink_mode = False
    model.train(epochs, train_dataloader, callbacks, dataset_sink_mode)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path",type=str,default="configs/rec_svtrnet.yaml",help="Config file path")
    args = parser.parse_args()
    # profiler = Profiler()
    ms.set_context(device_id=2)
    train(args)

    # profiler.analyse()