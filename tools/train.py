from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from ..data import build_dataloader
from ..modeling.architectures import build_model
from ..losses import build_loss
from ..optimizer import build_optimizer
from ..postprocess import build_post_process
from ..metrics import build_metric
from ..utils import load_config

import mindspore as ms
from mindspore import FixedLossScaleManager, Model, CheckpointConfig, ModelCheckpoint
from mindspore.train.callback import TimeMonitor, LossMonitor
ms.set_seed(0)

def train(args):
    # set up mindspore runing mode
    config_path = args.config_path
    config=load_config(config_path)
    mode=config['Global']['mode']
    enable_graph_kernel=config['Global']['enable_graph_kernel']
    ms.set_context(mode=mode)
    if mode ==0:
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
            parameter_broadcast=True,
        )

        if "DEVICE_ID" in os.environ:
            ms.set_context(device_id=int(os.environ["DEVICE_ID"]))
    else:
        device_num = None
        rank_id = None

    global_config = config['Global']

    # build dataloader
    train_dataloader = build_dataloader(config, 'Train')
# TODO: eval
    # if config['Eval']:
    #     valid_dataloader = build_dataloader(config, 'Eval')
    # else:
    #     valid_dataloader = None

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

    model = build_model(config['Architecture'])

    # use_sync_bn = config["Global"].get("use_sync_bn", False)
    # if use_sync_bn:
    #     model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     logger.info('convert_sync_batchnorm')
##TODO: static?
    # model = apply_to_static(model, config, logger)

    # build loss
    loss_class = build_loss(config['Loss'])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        model=model)

    # build metric
    eval_class = build_metric(config['Metric'])

    # logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    # if valid_dataloader is not None:
    #     logger.info('valid dataloader has {} iters'.format(
    #         len(valid_dataloader)))

    # use_amp = config["Global"].get("use_amp", False)
    # amp_level = config["Global"].get("amp_level", 'O2')
    # amp_custom_black_list = config['Global'].get('amp_custom_black_list', [])
    # if use_amp:
    #     AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
    #     if paddle.is_compiled_with_cuda():
    #         AMP_RELATED_FLAGS_SETTING.update({
    #             'FLAGS_cudnn_batchnorm_spatial_persistent': 1
    #         })
    #     paddle.fluid.set_flags(AMP_RELATED_FLAGS_SETTING)
    #     scale_loss = config["Global"].get("scale_loss", 1.0)
    #     use_dynamic_loss_scaling = config["Global"].get(
    #         "use_dynamic_loss_scaling", False)
    #     scaler = paddle.amp.GradScaler(
    #         init_loss_scaling=scale_loss,
    #         use_dynamic_loss_scaling=use_dynamic_loss_scaling)
    #     if amp_level == "O2":
    #         model, optimizer = paddle.amp.decorate(
    #             models=model,
    #             optimizers=optimizer,
    #             level=amp_level,
    #             master_weight=True)
    # else:
    scaler = None

    if config["Global"].get("use_ema", True):
        print("under_developing")
    else:
# TODO: dynamic loss scale
        loss_scale_manager = FixedLossScaleManager(loss_scale=config["Global"]["loss_scale"],drop_overflow_update=False)
        model=Model(network=model,optimizer=optimizer,metrics=eval_class,amp_level=config["Global"]["amp_level"],loss_scale_manager=loss_scale_manager)

    # # load pretrain model
    # pre_best_model_dict = load_model(config, model, optimizer,
    #                                  config['Architecture']["model_type"])

    # if config['Global']['distributed']:
    #     model = paddle.DataParallel(model)

# callbacks:   #TODO:eval and infer
    step_each_epoch=len(train_dataloader)
    epochs = config['Global']['epoch_num']
    callbacks = [LossMonitor(per_print_times=config["Global"]["per_print_time"]),
               TimeMonitor(data_size=step_each_epoch)]
#save checkpoints
    if config["Train"]["save_checkpoint"]:
        save_ckpt_path=os.path.join(config["Global"]["save_model_dir"],'ckpt')
        config_ck= CheckpointConfig(save_checkpoint_steps=config["Global"]["save_checkpoint_steps"],keep_checkpoint_max=config["Global"]["keep_checkpoint_max"])
        ckpt_cb=ModelCheckpoint(prefix="svtr",directory=save_ckpt_path,config=config_ck)
        callbacks.append(ckpt_cb)
# start train
    dataset_sink_mode = config["Global"]["device_target"] == "Ascend"
    model.train(epochs,train_dataloader,callbacks,dataset_sink_mode)


    # program.train(config, train_dataloader, valid_dataloader, device, model,
    #               loss_class, optimizer, lr_scheduler, post_process_class,
    #               eval_class, pre_best_model_dict, logger, vdl_writer, scaler,
    #               amp_level, amp_custom_black_list)


# def test_reader(config, device, logger):
#     loader = build_dataloader(config, 'Train', device, logger)
#     import time
#     starttime = time.time()
#     count = 0
#     try:
#         for data in loader():
#             count += 1
#             if count % 1 == 0:
#                 batch_time = time.time() - starttime
#                 starttime = time.time()
#                 logger.info("reader: {}, {}, {}".format(
#                     count, len(data[0]), batch_time))
#     except Exception as e:
#         logger.info(e)
#     logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser=argparse.ArgumentParser()
    parser.add_argument("--config_path",type=str,default="../configs/rec_svtrnet.yaml",help="Config file path")
    args = parser.parse_args()
    train(args)
    # test_reader(config, device, logger)