
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

def eval(args, data_dir=None):
    config_path = args.config_path
    config = load_config(config_path)
    global_config = config['Global']
    eval_config = config["Eval"]
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=global_config["device_target"], save_graphs=False)
    if global_config["device_target"] == 'Ascend':
        context.set_context(device_id=args.device_id)
    device_info = {
        "num_shards": 1,
        "shard_id": 0
    }
    model_no_loss = build_model(config['Architecture'])
    param_dict = load_checkpoint(global_config["checkpoints"])
    load_param_into_net(model_no_loss, param_dict)
    model_no_loss.set_train(False)
    if data_dir:
        eval_config["dataset"]["data_dir"] = data_dir

    if not args.evalset:
        eval_dataset = build_dataloader(config, 'Eval')
        loss_class = build_loss(config['Loss'])
        post_process_class = build_post_process(config['PostProcess'],
                                                global_config)
        eval_class = build_metric(config['Metric'], decoder=post_process_class)
        eval_model = Model(network=model_no_loss.set_train(False), loss_fn=loss_class,
                           metrics={'SVTRAccuracy': eval_class})
        dataset_sink_mode = False
        res = eval_model.eval(
            eval_dataset, dataset_sink_mode=dataset_sink_mode)
        print("result:", res, flush=True)

    else:
        import copy
        data_dir = eval_config["dataset"]["data_dir"]
        results = []
        for k, v in _data_list.items():
            data_folder = os.path.join(data_dir, v)
            eval_config["dataset"]["data_dir"] = data_folder
            eval_dataset = build_dataloader(config, 'Eval')

            loss_class = build_loss(config['Loss'])
            post_process_class = build_post_process(config['PostProcess'],
                                                    global_config)
            eval_class = build_metric(config['Metric'], decoder=post_process_class)
            eval_model = Model(network=model_no_loss.set_train(False), loss_fn=loss_class,
                               metrics={'SVTRAccuracy': eval_class})
            dataset_sink_mode = False
            result = eval_model.eval(
                eval_dataset, dataset_sink_mode=dataset_sink_mode)
            results.append(result["SVTRAccuracy"])
            print(f"{k} dataset result:", result, flush=True)

        avgscore = sum(results) / len(results)
        print(f"Average: {avgscore:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/crnn-lmdb.yaml",
                        help="Config file (.yaml) path")
    parser.add_argument("--device_id", type=int, default=0,
                        help="device id")
    parser.add_argument("--evalset", action='store_true')
    args = parser.parse_args()
    eval(args)


