"""Evaluation callback when training"""

import os
import stat
import glob
from time import time
import numpy as np
from mindspore import Tensor, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import log as logger
from mindspore.train.callback import Callback


class EvalCallback(Callback):
    """
    Evaluation callback when training.

    Args:
        eval_function (function): evaluation function.
        eval_param_dict (dict): evaluation parameters' configure dict.
        interval (int): run evaluation interval, default is 1.
        eval_start_epoch (int): evaluation start epoch, default is 1.
        save_best_ckpt (bool): Whether to save best checkpoint, default is True.
        best_ckpt_name (str): bast checkpoint name, default is `best.ckpt`.
        metrics_name (str): evaluation metrics name, default is `acc`.

    Returns:
        None

    Examples:
        >>> EvalCallback(eval_function, eval_param_dict)
    """

    def __init__(self, eval_function, eval_param_dict, rank_id=0, interval=1, eval_start_epoch=1, save_best_ckpt=True,
                 eval_all_saved_ckpts=False, ckpt_directory="./", best_ckpt_name="best.ckpt", metrics_name="acc"):
        super(EvalCallback, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.rank_id = rank_id
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.eval_all_saved_ckpts = eval_all_saved_ckpts
        self.best_res = 0
        self.best_epoch = 0
        self.ckpt_directory = ckpt_directory
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)
        self.last_ckpt_path = os.path.join(ckpt_directory, "last.ckpt")
        self.metrics_name = metrics_name

        os.makedirs(ckpt_directory, exist_ok=True)
        self.start = time()

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def on_train_step_end(self, run_context):

        cb_params = run_context.original_args()
        num_batches = int(cb_params.batch_num)
        cur_epoch = int(cb_params.cur_epoch_num)
        cur_step_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num)

        if cb_params.optimizer is not None:
            optimizer = cb_params.optimizer
        else:
            optimizer = cb_params.train_network.network.optimizer

        if (cur_step_in_epoch + 1) % self.interval == 0 or \
                (cur_step_in_epoch + 1) >= num_batches or cur_step_in_epoch == 0:
            step = optimizer.global_step
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(step - 1)[0].asnumpy()
            else:
                cur_lr = optimizer.learning_rate.asnumpy()
            loss = self._get_loss(cb_params)

            print("Epoch: {}, batch:[{}/{}], device: {}, loss:{:.6f}, lr: {:.7f},  time:{:.6f}s".format(cur_epoch,
                                                                                                        cur_step_in_epoch + 1,
                                                                                                        num_batches,
                                                                                                        self.rank_id,
                                                                                                        loss.asnumpy(),
                                                                                                        cur_lr,
                                                                                                        time() - self.start))
            self.start = time()

    def on_train_epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if self.rank_id == 0:
            if cur_epoch >= self.eval_start_epoch:
                res = self.eval_function(self.eval_param_dict)
                print("epoch: {}, {}: {}".format(cur_epoch, self.metrics_name, res), flush=True)
                if res >= self.best_res:
                    self.best_res = res
                    self.best_epoch = cur_epoch
                    print("update best result: {}".format(res), flush=True)
                    if self.save_best_ckpt:
                        if os.path.exists(self.best_ckpt_path):
                            self.remove_ckpoint_file(self.best_ckpt_path)
                        save_checkpoint(cb_params.train_network, self.best_ckpt_path)
                        print("update best checkpoint at: {}".format(self.best_ckpt_path), flush=True)

    def on_train_end(self, run_context):
        if self.rank_id == 0:
            print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name,
                                                                                         self.best_res,
                                                                                         self.best_epoch), flush=True)

    def _get_loss(self, cb_params):
        """
        Get loss from the network output.
        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
        Returns:
            Union[Tensor, None], if parse loss success, will return a Tensor value(shape is [1]), else return None.
        """
        output = cb_params.net_outputs
        if output is None:
            logger.warning("Can not find any output by this network, so SummaryCollector will not collect loss.")
            return None

        if isinstance(output, (int, float, Tensor)):
            loss = output
        elif isinstance(output, (list, tuple)) and output:
            # If the output is a list, since the default network returns loss first,
            # we assume that the first one is loss.
            loss = output[0]
        else:
            logger.warning("The output type could not be identified, expect type is one of "
                           "[int, float, Tensor, list, tuple], so no loss was recorded in SummaryCollector.")
            return None

        if not isinstance(loss, Tensor):
            loss = Tensor(loss)

        loss = Tensor(np.mean(loss.asnumpy()))
        return loss