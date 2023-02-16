from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import signal
import os
import sys

# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy

from typing import Optional
import os
from ..data.lmdb_dataset import LMDBDataSet
from mindspore.dataset import GeneratorDataset, DistributedSampler

import mindspore.dataset as ds


__all__ = ["build_dataloader"]

def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)

def build_dataloader(
    config, 
    mode,
    seed=None,
    shuffle: bool = True,
    num_samples: Optional[bool] = None,
    num_shards: Optional[int] = None,
    shard_id: Optional[int] = None,
):
    config = copy.deepcopy(config)
    support_dict = ['LMDBDataSet']    #TODO add more
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    if 'use_shared_memory' in loader_config.keys():
        use_shared_memory = loader_config['use_shared_memory']
    else:
        use_shared_memory = True

    # if mode == "Train":
    #     # Distribute data to multiple cards
    #     batch_sampler = DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)   # TODO add params, num_shards=none
    # else:
    #     # Distribute data to single card
    #     # batch_sampler = BatchSampler(
    #     #     dataset=dataset,
    #     #     batch_size=batch_size,
    #     #     shuffle=shuffle,
    #     #     drop_last=drop_last)
    #     if shuffle:
    #         batch_sampler = ds.RandomSampler(replacement=False, num_samples=num_samples)
    #     else:
    #         batch_sampler = ds.SequentialSampler(num_samples=num_samples)
    # mindspore_kwargs = dict(shuffle=None, sampler=batch_sampler,
    #                     num_parallel_workers=num_workers)   #TODO: num_parallel_workers=num_shards
    mindspore_kwargs = dict(shuffle=shuffle,
                        num_parallel_workers=num_workers)
    dataset_generator = eval(module_name)(config, mode, seed)
    dataset = GeneratorDataset(dataset_generator,["image", "label"],**mindspore_kwargs)
    dataset=dataset.batch(batch_size,drop_remainder=drop_last)

    return dataset
