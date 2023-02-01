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
from mindocr.data.lmdb_dataset import LMDBDataSet
from mindspore.dataset import GeneratorDataset, DistributedSampler

import mindspore.datset as ds 


__all__ = ["create_dataset"]

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
    device, 
    logger, 
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
    
    # dataset_generator = LMDBDataSet(config, mode, logger, seed)
    # dataset = GeneratorDataset(dataset_generator,**mindspore_kwargs) #TODO: fix no column names

    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    if 'use_shared_memory' in loader_config.keys():
        use_shared_memory = loader_config['use_shared_memory']
    else:
        use_shared_memory = True

    if mode == "Train":
        # Distribute data to multiple cards
        batch_sampler = DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)   # TODO add params
    else:
        # Distribute data to single card
        # batch_sampler = BatchSampler(
        #     dataset=dataset,
        #     batch_size=batch_size,
        #     shuffle=shuffle,
        #     drop_last=drop_last)
        if shuffle:
            sampler = ds.RandomSampler(replacement=False, num_samples=num_samples)
        else:
            sampler = ds.SequentialSampler(num_samples=num_samples)
    mindspore_kwargs = dict(shuffle=None, sampler=sampler,
                        num_parallel_workers=num_parallel_workers, **kwargs)
    dataset_generator = LMDBDataSet(config, mode, logger, seed)
    dataset = GeneratorDataset(dataset_generator,**mindspore_kwargs) #TODO: fix no column names

    # support exit using ctrl+c
    signal.signal(signal.SIGINT, term_mp)
    signal.signal(signal.SIGTERM, term_mp)

    return dataset



# def create_dataset(
#     config,
#     mode,
#     device,
#     logger,
#     seed=None,
#     name: str = '',
#     root: str = './',
#     split: str = 'train',
#     shuffle: bool = True,
#     num_samples: Optional[bool] = None,
#     num_shards: Optional[int] = None,
#     shard_id: Optional[int] = None,
#     num_parallel_workers: Optional[int] = None,
#     download: bool = False,
#     num_aug_repeats: int = 0,
#     **kwargs
# ):

#     assert (num_samples is None) or (num_aug_repeats==0), 'num_samples and num_aug_repeats can NOT be set together.'

#     name = name.lower()
#     # subset sampling
#     if num_samples is not None and num_samples > 0:
#         # TODO: rewrite ordered distributed sampler (subset sampling in distributed mode is not tested)
#         if num_shards is not None and num_shards > 1: # distributed
#             print('ns', num_shards, 'num_samples', num_samples)
#             sampler = DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)
#         else: # standalone
#             if shuffle:
#                 sampler = ds.RandomSampler(replacement=False, num_samples=num_samples)
#             else:
#                 sampler = ds.SequentialSampler(num_samples=num_samples)
#         mindspore_kwargs = dict(shuffle=None, sampler=sampler,
#                             num_parallel_workers=num_parallel_workers, **kwargs)
#     else:
#         sampler = None
#         mindspore_kwargs = dict(shuffle=shuffle, sampler=sampler, num_shards=num_shards, shard_id=shard_id,
#                             num_parallel_workers=num_parallel_workers, **kwargs)
#     if os.path.isdir(root):
#         dataset_generator = LMDBDataSet(config, mode, logger, seed)
#         dataset = GeneratorDataset(dataset_generator,**mindspore_kwargs)
#     return dataset