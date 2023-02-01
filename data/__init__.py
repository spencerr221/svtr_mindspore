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

def create_dataset(
    config,
    mode,
    device,
    logger,
    seed=None,
    name: str = '',
    root: str = './',
    split: str = 'train',
    shuffle: bool = True,
    num_samples: Optional[bool] = None,
    num_shards: Optional[int] = None,
    shard_id: Optional[int] = None,
    num_parallel_workers: Optional[int] = None,
    download: bool = False,
    num_aug_repeats: int = 0,
    **kwargs
):
    """Creates dataset by name.
    Args:
        name: dataset name like MNIST, CIFAR10, ImageNeT, ''. '' means a customized dataset. Default: ''.
        root: dataset root dir. Default: './'.
        split: data split: '' or split name string (train/val/test), if it is '', no split is used.
            Otherwise, it is a subfolder of root dir, e.g., train, val, test. Default: 'train'.
        shuffle: whether to shuffle the dataset. Default: True.
        num_samples: Number of elements to sample (default=None, which means sample all elements).
        num_shards: Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id: The shard ID within `num_shards` (default=None).
            This argument can only be specified when `num_shards` is also specified.
        num_parallel_workers: Number of workers to read the data (default=None, set in the config).
        download: whether to download the dataset. Default: False
        num_aug_repeats: Number of dataset repeatition for repeated augmentation. If 0 or 1, repeated augmentation is diabled. Otherwise, repeated augmentation is enabled and the common choice is 3. (Default: 0)
    Note:
        For custom datasets and imagenet, the dataset dir should follow the structure like:
        .dataset_name/
        ├── split1/
        │  ├── class1/
        │  │   ├── 000001.jpg
        │  │   ├── 000002.jpg
        │  │   └── ....
        │  └── class2/
        │      ├── 000001.jpg
        │      ├── 000002.jpg
        │      └── ....
        └── split2/
           ├── class1/
           │   ├── 000001.jpg
           │   ├── 000002.jpg
           │   └── ....
           └── class2/
               ├── 000001.jpg
               ├── 000002.jpg
               └── ....
    Returns:
        Dataset object
    """
    assert (num_samples is None) or (num_aug_repeats==0), 'num_samples and num_aug_repeats can NOT be set together.'

    name = name.lower()
    # subset sampling
    if num_samples is not None and num_samples > 0:
        # TODO: rewrite ordered distributed sampler (subset sampling in distributed mode is not tested)
        if num_shards is not None and num_shards > 1: # distributed
            print('ns', num_shards, 'num_samples', num_samples)
            sampler = DistributedSampler(num_shards, shard_id, shuffle=shuffle, num_samples=num_samples)
        else: # standalone
            if shuffle:
                sampler = ds.RandomSampler(replacement=False, num_samples=num_samples)
            else:
                sampler = ds.SequentialSampler(num_samples=num_samples)
        mindspore_kwargs = dict(shuffle=None, sampler=sampler,
                            num_parallel_workers=num_parallel_workers, **kwargs)
    else:
        sampler = None
        mindspore_kwargs = dict(shuffle=shuffle, sampler=sampler, num_shards=num_shards, shard_id=shard_id,
                            num_parallel_workers=num_parallel_workers, **kwargs)
    if os.path.isdir(root):
        dataset_generator = LMDBDataSet(config, mode, logger, seed)
        dataset = GeneratorDataset(dataset_generator,**mindspore_kwargs)
    return dataset