import numpy as np
import os
import lmdb
import cv2
import string
import six
from PIL import Image

from svtr_mindspore.data.imug import transform, create_operators
# from mindspore import dataset

class LMDBDataSet():
    def __init__(self, config, mode,seed=None):
        super(LMDBDataSet, self).__init__()

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        batch_size = loader_config['batch_size_per_card']
        data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']

        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        self.data_idx_order_list = self.dataset_traversal()
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx",
                                                       1)

        ratio_list = dataset_config.get("ratio_list", [1.0])
        self.need_reset = True in [x < 1 for x in ratio_list]

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        # print("data_dir:",data_dir)
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            # print("1,2,3",dirpath,dirnames,filenames)
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {"dirpath":dirpath, "env":env, \
                    "txn":txn, "num_samples":num_samples}
                dataset_idx += 1
        # print("lmdb_sets:",lmdb_sets)
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        # print("data_idx_order_list:",data_idx_order_list)
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data"""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            lmdb_idx, file_idx = self.data_idx_order_list[np.random.randint(
                len(self))]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            sample_info = self.get_lmdb_sample_info(
                self.lmdb_sets[lmdb_idx]['txn'], file_idx)
            if sample_info is None:
                continue
            img, label = sample_info
            data = {'image': img, 'label': label}
            data = transform(data, load_data_ops)
            if data is None:
                continue
            ext_data.append(data)
        return ext_data

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'],
                                                file_idx)
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        img, label = sample_info
        # img = np.frombuffer(img, dtype='uint8')
        # img = cv2.imdecode(img, 1)
        # img = img[:, :, ::-1]
        #
        # label = []
        # for c in label_str:
        #     if c in self.label_dict:
        #         label.append(self.label_dict.index(c))
        # label_length = len(label)
        # label.extend([int(self.blank)] * (self.max_text_length - len(label)))
        # label = np.array(label)
        #
        # return img, label


        data = {'image': img, 'label': label}
        data['ext_data'] = self.get_ext_data()
        outs = transform(data, self.ops)
        print("outs",outs)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        # print("outs:",type(outs[0]),len(outs))
        #TODO: what is ext_data for?
        return outs[0],outs[1]

    def __len__(self):
        return self.data_idx_order_list.shape[0]

# if __name__ == "__main__":
#     import argparse
#     from svtr_mindspore.utils import load_config
#     data_dir = "/old/katekong/crnn/datasets/ocr-datasets/evaluation/IC03_860/"
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_path", type=str, default="../configs/rec_svtrnet.yaml", help="Config file path")
#     args = parser.parse_args()
#     config_path = args.config_path
#     # print("config_path:",config_path)
#     config = load_config(config_path)
#     # config = {"data_dir": data_dir}
#     dataset = LMDBDataSet(config,mode='Train')
#     datasample = dataset.__getitem__(1)
#     # print("datasample:",datasample)
#     # import pdb
#     #
#     # pdb.set_trace()
