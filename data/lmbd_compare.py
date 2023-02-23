

class LMDBDataset():
    def __init__(self, config):
        super(LMDBDataset, self).__init__()

        data_dir = config["data_dir"]

        self.lmdb_sets = self.load_list_of_hierarchical_lmdb_dataset(data_dir)
        self.data_idx_order_list = self.dataset_traversal()

        self.max_text_length = config.get("max_text_length")
        self.blank = config.get("blank")
        self.class_num = config.get("class_num")
        self.label_dict = config.get("label_dict")

    def load_list_of_hierarchical_lmdb_dataset(self, data_dirs):
        if isinstance(data_dirs, str):
            results = self.load_hierarchical_lmdb_dataset(data_dirs)
        elif isinstance(data_dirs, list):
            results = {}
            for data_dir in data_dirs:
                start_idx = len(results)
                lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir, start_idx)
                results.update(lmdb_sets)
        else:
            results = {}

        return results

    def load_hierarchical_lmdb_dataset(self, data_dir, start_idx=0):

        lmdb_sets = {}
        dataset_idx = start_idx
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
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
                lmdb_sets[dataset_idx] = {"dirpath": dirpath, "env": env, \
                                          "txn": txn, "num_samples": num_samples}
                dataset_idx += 1
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
        # load_data_ops = self.ops[:self.ext_op_transform_idx]
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
            # data = transform(data, load_data_ops)
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
        img, label_str = sample_info
        label_str = normalize_text(label_str)

        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)
        img = img[:, :, ::-1]

        label = []
        for c in label_str:
            if c in self.label_dict:
                label.append(self.label_dict.index(c))
        label_length = len(label)
        if label_length >= self.max_text_length:
            label = label[:self.max_text_length]
        else:
            label.extend([int(self.blank)] * (self.max_text_length - len(label)))
        label = np.array(label)

        return img, label

    def __len__(self):
        return self.data_idx_order_list.shape[0]