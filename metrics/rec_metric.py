# from rapidfuzz.distance import Levenshtein
# # from difflib import SequenceMatcher
#
# import numpy as np
# import string
# from mindspore import nn
#
# class RecMetric(nn.Metric):
#     def __init__(self,
#                  main_indicator='acc',
#                  is_filter=False,
#                  ignore_space=True,
#                  **kwargs):
#         self.main_indicator = main_indicator
#         self.is_filter = is_filter
#         self.ignore_space = ignore_space
#         self.eps = 1e-5
#         self.clear()
#
#     def _normalize_text(self, text):
#         text = ''.join(
#             filter(lambda x: x in (string.digits + string.ascii_letters), text))
#         return text.lower()
#
#     def update(self, pred_label, *args, **kwargs):
#         preds, labels = pred_label
#         correct_num = 0
#         all_num = 0
#         norm_edit_dis = 0.0
#         for (pred, pred_conf), (target, _) in zip(preds, labels):
#             if self.ignore_space:
#                 pred = pred.replace(" ", "")
#                 target = target.replace(" ", "")
#             if self.is_filter:
#                 pred = self._normalize_text(pred)
#                 target = self._normalize_text(target)
#             norm_edit_dis += Levenshtein.normalized_distance(pred, target)
#             if pred == target:
#                 correct_num += 1
#             all_num += 1
#         self.correct_num += correct_num
#         self.all_num += all_num
#         self.norm_edit_dis += norm_edit_dis
#         # return {
#         #     'acc': correct_num / (all_num + self.eps),
#         #     'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps)
#         # }
#
#
#     def eval(self):
#         """
#         return metrics {
#                  'acc': 0,
#                  'norm_edit_dis': 0,
#             }
#         """
#         acc = 1.0 * self.correct_num / (self.all_num + self.eps)
#         norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
#         self.clear()
#         return {'acc': acc, 'norm_edit_dis': norm_edit_dis}
#
#     def clear(self):
#         self.correct_num = 0
#         self.all_num = 0
#         self.norm_edit_dis = 0



import string
import numpy as np
from mindspore import nn
import Levenshtein


class SVTRAccuracy(nn.Metric):
    """
    Define accuracy metric for warpctc network.
    """

    def __init__(self, decoder, print_flag=True):
        super(SVTRAccuracy, self).__init__()
        self._correct_num = 0
        self._total_num = 0
        self.print_flag = print_flag
        self.decoder = decoder
        self.label_dict = self.decoder.character

    def clear(self):
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are a `Tensor`, a list or an array.
                For the 'classification' evaluation type, `y_pred` is in most cases (not strictly) a list
                of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. Shape of `y` can be :math:`(N, C)` with values 0 and 1 if one-hot
                encoding is used or the shape is :math:`(N,)` with integer values if index of category is used.
                For 'multilabel' evaluation type, `y_pred` and `y` can only be one-hot encoding with
                values 0 or 1. Indices with 1 indicate the positive category. The shape of `y_pred` and `y`
                are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError(
                'Accuracy need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        # import pdb;pdb.set_trace()
        y_pred = self._convert_data(inputs[0])
        # y_pred = np.transpose(y_pred, (1, 0, 2))

        if isinstance(inputs[1], list) and isinstance(inputs[1][0], str):
            str_pred = self.decoder(y_pred)
            str_label = [x.lower() for x in inputs[1]]
        else:
            y = self._convert_data(inputs[1])
            str_pred, str_label = self.decoder(y_pred, y)

        for pred, label in zip(str_pred, str_label):
            if self.print_flag:
                print(pred, " :: ", label)
            edit_distance = Levenshtein.distance(pred, label)
            self._total_num += 1
            if edit_distance == 0:
                self._correct_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError(
                'Accuary can not be calculated, because the number of samples is 0.')
        print('correct num: ', self._correct_num,
              ', total num: ', self._total_num)
        sequence_accurancy = self._correct_num / self._total_num
        return sequence_accurancy