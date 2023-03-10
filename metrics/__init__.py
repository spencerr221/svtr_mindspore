from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ["build_metric"]

from .rec_metric import SVTRAccuracy



def build_metric(config, **kwargs):
    support_dict = ["RecMetric","SVTRAccuracy"]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict))
    module_class = eval(module_name)(**config, **kwargs)
    # module_class = module_name()
    return module_class