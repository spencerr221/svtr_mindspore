from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals




class L2Decay(object):
    """
    L2 Weight Decay Regularization, which helps to prevent the model over-fitting.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        self.coeff = float(factor)

    def __call__(self):
        return self.coeff