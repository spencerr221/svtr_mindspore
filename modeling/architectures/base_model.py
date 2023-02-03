from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mindspore import nn
from ..transforms import build_transform
from ..backbones import build_backbone
from ..necks import build_neck
from ..head import build_head

__all__ = ['BaseModel']


class BaseModel(nn.Cell):
    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get('in_channels', 3)
        model_type = config['model_type']
        # build transfrom,
        # for rec, transfrom can be TPS,None
        # for det and cls, transfrom shoule to be None,
        # if you make model differently, you can use transfrom in det and cls
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for del, rec and cls
        if 'Backbone' not in config or config['Backbone'] is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            config["Backbone"]['in_channels'] = in_channels
            self.backbone = build_backbone(config["Backbone"], model_type)
            in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels

        # # build head, head is need for det, rec and cls
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            config["Head"]['in_channels'] = in_channels
            self.head = build_head(config["Head"])

        self.return_all_feats = config.get("return_all_feats", False)

    def construct(self, x, data=None):

        y = dict()
        if self.use_transform:
            print("before transformer:",x.shape)    #before transformer: [64, 3, 64, 256]
            x = self.transform(x)
            print("after transformer:", x.shape)   #after transformer: [64, 3, 32, 100]
        if self.use_backbone:
            print("before backbone:",x.shape)   #before backbone: [64, 3, 32, 100]
            x = self.backbone(x)
            print("after backbone:", x.shape)    #after backbone: [64, 192, 1, 25]
        if isinstance(x, dict):
            y.update(x)
        else:
            y["backbone_out"] = x
        final_name = "backbone_out"
        if self.use_neck:
            print("before neck:",x.shape)   #before neck: [64, 192, 1, 25]
            x = self.neck(x)
            print("after neck:",x.shape)  #after neck: [64, 25, 192]
            if isinstance(x, dict):
                y.update(x)
            else:
                y["neck_out"] = x
            final_name = "neck_out"
        if self.use_head:
            print("before head:",x.shape)   # before head: [64, 25, 192]
            x = self.head(x, targets=data)  # after head: [64, 25, 37]
            print("after head:",x.shape)
            # for multi head, save ctc neck out for udml
            if isinstance(x, dict) and 'ctc_neck' in x.keys():
                y["neck_out"] = x["ctc_neck"]
                y["head_out"] = x
            elif isinstance(x, dict):
                y.update(x)
            else:
                y["head_out"] = x
            final_name = "head_out"
        if self.return_all_feats:
            if self.training:     #TODO False by default
                return y
            elif isinstance(x, dict):
                return x
            else:
                return {final_name: x}
        else:
            return x

#
# class BaseModelWithLoss(nn.Cell):

