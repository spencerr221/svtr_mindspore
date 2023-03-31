from addict import Dict
from mindspore import nn
from .transforms import build_trans
from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head

__all__ = ['BaseModel']

class BaseModel(nn.Cell):
    def __init__(self, config: dict):
        """
        Args:
            config (dict): model config 
        """
        super(BaseModel, self).__init__()

        config = Dict(config)
        in_channels = config.get('in_channels', 3)

        if 'transform' not in config or config['transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['transform']['in_channels'] = in_channels
            trans_name = config.transform.pop('name')
            self.transform = build_trans(trans_name, **config.transform)
            assert hasattr(self.transform, 'out_channels'), f'Transforms are required to provide out_channels attribute, but not found in {trans_name}'
            in_channels = self.transform.out_channels

        backbone_name = config.backbone.pop('name')
        config['backbone']['in_channels'] = in_channels
        self.backbone = build_backbone(backbone_name, **config.backbone)
        assert hasattr(self.backbone, 'out_channels'), f'Backbones are required to provide out_channels attribute, but not found in {backbone_name}'

        if 'neck' not in config or config.neck is None:
            neck_name = 'Select'
        else:
            neck_name = config.neck.pop('name')
        self.neck = build_neck(neck_name, in_channels=self.backbone.out_channels, **config.neck)

        assert hasattr(self.neck, 'out_channels'), f'Necks are required to provide out_channels attribute, but not found in {neck_name}'

        head_name = config.head.pop('name')
        self.head = build_head(head_name, in_channels=self.neck.out_channels, **config.head)

        self.model_name = f'{backbone_name}_{neck_name}_{head_name}'  

    def construct(self, x):
        # TODO: return bout, hout for debugging, using a dict.
        if self.use_transform:
            tout = self.transform(x)
        else:
            tout = x
        bout = self.backbone(tout)

        nout = self.neck(bout)

        hout = self.head(nout)

        # resize back for postprocess 
        #y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)

        # for multi head, save ctc neck out for udml
        '''
        if isinstance(x, dict) and 'ctc_neck' in x.keys():
            y["neck_out"] = x["ctc_neck"]
            y["head_out"] = x
        elif isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        
        
        '''
        
        return hout

# def parse_args():
#     parser = argparse.ArgumentParser(description='Training Config', add_help=False)
#     parser.add_argument('-c', '--config', type=str, default='/home/mindocr/lby_spencer/mindocr/mindocr/configs/rec/svtr_tiny.yaml',
#                         help='YAML config file specifying default arguments (default='')')
#     args = parser.parse_args()

#     return args


if __name__=='__main__':
    # model_config = {
    #         "backbone": {
    #             'name': 'det_resnet50',
    #             'pretrained': False 
    #             },
    #         "neck": {
    #             "name": 'FPN',
    #             "out_channels": 256,
    #             },
    #         "head": {
    #             "name": 'ConvHead',
    #             "out_channels": 2,
    #             "k": 50
    #             }
            
    #         }
    # model_config.pop('neck')
    import pdb; pdb.set_trace()
    args = parse_args()
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        config = yaml.safe_load(fp)
    config = Dict(config)
    model_config = config.model
    model = BaseModel(model_config) 

    import mindspore as ms
    import time
    import numpy as np

    bs = 8
    x = ms.Tensor(np.random.rand(bs, 3, 640, 640), dtype=ms.float32)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    def predict(model, x):
        start = time.time()
        y = model(x)
        print(time.time()-start)
        print(y.shape)

    predict(model, x)
