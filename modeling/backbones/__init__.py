
__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    if model_type == "det" or model_type == "table":
        #
        # support_dict = [
        #     "MobileNetV3", "ResNet", "ResNet_vd", "ResNet_SAST", "PPLCNet"
        # ]
        print("not support yet")
        if model_type == "table":
            print("not support yet")
            # from .table_master_resnet import TableResNetExtra
            # support_dict.append('TableResNetExtra')
    elif model_type == "rec" or model_type == "cls":

        from .svtrnet import SVTRNet

        support_dict = [
            'SVTRNet'
        ]
    elif model_type == 'e2e':
        print("not support yet")
        # from .e2e_resnet_vd_pg import ResNet
        # support_dict = ['ResNet']
    elif model_type == 'kie':
        print("not support yet")
        # from .kie_unet_sdmgr import Kie_backbone
        # from .vqa_layoutlm import LayoutLMForSer, LayoutLMv2ForSer, LayoutLMv2ForRe, LayoutXLMForSer, LayoutXLMForRe
        # support_dict = [
        #     'Kie_backbone', 'LayoutLMForSer', 'LayoutLMv2ForSer',
        #     'LayoutLMv2ForRe', 'LayoutXLMForSer', 'LayoutXLMForRe'
        # ]

    elif model_type == 'table':
        print("not support yet")
        # from .table_resnet_vd import ResNet
        # from .table_mobilenet_v3 import MobileNetV3
        # support_dict = ['ResNet', 'MobileNetV3']
    else:
        raise NotImplementedError

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(model_type,
                                                                 support_dict))
    module_class = eval(module_name)(**config)
    return module_class