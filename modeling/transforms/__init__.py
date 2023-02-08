
__all__ = ['build_transform']


def build_transform(config):
    from .stn import STN_ON
    from .tps_spatical_transformer import TPSSpatialTransformer

    support_dict = ['STN_ON']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'transform only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class