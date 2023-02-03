
__all__ = ['build_head']


def build_head(config):
    # rec head
    from .ctc_head import CTCHead


    support_dict = [
        'CTCHead'
    ]

    #table head

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class