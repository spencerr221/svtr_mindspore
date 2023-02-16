import mindspore.nn as nn

class with_loss_cell(nn.Cell):
    def __init__(self, backbone, loss):
        super(with_loss_cell, self).__init__()
        self.backbone = backbone
        self.loss = loss

    def construct(self, data, label):
        logits = self.backbone(data, label)
        loss = self.loss(logits, label)
        return loss
