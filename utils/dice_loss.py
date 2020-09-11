import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(input.shape[0]).cuda().zero_()
    else:
        s = torch.FloatTensor(input.shape[0]).zero_()

    for i, c in enumerate(zip(input, target)):
        s[i] = DiceCoeff().forward(c[0], c[1])

    return s

# class DiceLoss(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, pred, target):

#         pred = pred.squeeze(dim=1)

#         smooth = 1

#         # dice系数的定义
#         dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
#                                             target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

#         # 返回的是dice距离
#         return torch.clamp((1 - dice).mean(), 0, 1)
