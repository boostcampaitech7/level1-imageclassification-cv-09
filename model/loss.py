import torch
import torch.nn as nn

class CELoss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)
    

class SmoothingLoss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(SmoothingLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        return self.loss_fn(outputs, targets)
    

# class FocalLoss(nn.Module):

#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         #gamma: Focal Loss의 주요 파라미터로, 잘 분류된 샘플의 손실을 줄이는 정도를 조절
#         self.gamma = gamma
#         #alpha: 클래스 가중치를 조절하는 파라미터
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
#         #size_average: 손실을 평균으로 계산할지 합산할지.
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)

#         logpt = nn.functional.log_softmax(input, dim=1)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = logpt.exp()

#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * at

#         loss = -1 * (1 - pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
        # else: return loss.sum()