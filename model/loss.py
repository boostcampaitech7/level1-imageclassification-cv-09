import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        alpha: 클래스 불균형 조정을 위한 가중치. (기본값: 1)
        gamma: 손실을 더 집중시키는 파라미터. (기본값: 2)
        reduction: 손실값을 처리하는 방식. 'mean', 'sum' 또는 'none' 중 하나.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: 모델의 출력 (Logits)
        targets: 실제 레이블 (One-hot encoding이 아닌 class index 형태)
        """
        # Cross entropy loss 계산
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        
        # 예측 확률을 계산
        pt = torch.exp(-BCE_loss)  # p_t = e^(-BCE)
        
        # Focal Loss 계산
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # Reduction 방식에 따라 손실값 반환
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss