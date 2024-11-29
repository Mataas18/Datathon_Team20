import torch

import torch.nn as nn

# from ..metric_files.helper import _CYME

class CYMELoss(nn.Module):
    def __init__(self):
        super(CYMELoss, self).__init__()

    def forward(self, input, target):
        # Example: Mean Squared Error with L2 regularization
        mse_loss = nn.MSELoss()(input, target)
        cyme_loss = nn.L1Loss()(input, target)
        
        return 0.25 * mse_loss + 0.75 * cyme_loss

# Example usage:
# criterion = CustomLoss()
# loss = criterion(predictions, targets)