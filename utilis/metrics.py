import torch

def dice_score(preds, targets, threshold=0.5, epsilon=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()