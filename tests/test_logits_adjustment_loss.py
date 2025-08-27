import torch

from hurricore.criterions import LogitsAdjustmentLoss


def test_logits_adjustment_loss():
    dummy_logits = torch.tensor([[1, 50, 3, 99]])
    class_prob_dist = [0.1, 0.2, 0.3, 0.4]
    print(LogitsAdjustmentLoss(class_prob_dist)(dummy_logits, torch.tensor([1])))


if __name__ == "__main__":  
    test_logits_adjustment_loss()
