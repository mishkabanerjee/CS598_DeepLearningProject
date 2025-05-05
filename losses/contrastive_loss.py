import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_anchor, z_pos, z_neg):
        """
        z_anchor: (B, D) - representation of current window
        z_pos: (B, D) - representation of neighbor window (positive)
        z_neg: (B, D) - representation of negative samples (B, K, D)
        """
        B, D = z_anchor.size()
        K = z_neg.size(1)  # number of negative samples per anchor

        # Cosine similarity
        sim_pos = F.cosine_similarity(z_anchor, z_pos, dim=-1)  # (B,)
        sim_pos = sim_pos / self.temperature

        sim_neg = F.cosine_similarity(
            z_anchor.unsqueeze(1).expand(-1, K, -1), z_neg, dim=-1
        )  # (B, K)
        sim_neg = sim_neg / self.temperature

        # Combine for denominator
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # (B, 1+K)
        labels = torch.zeros(B, dtype=torch.long).to(z_anchor.device)  # positives at index 0

        loss = F.cross_entropy(logits, labels)
        return loss
