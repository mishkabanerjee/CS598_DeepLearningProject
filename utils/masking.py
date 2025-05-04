import torch

def random_masking(x, mask_ratio=0.15):
    """
    Randomly mask parts of the input sequence.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, feature_dim)
        mask_ratio (float): Fraction of elements to mask

    Returns:
        masked_x: input with masked elements set to zero
        mask: binary mask (1 if masked, 0 otherwise)
    """
    batch_size, seq_len, feature_dim = x.shape  
    seq_len = x.size(1)   
    num_mask = int(seq_len * mask_ratio)


    mask = torch.zeros((batch_size, seq_len), device=x.device)

    for i in range(batch_size):
        mask_idx = torch.randperm(seq_len)[:num_mask]
        mask[i, mask_idx] = 1

    mask = mask.bool()
    masked_x = x.clone()
    masked_x[mask.unsqueeze(-1).expand_as(x)] = 0.0

    return masked_x, mask
