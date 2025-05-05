import torch

# Load the full model checkpoint
checkpoint_path = "ckpt/trace_nocontrast_epoch50.pt"
full_state_dict = torch.load(checkpoint_path, map_location="cpu")

# Extract just the encoder weights
encoder_state_dict = {k.replace("encoder.", ""): v for k, v in full_state_dict.items() if k.startswith("encoder.")}

# Save the encoder-only weights
torch.save(encoder_state_dict, "ckpt/trace_nocontrast_encoder_epoch50.pt")
print("✅ Saved cleaned encoder weights to ckpt/trace_nocontrast_encoder_epoch50.pt")
