# Configuration for TRACE Pretraining

class PretrainConfig:
    # Dataset
    data_dir = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\npy"  # <-- your npy folder
    batch_size = 128
    max_len = 256

    # Model
    input_dim = 18
    hidden_dim = 128
    num_layers = 2
    num_heads = 4
    dropout = 0.1

    # Training
    num_epochs = 300
    learning_rate = 1e-3
    mask_ratio = 0.15

    # Saving
    save_dir = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt"
    save_every = 10  # Save model every 10 epochs
