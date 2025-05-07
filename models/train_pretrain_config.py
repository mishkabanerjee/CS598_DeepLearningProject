# Configuration for TRACE Pretraining

class PretrainConfig:
    # Dataset
    data_dir = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy_old" 
    npy_dir = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\npy"
    general_table_path = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\reference_data\general_table.csv"    
    
    batch_size = 128
    max_len = 256

    # Model
    input_dim = 18
    hidden_dim = 128
    num_layers = 2
    num_heads = 4
    dropout = 0.1

    # Training
    num_epochs = 50
    learning_rate = 1e-3
    mask_ratio = 0.15

    # Saving
    save_dir = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt"
    save_every = 10  # Save model every 10 epochs

    """ Original TRACE configs
    # Dataset    
    batch_size = 32   

    # Training
    num_epochs = 300
    learning_rate = 5e-4  # 0.0005

    """

# # Dataset
#     data_dir = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\data\npy"
#     batch_size = 30
#     window_size = 12  # how many timepoints in each window
#     input_dim = 36     # number of variables in HiRID npy array

#     # Model (Causal CNN)
#     cnn_channels = 4
#     cnn_depth = 1
#     cnn_kernel_size = 2
#     encoding_size = 10  # latent dim, will be pruned later

#     # Training
#     num_epochs = 150
#     learning_rate = 5e-5
#     weight_decay = 5e-4  # optional
#     acf_nghd_threshold = 0.6
#     acf_out_threshold = 0.1

#     # Saving
#     save_dir = r"C:\Users\mishka.banerjee\Documents\DeepLearning_TRACE_Project\ckpt"
#     save_every = 10