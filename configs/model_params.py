import torch

'''
Training, 1DUnet and Sampler parameters
'''

SR = 16_384 # Sampling rate used in traninig
CROP_S = 4 # Cropped audio signal length in seconds used in training
L_C = 2 # Left context length in seconds
R_C = 1.8 # Right context length in seconds

TRAIN_TEST_SPLIT_RATIO = 0.85
BATCH_SIZE = 50 # Batch size used in training and evaluation
EPOCHS = 15 # Epochs
CKPT_INT = 5 # Checkpoint and evaluation rate during training
LR = 1e-4 # Learning rate
WARMUP_STEPS = 5_000 # Number of LinearLR scheduler steps at the start
ACCUM_GRADS = 380 # Gradients accumulated before optimizer.step()
ACCUM_ITER = ACCUM_GRADS // BATCH_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Auto device selection

IN_CHANNELS = 1
CHANNELS = [8, 32, 64, 128, 256, 512, 512] # U-Net: number of input/output (audio) channels
FACTORS = [1, 4, 4, 4, 2, 2, 2] # U-Net: channels at each layer
ITEMS = [1, 2, 2, 2, 2, 2, 2] # U-Net: number of repeating items at each layer
ATTENTIONS = [0, 0, 0, 0, 0, 1, 1] # U-Net: attention enabled/disabled at each layer
ATTENTION_HEADS = 8 # U-Net: number of attention heads per attention item
ATTENTION_FEATURES = 64 # U-Net: number of attention features per attention item

EMBEDDING_EXTRACTOR_MODEL = None # Audio feature extractor model instance, if None - VGGish pytorch port is used
NUM_STEPS_FOR_EVALUATION = 10 # Defines number of steps made in sampling procedure during evaluation

CHECKPOINTS_OUT_DIR = None # Directory path to saved checkpoints
MODEL_DIR = '.' # Directory path to models