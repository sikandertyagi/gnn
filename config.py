DATA_PATH = "sysmondataless.csv"

SEQUENCE_LENGTH = 20

TRAIN_LABEL = 0

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

EMBED_DIM = 96
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256

VAL_RATIO = 0.2

MODEL_PATH = "transformer_autoencoder.pt"
SCALER_PATH = "scaler.pkl"

# Fixed seed for reproducible training shuffles and DataLoader workers
RANDOM_SEED = 42
