DATA_PATH = "sysmondataless.csv"

# ── sequence / transformer ─────────────────────────────────────────────────
SEQUENCE_LENGTH = 20
TRAIN_LABEL     = 0        # label used for benign-only training

BATCH_SIZE    = 64
EPOCHS        = 20
LEARNING_RATE = 1e-3

EMBED_DIM  = 128
NUM_HEADS  = 4
NUM_LAYERS = 2
FF_DIM     = 256

MODEL_PATH = "transformer_autoencoder.pt"
FEATURE_COLUMNS = []

# ── GNN encoder ────────────────────────────────────────────────────────────
GNN_EMBED_DIM   = 64        # embedding dim per node
GNN_EPOCHS      = 15
GNN_LR          = 1e-3
GNN_MODEL_PATH  = "gnn_encoder.pt"

# ── anomaly scoring weights  (must sum to 1.0) ────────────────────────────
RECON_WEIGHT  = 0.5         # transformer reconstruction error
GRAPH_WEIGHT  = 0.3         # GNN graph anomaly score
RARITY_WEIGHT = 0.2         # rare behaviour score

# ── alert aggregation ─────────────────────────────────────────────────────
ALERT_THRESHOLD = 0.6       # score threshold to flag an event
ALERT_WINDOW    = 300       # seconds: max gap to chain consecutive alerts
ALERTS_PATH     = "alerts.csv"
SCORES_PATH     = "anomaly_scores.csv"
