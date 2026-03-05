DATA_PATH = "sysmondataless.csv"

# ── reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42                    # set in main.py for numpy / torch / python

# ── sequence / transformer ─────────────────────────────────────────────────
SEQUENCE_LENGTH = 20
TRAIN_LABEL     = 0                 # label value that denotes benign

BATCH_SIZE    = 64
EPOCHS        = 20
LEARNING_RATE = 1e-3

# fraction of benign sequences held out for validation during transformer training
VAL_RATIO               = 0.15
EARLY_STOPPING_PATIENCE = 5         # epochs without val-loss improvement → stop

EMBED_DIM  = 128
NUM_HEADS  = 4
NUM_LAYERS = 2
FF_DIM     = 256

MODEL_PATH  = "transformer_autoencoder.pt"
SCALER_PATH = "scaler.pkl"          # StandardScaler fitted on benign features only

FEATURE_COLUMNS = []

# ── GNN encoder ────────────────────────────────────────────────────────────
GNN_EMBED_DIM            = 64
GNN_EPOCHS               = 15
GNN_LR                   = 1e-3
GNN_MODEL_PATH           = "gnn_encoder.pt"
GNN_EARLY_STOPPING_PAT   = 5        # patience for GNN training

# ── anomaly scoring weights  (must sum to 1.0) ────────────────────────────
RECON_WEIGHT  = 0.5         # transformer reconstruction error
GRAPH_WEIGHT  = 0.3         # GNN graph anomaly score
RARITY_WEIGHT = 0.2         # rare behaviour score

# ── alert aggregation ─────────────────────────────────────────────────────
ALERT_THRESHOLD = 0.6       # score threshold to flag an event
ALERT_WINDOW    = 300       # seconds: max gap to chain consecutive alerts
ALERTS_PATH     = "alerts.csv"
SCORES_PATH     = "anomaly_scores.csv"

# ── evaluation output (research-paper artefacts) ──────────────────────────
METRICS_JSON_PATH = "metrics.json"  # all scalar metrics; import directly into paper tables
ROC_CURVE_PATH    = "roc_curve.csv"  # fpr / tpr / threshold columns → Figure: ROC curve
PR_CURVE_PATH     = "pr_curve.csv"   # precision / recall / threshold  → Figure: PR curve
SCORE_DIST_PATH   = "score_distributions.csv"  # per-class score stats

# ── large-dataset scaling ─────────────────────────────────────────────────
# Above this event count the pipeline switches to the disk-backed (memmap)
# path to avoid loading millions of sequences into RAM simultaneously.
LARGE_DATASET_THRESHOLD = 200_000

# Sequences written to disk by build_sequences_memmap()
SEQ_MEMMAP_PATH    = "sequences.dat"
LABELS_MEMMAP_PATH = "seq_labels.dat"

# Sequences processed per forward pass during inference
INFER_BATCH_SIZE = 512
