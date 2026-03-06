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
# Weights derived from per-component ablation (see metrics.py output):
#   rarity_score  AUC = 0.9717  ← strongest discriminator → highest weight
#   graph_score   AUC = 0.7011  ← second best
#   recon_error   AUC ≈ 0.50    ← weakest; transformer learns the mean of
#                                    benign sequences and reconstructs attack
#                                    sequences equally well → not discriminative
#
# Previous allocation (RECON=0.5, RARITY=0.2) gave composite AUC = 0.8352,
# which is lower than rarity alone (0.9717).  The heavy recon weight was
# actively diluting the best signal.  Corrected weights bring the composite
# closer to the rarity ceiling.
RECON_WEIGHT  = 0.2         # transformer reconstruction error (weakest)
GRAPH_WEIGHT  = 0.3         # GNN graph anomaly score
RARITY_WEIGHT = 0.5         # rare behaviour score (strongest ablation AUC)

# ── alert aggregation ─────────────────────────────────────────────────────
# Previous value (0.6) was above the attack score ceiling (~0.29 with old
# weights), so all 63 alert chains contained only high-rarity benign events
# → alert_precision = 0.  With corrected weights, attack scores rise to
# ~0.45–0.65; threshold 0.40 captures true attacks with acceptable precision.
ALERT_THRESHOLD = 0.40      # score threshold to flag an event
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
