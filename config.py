import os

# ── directory layout ──────────────────────────────────────────────────────────
#
# DATA_DIR      Large-capacity volume for bulk data files.
#               Holds the input CSV, per-event score output, and the
#               memory-mapped sequence files that can reach tens of GB.
#
# ARTIFACTS_DIR Smaller code-adjacent drive for trained models, caches,
#               reports, and metric artefacts (typically < 1 GB total).
#
# Change either path here; every derived path below updates automatically.
#
DATA_DIR      = "/gulshan_data/anomaly_detection"
ARTIFACTS_DIR = "/home/rohit/elastic"


# ── data source ───────────────────────────────────────────────────────────────
# Default: elastic_data.csv produced by elastic_ingest.py.
# For the legacy Sysmon CSV: os.path.join(DATA_DIR, "sysmondataless.csv")
DATA_PATH = os.path.join(DATA_DIR, "elastic_data.csv")

# ── reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42                    # set in main.py for numpy / torch / python

# ── sequence / transformer ────────────────────────────────────────────────────
SEQUENCE_LENGTH = 20
TRAIN_LABEL     = 0                 # label value that denotes benign

BATCH_SIZE    = 512         # A5000 has 24 GB VRAM; 512 saturates the GPU
EPOCHS        = 20
LEARNING_RATE = 1e-3

# fraction of benign sequences held out for validation during transformer training
VAL_RATIO               = 0.15
EARLY_STOPPING_PATIENCE = 5         # epochs without val-loss improvement → stop

EMBED_DIM  = 96
NUM_HEADS  = 4
NUM_LAYERS = 2
FF_DIM     = 256

# trained model weights — small files, live on the code drive
MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "transformer_autoencoder.pt")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

FEATURE_COLUMNS = []

# ── transformer event filter ──────────────────────────────────────────────────
# Only EventID 1 (process creation) and 3 (network connection) carry strong
# attack signal for the transformer autoencoder.  Using all event types dilutes
# the training signal: the model learns to reconstruct benign module-loads,
# registry writes, and terminations equally well — so attack sequences look no
# different from benign ones (AUC ≈ 0.50).
HIGH_SIGNAL_EVENTIDS = [1, 3]       # process creation + network connection

# ── GNN encoder ───────────────────────────────────────────────────────────────
GNN_EMBED_DIM          = 64
GNN_EPOCHS             = 15
GNN_LR                 = 1e-3
GNN_MODEL_PATH         = os.path.join(ARTIFACTS_DIR, "gnn_encoder.pt")
GNN_EARLY_STOPPING_PAT = 5

# ── anomaly scoring weights  (must sum to 1.0) ────────────────────────────────
RECON_WEIGHT  = 0.45        # transformer reconstruction error
GRAPH_WEIGHT  = 0.00        # GNN graph anomaly score (disabled: ablation AUC=0.31 < random)
RARITY_WEIGHT = 0.55        # rare behaviour score

# ── alert aggregation ─────────────────────────────────────────────────────────
ALERT_THRESHOLD = 0.40      # score threshold to flag an event
ALERT_WINDOW    = 300       # seconds: max gap to chain consecutive alerts
ALERTS_PATH     = os.path.join(ARTIFACTS_DIR, "alerts.csv")

# per-event scores — one row per event, same scale as input → data volume
SCORES_PATH = os.path.join(DATA_DIR, "anomaly_scores.csv")

# ── ground-truth availability ─────────────────────────────────────────────────
# Set True only when the dataset contains manually verified Label=1 events
# (confirmed attacks).  When False the pipeline skips AUC/F1/ROC metrics
# (which are meaningless without ground truth) and instead writes a
# human-readable investigation report for manual triage.
HAS_GROUND_TRUTH = False

# ── investigation report (written when HAS_GROUND_TRUTH = False) ─────────────
# anomaly_report.txt  — narrative triage report grouped by host / alert chain
# flagged_events.csv  — flagged rows with full context, sorted by score
ANOMALY_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "anomaly_report.txt")
FLAGGED_EVENTS_PATH = os.path.join(ARTIFACTS_DIR, "flagged_events.csv")

# ── evaluation output (research-paper artefacts, HAS_GROUND_TRUTH = True) ────
METRICS_JSON_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
ROC_CURVE_PATH    = os.path.join(ARTIFACTS_DIR, "roc_curve.csv")
PR_CURVE_PATH     = os.path.join(ARTIFACTS_DIR, "pr_curve.csv")
SCORE_DIST_PATH   = os.path.join(ARTIFACTS_DIR, "score_distributions.csv")

# ── large-dataset scaling ─────────────────────────────────────────────────────
# Above this event count the pipeline switches to the disk-backed (memmap)
# path to avoid loading millions of sequences into RAM simultaneously.
LARGE_DATASET_THRESHOLD = 200_000

# Memory-mapped sequence files — potentially tens of GB, live on the data volume
SEQ_MEMMAP_PATH    = os.path.join(DATA_DIR, "sequences.dat")
LABELS_MEMMAP_PATH = os.path.join(DATA_DIR, "seq_labels.dat")

# Sequences processed per forward pass during inference
INFER_BATCH_SIZE = 2048

# ── semantic command-line embeddings ──────────────────────────────────────────
CMD_EMBED_MODEL        = "all-MiniLM-L6-v2"
CMD_EMBED_N_COMPONENTS = 32
CMD_EMBED_BATCH_SIZE   = 256
# Embedding cache can grow large (one .npy per unique command line) → data volume
CMD_EMBED_CACHE_DIR    = os.path.join(DATA_DIR, ".cmd_embed_cache")
CMD_EMBED_PCA_PATH     = os.path.join(ARTIFACTS_DIR, "cmd_pca.pkl")

# ── process chain embeddings ──────────────────────────────────────────────────
CHAIN_EMBED_DIM      = 32
CHAIN_MAX_DEPTH      = 4
CHAIN_W2V_WINDOW     = 4
CHAIN_W2V_MIN_COUNT  = 5
CHAIN_W2V_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "chain_w2v.model")
