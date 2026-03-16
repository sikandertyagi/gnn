# ── data source ────────────────────────────────────────────────────────────
# Change to "elastic_data.csv" after running elastic_ingest.py, or pass the
# output_path you set in elastic_config.yml.
DATA_PATH = "sysmondataless.csv"

# ── reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42                    # set in main.py for numpy / torch / python

# ── sequence / transformer ─────────────────────────────────────────────────
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

MODEL_PATH  = "transformer_autoencoder.pt"
SCALER_PATH = "scaler.pkl"          # StandardScaler fitted on benign features only

FEATURE_COLUMNS = []

# ── transformer event filter ───────────────────────────────────────────────
# Only EventID 1 (process creation) and 3 (network connection) carry strong
# attack signal for the transformer autoencoder.  Using all event types dilutes
# the training signal: the model learns to reconstruct benign module-loads,
# registry writes, and terminations equally well — so attack sequences look no
# different from benign ones (AUC ≈ 0.50).
#
# The GNN and rarity engine are NOT filtered — they benefit from the full
# event graph (parent-child trees, user context, host behaviour).
HIGH_SIGNAL_EVENTIDS = [1, 3]       # process creation + network connection

# ── GNN encoder ────────────────────────────────────────────────────────────
GNN_EMBED_DIM            = 64
GNN_EPOCHS               = 15
GNN_LR                   = 1e-3
GNN_MODEL_PATH           = "gnn_encoder.pt"
GNN_EARLY_STOPPING_PAT   = 5        # patience for GNN training

# ── anomaly scoring weights  (must sum to 1.0) ────────────────────────────
# NOTE: the old ablation AUCs below were measured on a broken pipeline:
#   · recon_error AUC ≈ 0.50 was an artefact of two bugs —
#       (a) multi-host alignment error scrambling sequence→event mapping
#       (b) rare_process_score / has_ip / has_download / has_encodedcommand /
#           eventid features missing from the transformer input
#     Both bugs are now fixed.  Re-run main.py and read roc_auc_recon_error
#     from metrics.json to derive updated weights.
#
# Interim weights give the transformer equal footing with the rarity engine
# pending re-derivation from fresh ablation results:
#   rarity_score  AUC = 0.9717  (old; expected similar)
#   graph_score   AUC = 0.7011  (old; expected similar)
#   recon_error   AUC ≈ 0.50    (old; expected much higher after fixes)
RECON_WEIGHT  = 0.45        # transformer reconstruction error
GRAPH_WEIGHT  = 0.00        # GNN graph anomaly score (disabled: ablation AUC=0.31 < random)
RARITY_WEIGHT = 0.55        # rare behaviour score

# ── alert aggregation ─────────────────────────────────────────────────────
# Previous value (0.6) was above the attack score ceiling (~0.29 with old
# weights), so all 63 alert chains contained only high-rarity benign events
# → alert_precision = 0.  With corrected weights, attack scores rise to
# ~0.45–0.65; threshold 0.40 captures true attacks with acceptable precision.
ALERT_THRESHOLD = 0.40      # score threshold to flag an event
ALERT_WINDOW    = 300       # seconds: max gap to chain consecutive alerts
ALERTS_PATH     = "alerts.csv"
SCORES_PATH     = "anomaly_scores.csv"

# ── ground-truth availability ─────────────────────────────────────────────
# Set True only when the dataset contains manually verified Label=1 events
# (confirmed attacks).  When False the pipeline skips AUC/F1/ROC metrics
# (which are meaningless without ground truth) and instead writes a
# human-readable investigation report for manual triage.
HAS_GROUND_TRUTH = False

# ── investigation report (written when HAS_GROUND_TRUTH = False) ──────────
# anomaly_report.txt  — narrative triage report grouped by host / alert chain
# flagged_events.csv  — all flagged event rows with full context fields,
#                       sorted by anomaly score for spreadsheet review
ANOMALY_REPORT_PATH  = "anomaly_report.txt"
FLAGGED_EVENTS_PATH  = "flagged_events.csv"

# ── evaluation output (research-paper artefacts, HAS_GROUND_TRUTH = True) ─
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
INFER_BATCH_SIZE = 2048

# ── semantic command-line embeddings ───────────────────────────────────────────
# SentenceTransformer model: 384-d output, ~22 M params, CPU-friendly.
CMD_EMBED_MODEL        = "all-MiniLM-L6-v2"
# PCA target dimensionality (384 → 32).
CMD_EMBED_N_COMPONENTS = 32
# Sentences encoded per forward pass; 256 balances throughput vs peak RAM.
CMD_EMBED_BATCH_SIZE   = 256
# Directory for SHA-256 keyed raw-embedding cache files (.npy).
CMD_EMBED_CACHE_DIR    = ".cmd_embed_cache"
# Path to the fitted PCA model (joblib); created on first run, reused after.
CMD_EMBED_PCA_PATH     = "cmd_pca.pkl"

# ── process chain embeddings ─────────────────────────────────────────────
# Word2Vec trained on process ancestry chains (ProcessGuid → ParentProcessGuid).
CHAIN_EMBED_DIM       = 32          # Word2Vec vector_size
CHAIN_MAX_DEPTH       = 4           # max ancestors to walk per event
CHAIN_W2V_WINDOW      = 4           # Word2Vec context window
CHAIN_W2V_MIN_COUNT   = 5           # minimum frequency to keep a process name
CHAIN_W2V_MODEL_PATH  = "chain_w2v.model"  # saved gensim Word2Vec model
