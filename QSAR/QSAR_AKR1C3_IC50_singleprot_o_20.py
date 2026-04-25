# =========================
# Part B: Load cleaned CSV, standardize SMILES, deduplicate by Molecule ChEMBL ID (median pIC50_value)
# =========================

import pandas as pd
import numpy as np

# ---- RDKit: use if available; only install if missing ----
try:
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    print("RDKit available.")
except Exception as e:
    print("RDKit not found. Trying to install 'rdkit'...")
    !pip -q install rdkit
    from rdkit import Chem
    from rdkit.Chem.SaltRemover import SaltRemover
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    print("RDKit installed.")

from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]
df_raw = pd.read_csv(filename)

print(f"Loaded {len(df_raw)} rows from: {filename}")
print("Columns:", df_raw.columns.tolist())

# ---- Required: first two columns = ID, Smiles; third column = pIC50_value (per your new format) ----
# If your column names differ, we still take by position safely.
if df_raw.shape[1] < 3:
    raise ValueError("CSV must have at least 3 columns: [ChEMBL_ID, Smiles, pIC50_value].")

id_col = df_raw.columns[0]
smi_col = df_raw.columns[1]
pic50_col = df_raw.columns[2]  # third col is pIC50_value

df = df_raw[[id_col, smi_col, pic50_col]].copy()
df = df.rename(columns={id_col: "ChEMBL_ID", smi_col: "Smiles", pic50_col: "pIC50_value"})

# ---- Clean pIC50_value ----
df["pIC50_value"] = pd.to_numeric(df["pIC50_value"], errors="coerce")
df = df.dropna(subset=["ChEMBL_ID", "Smiles", "pIC50_value"]).copy()

# Optional: remove extreme/unrealistic pIC50 if you want (comment out if not needed)
# df = df[(df["pIC50_value"] > 0) & (df["pIC50_value"] < 14)].copy()

print("After pIC50 cleaning:", len(df), "rows")

# ---- Parseable SMILES filter ----
def is_parseable(s):
    try:
        if pd.isna(s):
            return False
        return Chem.MolFromSmiles(str(s)) is not None
    except:
        return False

df = df[df["Smiles"].apply(is_parseable)].copy()
print("After SMILES parse filter:", len(df), "rows")

# ---- Standardize SMILES ----
salt_remover = SaltRemover()
uncharger = rdMolStandardize.Uncharger()

def standardize_smiles(smi: str):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    mol = uncharger.uncharge(mol)
    return Chem.MolToSmiles(mol, canonical=True)

df["SMILES"] = df["Smiles"].apply(standardize_smiles)
df = df.dropna(subset=["SMILES"]).copy()

# ---- Deduplicate by ChEMBL_ID using median pIC50_value ----
# For SMILES: pick mode (most frequent standardized SMILES); if tie -> first
def mode_first(x: pd.Series):
    vc = x.value_counts(dropna=True)
    if len(vc) == 0:
        return np.nan
    return vc.index[0]

before = len(df)
dup_id_count = (df["ChEMBL_ID"].value_counts() > 1).sum()
print("ChEMBL_ID with >=2 rows:", int(dup_id_count))

# Track how many SMILES variants per ID (useful QC)
variants = df.groupby("ChEMBL_ID")["SMILES"].nunique().rename("n_smiles_variants").reset_index()

df_grouped = (
    df.groupby("ChEMBL_ID", as_index=False)
      .agg(
          SMILES=("SMILES", mode_first),
          pIC50_value=("pIC50_value", "median"),
          n_records=("pIC50_value", "size")
      )
      .merge(variants, on="ChEMBL_ID", how="left")
)

after = len(df_grouped)
print(f"After deduplication by ChEMBL_ID: {before} -> {after} rows")
print("IDs with >1 SMILES variants (after standardization):", int((df_grouped["n_smiles_variants"] > 1).sum()))

# Keep final curated table
df_curated = df_grouped[["ChEMBL_ID", "SMILES", "pIC50_value"]].copy()

out_name = "AKR1C3_pIC50_curated_uniqueID.csv"
df_curated.to_csv(out_name, index=False)
print("Saved:", out_name)

files.download(out_name)






# =========================
# Part C: SMILES -> descriptors + ECFP4, build df_features
# =========================

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd

df_curated = df_curated.copy()

required_cols = ["ChEMBL_ID", "SMILES", "pIC50_value"]
missing = [c for c in required_cols if c not in df_curated.columns]
if missing:
    raise ValueError(f"Missing columns in df_curated: {missing}")

df_curated["Mol"] = df_curated["SMILES"].apply(Chem.MolFromSmiles)
df_curated = df_curated[df_curated["Mol"].notna()].reset_index(drop=True)

# ---- RDKit 2D descriptors ----
desc_names = [d[0] for d in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

def get_descriptors(mol):
    try:
        return list(calc.CalcDescriptors(mol))
    except Exception:
        return [np.nan] * len(desc_names)

desc_values = df_curated["Mol"].apply(get_descriptors).tolist()
df_desc = pd.DataFrame(desc_values, columns=desc_names)

# Clean descriptor values: inf -> nan; drop all-nan columns; fill numeric nan with median
df_desc = df_desc.replace([np.inf, -np.inf], np.nan)
df_desc = df_desc.dropna(axis=1, how="all")
for col in df_desc.columns:
    if pd.api.types.is_numeric_dtype(df_desc[col]):
        df_desc[col] = df_desc[col].fillna(df_desc[col].median())

# ---- ECFP4 (Morgan radius=2) ----
def get_ecfp4(mol, radius=2, nBits=2048):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return list(fp)
    except Exception:
        return [0] * nBits

ecfp_values = df_curated["Mol"].apply(get_ecfp4).tolist()
df_ecfp = pd.DataFrame(ecfp_values, columns=[f"ECFP4_{i}" for i in range(2048)])

# ---- Combine features ----
df_features = pd.concat(
    [
        df_curated[["ChEMBL_ID", "SMILES"]].reset_index(drop=True),
        df_desc.reset_index(drop=True),
        df_ecfp.reset_index(drop=True),
        df_curated[["pIC50_value"]].reset_index(drop=True),
    ],
    axis=1
)

print("Feature matrix shape:", df_features.shape)
print("Target column: pIC50_value")
print(df_features.head(3))

df_features.to_csv("AKR1C3_features_raw.csv", index=False)
print("Saved: AKR1C3_features_raw.csv")
files.download("AKR1C3_features_raw.csv")








# =========================
# Part E: Train-test split (stratified by y quantile bins)
# =========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET_COL = "pIC50_value"

meta_cols = ["ChEMBL_ID", "SMILES", TARGET_COL]
X_all = df_features.drop(columns=meta_cols).copy()
y_all = df_features[TARGET_COL].copy()

# Stratify bins using qcut (better balanced than cut)
n_bins = min(5, y_all.nunique())
y_binned = pd.qcut(y_all, q=n_bins, labels=False, duplicates="drop")

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_all, y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_binned
)

df_train_raw = pd.concat([df_features.loc[X_train_raw.index, ["ChEMBL_ID","SMILES"]].reset_index(drop=True),
                          X_train_raw.reset_index(drop=True),
                          y_train_raw.reset_index(drop=True)], axis=1)

df_test_raw = pd.concat([df_features.loc[X_test_raw.index, ["ChEMBL_ID","SMILES"]].reset_index(drop=True),
                         X_test_raw.reset_index(drop=True),
                         y_test_raw.reset_index(drop=True)], axis=1)

print(f"Training: {len(df_train_raw)} compounds")
print(f"Testing:  {len(df_test_raw)} compounds")
print("Train y mean:", float(y_train_raw.mean()))
print("Test  y mean:", float(y_test_raw.mean()))







# =========================
# Part D: Removing Noise (constant features + correlation filter), then scale (fit on train only)
# =========================

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

TARGET_COL = "pIC50_value"
LEAK_COLS = ["IC50_value"]  # just in case; will be ignored if not present

# Separate X/y from raw splits
meta_cols = ["ChEMBL_ID", "SMILES", TARGET_COL]
Xtr = df_train_raw.drop(columns=meta_cols, errors="ignore").copy()
ytr = df_train_raw[TARGET_COL].copy()

Xte = df_test_raw.drop(columns=meta_cols, errors="ignore").copy()
yte = df_test_raw[TARGET_COL].copy()

# Safety: remove leakage columns if exist
Xtr = Xtr.drop(columns=[c for c in LEAK_COLS if c in Xtr.columns], errors="ignore")
Xte = Xte.drop(columns=[c for c in LEAK_COLS if c in Xte.columns], errors="ignore")

# 1) Remove constant features (threshold=0.0)
vt = VarianceThreshold(threshold=0.0)
Xtr_v = vt.fit_transform(Xtr)
Xte_v = vt.transform(Xte)

kept_cols = Xtr.columns[vt.get_support()].tolist()
Xtr_v = pd.DataFrame(Xtr_v, columns=kept_cols)
Xte_v = pd.DataFrame(Xte_v, columns=kept_cols)

print("After constant filter:", Xtr_v.shape[1], "features")

# 2) Correlation filter (train only) — drop features with |r| > 0.95
corr = Xtr_v.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if (upper[col] > 0.95).any()]

Xtr_f = Xtr_v.drop(columns=to_drop)
Xte_f = Xte_v.drop(columns=to_drop)

print(f"After correlation filter: {Xtr_f.shape[1]} features (dropped {len(to_drop)})")

# 3) Scale (fit scaler on train only)
scaler = StandardScaler()
Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr_f), columns=Xtr_f.columns)
Xte_s = pd.DataFrame(scaler.transform(Xte_f), columns=Xte_f.columns)

# Reassemble processed train/test
df_train = pd.concat(
    [df_train_raw[["ChEMBL_ID","SMILES"]].reset_index(drop=True),
     Xtr_s.reset_index(drop=True),
     ytr.reset_index(drop=True)],
    axis=1
)

df_test = pd.concat(
    [df_test_raw[["ChEMBL_ID","SMILES"]].reset_index(drop=True),
     Xte_s.reset_index(drop=True),
     yte.reset_index(drop=True)],
    axis=1
)

print("Processed train shape:", df_train.shape)
print("Processed test  shape:", df_test.shape)

df_train.to_csv("AKR1C3_train_processed.csv", index=False)
df_test.to_csv("AKR1C3_test_processed.csv", index=False)
print("Saved: AKR1C3_train_processed.csv, AKR1C3_test_processed.csv")
files.download("AKR1C3_train_processed.csv")
files.download("AKR1C3_test_processed.csv")







# =========================
# Part F (FIXED): Train 20 unique models (no duplicates)
# - Fix CNN/RNN/LSTM/UNet input shapes
# - Keep B–E unchanged; assumes df_train/df_test already exist
# Output:
#   - F_compare_20models.csv
#   - fitted_models (dict)
#   - thr_pic50 (float) balanced threshold (1 decimal)
# =========================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---- sklearn regressors ----
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge

# ---- xgboost/lightgbm ----
try:
    import xgboost as xgb
    print("xgboost available.")
except:
    print("installing xgboost...")
    !pip -q install xgboost
    import xgboost as xgb
    print("xgboost installed.")

try:
    import lightgbm as lgb
    print("lightgbm available.")
except:
    print("installing lightgbm...")
    !pip -q install lightgbm
    import lightgbm as lgb
    print("lightgbm installed.")

# ---- torch DL ----
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

TARGET_COL = "pIC50_value"
META_COLS = ["ChEMBL_ID", "SMILES"]

# ========== Prepare data ==========
X_train_tab = df_train.drop(columns=META_COLS + [TARGET_COL]).copy()
y_train = df_train[TARGET_COL].astype(float).copy()

X_test_tab  = df_test.drop(columns=META_COLS + [TARGET_COL]).copy()
y_test  = df_test[TARGET_COL].astype(float).copy()

print("Train X:", X_train_tab.shape, " Train y:", y_train.shape)
print("Test  X:", X_test_tab.shape,  " Test  y:", y_test.shape)

# ========== Balanced threshold selection on TRAIN (1 decimal) ==========
def pick_balanced_threshold(y: pd.Series, decimals=1):
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    vals = np.unique(np.round(y.values, decimals))
    vals = np.sort(vals)
    best = None  # (diff, thr, pos, neg)
    for t in vals:
        pos = int((y >= t).sum())
        neg = int((y <  t).sum())
        diff = abs(pos - neg)
        if best is None or diff < best[0]:
            best = (diff, float(t), pos, neg)
    return best

diff, thr_pic50, pos_n, neg_n = pick_balanced_threshold(y_train, decimals=1)
print(f"Balanced threshold chosen on TRAIN: {thr_pic50:.1f} | pos={pos_n}, neg={neg_n}, diff={diff}")

# =========================================================
# Common wrappers
# =========================================================
class SklearnWrapper:
    def __init__(self, est):
        self.est = est
    def fit(self, X, y):
        self.est.fit(X, y)
        return self
    def predict(self, X, smiles=None):
        return self.est.predict(X)

# -------------------------
# Torch datasets (FIXED)
# -------------------------
class TabDatasetTorch(Dataset):
    def __init__(self, X_np, y_np, mode: str):
        self.X = np.asarray(X_np, dtype=np.float32)
        self.y = np.asarray(y_np, dtype=np.float32).reshape(-1, 1)
        self.mode = mode  # "mlp"|"cnn1d"|"unet1d"|"rnn"|"lstm"
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.mode in ["cnn1d", "unet1d"]:
            # (C=1, L=D)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        elif self.mode in ["rnn", "lstm"]:
            # (L=D, input=1)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        else:
            # (D,)
            x = torch.tensor(x, dtype=torch.float32)
        return x, torch.tensor(y, dtype=torch.float32)

def train_torch(model, mode, X_train_np, y_train_np, device=None,
                lr=1e-3, weight_decay=1e-4, batch_size=64, max_epochs=200, patience=25):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # holdout split inside TRAIN
    n = len(X_train_np)
    idx = np.arange(n)
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(idx)
    split = int(n * 0.85)
    tr_idx, va_idx = idx[:split], idx[split:]

    ds_tr = TabDatasetTorch(X_train_np[tr_idx], y_train_np[tr_idx], mode)
    ds_va = TabDatasetTorch(X_train_np[va_idx], y_train_np[va_idx], mode)
    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    best_state, best_val = None, float("inf")
    bad = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation RMSE
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy().ravel()
                preds.append(pred)
                ys.append(yb.numpy().ravel())
        preds = np.concatenate(preds)
        ys = np.concatenate(ys)
        rmse = float(np.sqrt(np.mean((preds - ys) ** 2)))

        if rmse < best_val - 1e-6:
            best_val = rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, device

class TorchWrapper:
    def __init__(self, model: nn.Module, mode: str):
        self.model = model
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_val_rmse = np.nan

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        self.model, self.best_val_rmse, self.device = train_torch(
            self.model, self.mode, X_np, y_np, device=self.device
        )
        return self

    def predict(self, X, smiles=None):
        X_np = np.asarray(X, dtype=np.float32)
        self.model.eval()
        preds = []
        # batch prediction
        ds = TabDatasetTorch(X_np, np.zeros(len(X_np), dtype=np.float32), self.mode)
        loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=False)
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                pred = self.model(xb).cpu().numpy().ravel()
                preds.append(pred)
        return np.concatenate(preds)

# -------------------------
# Torch DL models
# -------------------------
d = X_train_tab.shape[1]

class TorchMLPNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        # x: (B, D)
        return self.net(x)

class TorchCNN1DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        # x: (B, 1, L)
        h = self.conv(x).squeeze(-1)
        return self.fc(h)

class TorchRNNNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: (B, L, 1)
        out, _ = self.rnn(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)

class TorchLSTMNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: (B, L, 1)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.fc(h_last)

class TorchUNet1DNet(nn.Module):
    """Lightweight 1D UNet for sequence length=L (tabular treated as sequence)."""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool1d(2)
        self.bot  = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(128, 128, 3, padding=1), nn.ReLU())

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(nn.Conv1d(128+64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Sequential(nn.Conv1d(64+32, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())

        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        # x: (B,1,L)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b  = self.bot(p2)

        u2 = self.up2(b)
        if u2.shape[-1] != e2.shape[-1]:
            u2 = nn.functional.pad(u2, (0, e2.shape[-1] - u2.shape[-1]))
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-1] != e1.shape[-1]:
            u1 = nn.functional.pad(u1, (0, e1.shape[-1] - u1.shape[-1]))
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.head(d1)

# -------------------------
# GNN (RDKit graph). Requires RDKit already available from earlier steps.
# -------------------------
try:
    from rdkit import Chem
except Exception as e:
    raise RuntimeError("RDKit not available in this runtime. Please ensure RDKit is installed/working before running GNN.") from e

def atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(includeNeighbors=True),
        int(atom.IsInRing())
    ], dtype=np.float32)

def build_graph_from_smiles(smiles, max_atoms):
    mol = Chem.MolFromSmiles(smiles)
    X = np.zeros((max_atoms, 6), dtype=np.float32)
    A = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    mask = np.zeros((max_atoms,), dtype=np.float32)

    if mol is None:
        return X, A, mask

    n = mol.GetNumAtoms()
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_atoms: break
        X[i] = atom_features(atom)
        mask[i] = 1.0

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < max_atoms and j < max_atoms:
            A[i, j] = 1.0
            A[j, i] = 1.0
    for i in range(min(n, max_atoms)):
        A[i, i] = 1.0

    deg = A.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D = np.diag(deg_inv_sqrt)
    A_norm = (D @ A @ D).astype(np.float32)
    return X, A_norm, mask

class GraphDataset(Dataset):
    def __init__(self, smiles_list, y_list, max_atoms):
        self.smiles = list(smiles_list)
        self.y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)
        self.max_atoms = max_atoms
        self.graphs = [build_graph_from_smiles(s, max_atoms) for s in self.smiles]
    def __len__(self): return len(self.smiles)
    def __getitem__(self, idx):
        X, A, mask = self.graphs[idx]
        y = self.y[idx]
        return (torch.tensor(X), torch.tensor(A), torch.tensor(mask), torch.tensor(y))

class SimpleGNN(nn.Module):
    def __init__(self, in_dim=6, hidden=64, steps=3):
        super().__init__()
        self.steps = steps
        self.lin0 = nn.Linear(in_dim, hidden)
        self.self_lin = nn.Linear(hidden, hidden)
        self.neigh_lin = nn.Linear(hidden, hidden)
        self.out = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, X, A, mask):
        # X: (B,N,F), A:(B,N,N), mask:(B,N)
        h = torch.relu(self.lin0(X))
        for _ in range(self.steps):
            neigh = torch.matmul(A, h)
            h = torch.relu(self.self_lin(h) + self.neigh_lin(neigh))
        mask_ = mask.unsqueeze(-1)
        h = (h * mask_).sum(dim=1) / mask_.sum(dim=1).clamp(min=1.0)
        return self.out(h)

class GNNWrapper:
    def __init__(self, max_atoms=60):
        self.max_atoms = max_atoms
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SimpleGNN().to(self.device)
        self.best_val_rmse = np.nan

    def fit(self, smiles, y):
        smiles = list(smiles)
        y = np.asarray(y, dtype=np.float32)

        n = len(smiles)
        idx = np.arange(n)
        rng = np.random.RandomState(RANDOM_STATE)
        rng.shuffle(idx)
        split = int(n * 0.85)
        tr, va = idx[:split], idx[split:]

        ds_tr = GraphDataset([smiles[i] for i in tr], y[tr], self.max_atoms)
        ds_va = GraphDataset([smiles[i] for i in va], y[va], self.max_atoms)
        tr_loader = DataLoader(ds_tr, batch_size=32, shuffle=True)
        va_loader = DataLoader(ds_va, batch_size=32, shuffle=False)

        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best_state, best_val = None, float("inf")
        bad, patience = 0, 25

        for epoch in range(200):
            self.model.train()
            for X,A,mask,yb in tr_loader:
                X,A,mask,yb = X.to(self.device), A.to(self.device), mask.to(self.device), yb.to(self.device)
                pred = self.model(X,A,mask)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            self.model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for X,A,mask,yb in va_loader:
                    X,A,mask = X.to(self.device), A.to(self.device), mask.to(self.device)
                    pred = self.model(X,A,mask).cpu().numpy().ravel()
                    preds.append(pred)
                    ys.append(yb.numpy().ravel())
            preds = np.concatenate(preds); ys = np.concatenate(ys)
            rmse = float(np.sqrt(np.mean((preds-ys)**2)))

            if rmse < best_val - 1e-6:
                best_val = rmse
                best_state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val_rmse = best_val
        return self

    def predict(self, X_unused=None, smiles=None):
        smiles = list(smiles)
        ds = GraphDataset(smiles, np.zeros(len(smiles), dtype=np.float32), self.max_atoms)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for X,A,mask,yb in loader:
                X,A,mask = X.to(self.device), A.to(self.device), mask.to(self.device)
                pred = self.model(X,A,mask).cpu().numpy().ravel()
                preds.append(pred)
        return np.concatenate(preds)

# ========== Define 20 unique models ==========
models = {
    # --- 14 classical ML (tabular) ---
    "SVR_RBF": SklearnWrapper(SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)),
    "LinearSVR": SklearnWrapper(LinearSVR(C=1.0, epsilon=0.1, random_state=RANDOM_STATE)),
    "RF": SklearnWrapper(RandomForestRegressor(n_estimators=800, random_state=RANDOM_STATE, max_features="sqrt")),
    "ExtraTrees": SklearnWrapper(ExtraTreesRegressor(n_estimators=1200, random_state=RANDOM_STATE, max_features="sqrt")),
    "GBDT": SklearnWrapper(GradientBoostingRegressor(random_state=RANDOM_STATE)),
    "HistGB": SklearnWrapper(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    "AdaBoost": SklearnWrapper(AdaBoostRegressor(random_state=RANDOM_STATE, n_estimators=500)),
    "KNN": SklearnWrapper(KNeighborsRegressor(n_neighbors=11, weights="distance")),
    "Ridge": SklearnWrapper(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    "Lasso": SklearnWrapper(Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=20000)),
    "ElasticNet": SklearnWrapper(ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=20000)),
    "BayesianRidge": SklearnWrapper(BayesianRidge()),
    "XGBoost": SklearnWrapper(xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        random_state=RANDOM_STATE, objective="reg:squarederror",
        tree_method="hist"
    )),
    "LightGBM": SklearnWrapper(lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        random_state=RANDOM_STATE
    )),

    # --- 5 DL on tabular as vector/sequence (FIXED shapes) ---
    "TorchMLP": TorchWrapper(TorchMLPNet(d), mode="mlp"),
    "TorchCNN1D": TorchWrapper(TorchCNN1DNet(), mode="cnn1d"),
    "TorchRNN": TorchWrapper(TorchRNNNet(hidden=64), mode="rnn"),
    "TorchLSTM": TorchWrapper(TorchLSTMNet(hidden=64), mode="lstm"),
    "TorchUNet1D": TorchWrapper(TorchUNet1DNet(), mode="unet1d"),

    # --- 1 GNN from SMILES ---
    "TorchGNN": GNNWrapper(max_atoms=60),
}

assert len(models) == 20 and len(set(models.keys())) == 20, "Must be exactly 20 unique, non-duplicate models."

# ========== Train + CV (sklearn only) ==========
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fitted_models = {}
rows = []

print("\n===== Part F: Training 20 models (FIXED) =====")
for name, wrapper in models.items():
    print(f"\n--- {name} ---")

    # CV only for sklearn models
    cv_mean, cv_std = np.nan, np.nan
    if isinstance(wrapper, SklearnWrapper):
        try:
            scores = cross_val_score(wrapper.est, X_train_tab, y_train, cv=cv, scoring="r2", n_jobs=-1)
            cv_mean, cv_std = float(scores.mean()), float(scores.std())
        except:
            pass

    # Fit + train preds
    if name == "TorchGNN":
        wrapper.fit(df_train["SMILES"], y_train.values)
        y_tr_pred = wrapper.predict(smiles=df_train["SMILES"])
        note = f"val_rmse~{wrapper.best_val_rmse:.3f}"
    elif isinstance(wrapper, TorchWrapper):
        wrapper.fit(X_train_tab.values, y_train.values)
        y_tr_pred = wrapper.predict(X_train_tab.values)
        note = f"val_rmse~{wrapper.best_val_rmse:.3f}"
    else:
        wrapper.fit(X_train_tab, y_train)
        y_tr_pred = wrapper.predict(X_train_tab)
        note = ""

    fitted_models[name] = wrapper

    tr_r2 = float(r2_score(y_train, y_tr_pred))
    tr_rmse = float(np.sqrt(mean_squared_error(y_train, y_tr_pred)))
    tr_mae = float(mean_absolute_error(y_train, y_tr_pred))

    rows.append({
        "model": name,
        "cv_r2_mean(5fold,sklearn_only)": cv_mean,
        "cv_r2_std(5fold,sklearn_only)": cv_std,
        "train_r2": tr_r2,
        "train_rmse": tr_rmse,
        "train_mae": tr_mae,
        "note": note
    })

df_F = pd.DataFrame(rows)
df_F = df_F.sort_values(["cv_r2_mean(5fold,sklearn_only)", "train_r2"], ascending=False).reset_index(drop=True)

print("\nF summary (sorted by CV where available):")
print(df_F[["model","cv_r2_mean(5fold,sklearn_only)","train_r2","train_rmse","note"]])

df_F.to_csv("F_compare_20models.csv", index=False)
print("Saved: F_compare_20models.csv")

from google.colab import files
files.download("F_compare_20models.csv")






# =========================
# Part G (FIXED): External validation for all 20 models
# - Regression: test R2/RMSE/MAE
# - Classification AUC: use predicted pIC50 as score, label by thr_pic50 (balanced from TRAIN)
# Output:
#   - G_test_20models.csv
#   - predicted_vs_experimental_best.png
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.base import clone

y_test_cls = (y_test >= thr_pic50).astype(int).values

rows = []
test_preds = {}

print("\n===== Part G: External validation for 20 models (FIXED) =====")
for name, wrapper in fitted_models.items():
    if name == "TorchGNN":
        y_pred = wrapper.predict(smiles=df_test["SMILES"])
    elif isinstance(wrapper, TorchWrapper):
        y_pred = wrapper.predict(X_test_tab.values)
    else:
        y_pred = wrapper.predict(X_test_tab)

    test_preds[name] = y_pred

    te_r2 = float(r2_score(y_test, y_pred))
    te_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    te_mae = float(mean_absolute_error(y_test, y_pred))

    auc = float(roc_auc_score(y_test_cls, y_pred))  # score=predicted pIC50
    pred_cls = (y_pred >= thr_pic50).astype(int)
    acc = float(accuracy_score(y_test_cls, pred_cls))
    f1  = float(f1_score(y_test_cls, pred_cls))

    rows.append({
        "model": name,
        "test_r2": te_r2,
        "test_rmse": te_rmse,
        "test_mae": te_mae,
        f"test_auc(thr={thr_pic50:.1f})": auc,
        f"test_acc(thr={thr_pic50:.1f})": acc,
        f"test_f1(thr={thr_pic50:.1f})": f1
    })

df_G = pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
print(df_G)

df_G.to_csv("G_test_20models.csv", index=False)
print("Saved: G_test_20models.csv")
from google.colab import files
files.download("G_test_20models.csv")

# ========== Best model by TEST R2 ==========
best_name = df_G.loc[0, "model"]
best_model = fitted_models[best_name]
print("\nBest model by TEST R²:", best_name)

# ========== Y-randomization on BEST model (evaluate on TEST) ==========
N_PERM = 50
rand_r2 = []

print(f"\nRunning Y-randomization on BEST model ({best_name})...")

for i in range(N_PERM):
    y_shuf = y_train.sample(frac=1, random_state=i).reset_index(drop=True).values

    if isinstance(best_model, SklearnWrapper):
        est = clone(best_model.est)
        est.fit(X_train_tab, y_shuf)
        y_perm = est.predict(X_test_tab)

    elif isinstance(best_model, TorchWrapper):
        # retrain a fresh copy with same architecture
        if best_name == "TorchMLP":
            fresh = TorchWrapper(TorchMLPNet(X_train_tab.shape[1]), mode="mlp")
        elif best_name == "TorchCNN1D":
            fresh = TorchWrapper(TorchCNN1DNet(), mode="cnn1d")
        elif best_name == "TorchRNN":
            fresh = TorchWrapper(TorchRNNNet(hidden=64), mode="rnn")
        elif best_name == "TorchLSTM":
            fresh = TorchWrapper(TorchLSTMNet(hidden=64), mode="lstm")
        elif best_name == "TorchUNet1D":
            fresh = TorchWrapper(TorchUNet1DNet(), mode="unet1d")
        else:
            fresh = TorchWrapper(TorchMLPNet(X_train_tab.shape[1]), mode="mlp")

        fresh.fit(X_train_tab.values, y_shuf)
        y_perm = fresh.predict(X_test_tab.values)

    else:
        # GNN
        fresh = GNNWrapper(max_atoms=60)
        fresh.fit(df_train["SMILES"], y_shuf)
        y_perm = fresh.predict(smiles=df_test["SMILES"])

    rand_r2.append(r2_score(y_test, y_perm))

rand_r2 = np.array(rand_r2, dtype=float)
orig_r2 = float(df_G.loc[df_G["model"] == best_name, "test_r2"].values[0])

print(f"Randomized TEST R² (mean): {rand_r2.mean():.4f} ± {rand_r2.std():.4f}")
print(f"Original  TEST R²:         {orig_r2:.4f}")
print("✓ Passed Y-randomization (on TEST)" if orig_r2 > rand_r2.max() else "⚠ Warning: close to randomized")

# ========== Plot predicted vs experimental for BEST ==========
if best_name == "TorchGNN":
    y_tr_pred_best = best_model.predict(smiles=df_train["SMILES"])
    y_te_pred_best = best_model.predict(smiles=df_test["SMILES"])
elif isinstance(best_model, TorchWrapper):
    y_tr_pred_best = best_model.predict(X_train_tab.values)
    y_te_pred_best = best_model.predict(X_test_tab.values)
else:
    y_tr_pred_best = best_model.predict(X_train_tab)
    y_te_pred_best = best_model.predict(X_test_tab)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_train, y_tr_pred_best, alpha=0.6, edgecolors="k", linewidth=0.5)
mn, mx = float(y_train.min()), float(y_train.max())
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal")
axes[0].set_xlabel("Experimental pIC50")
axes[0].set_ylabel("Predicted pIC50")
axes[0].set_title(f"Train ({best_name}) R²={r2_score(y_train, y_tr_pred_best):.4f}")
axes[0].legend()
axes[0].set_aspect("equal", adjustable="box")

axes[1].scatter(y_test, y_te_pred_best, alpha=0.6, edgecolors="k", linewidth=0.5)
mn, mx = float(y_test.min()), float(y_test.max())
axes[1].plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal")
axes[1].set_xlabel("Experimental pIC50")
axes[1].set_ylabel("Predicted pIC50")
axes[1].set_title(f"Test ({best_name}) R²={r2_score(y_test, y_te_pred_best):.4f}")
axes[1].legend()
axes[1].set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("predicted_vs_experimental_best.png", dpi=220, bbox_inches="tight")
plt.show()

print("Saved: predicted_vs_experimental_best.png")
files.download("predicted_vs_experimental_best.png")






# =========================
# Part H (FIXED): Interpret BEST model (importance + SHAP if applicable) + ROC curve
# Output:
#   - H_best_importance.png
#   - H_shap_top15_bar.png / H_shap_top15_beeswarm.png (if applicable)
#   - H_ROC_best_model.png
# =========================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

print("Best model:", best_name)

# ========== H1: Feature importance (tabular models only) ==========
if best_name == "TorchGNN":
    print("Best model is GNN (graph-based). Tabular feature importance is not defined.")
else:
    feature_names = list(X_train_tab.columns)

    if isinstance(best_model, SklearnWrapper) and hasattr(best_model.est, "feature_importances_"):
        imp = np.asarray(best_model.est.feature_importances_, dtype=float)
        method = "feature_importances_"
    elif isinstance(best_model, SklearnWrapper) and hasattr(best_model.est, "coef_"):
        coef = np.asarray(best_model.est.coef_).ravel()
        imp = np.abs(coef)
        method = "|coef|"
    else:
        # manual permutation importance on TEST (ΔR2)
        from sklearn.metrics import r2_score
        rng = np.random.RandomState(42)

        def pred_fn(X_np):
            if isinstance(best_model, TorchWrapper):
                return best_model.predict(X_np)
            else:
                return best_model.predict(pd.DataFrame(X_np, columns=feature_names))

        base_pred = pred_fn(X_test_tab.values)
        base_r2 = r2_score(y_test.values, base_pred)

        imp_list = []
        Xv = X_test_tab.values.copy()
        for j in range(Xv.shape[1]):
            drops = []
            for _ in range(5):
                Xp = Xv.copy()
                rng.shuffle(Xp[:, j])
                pred = pred_fn(Xp)
                drops.append(base_r2 - r2_score(y_test.values, pred))
            imp_list.append(np.mean(drops))
        imp = np.asarray(imp_list, dtype=float)
        method = "manual_permutation(ΔR2)"

    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
    topk = 15
    top_idx = np.argsort(imp)[::-1][:topk]

    plt.figure(figsize=(10, 7))
    plt.barh([feature_names[i] for i in top_idx][::-1], [imp[i] for i in top_idx][::-1])
    plt.title(f"Best model feature importance ({best_name}) via {method}")
    plt.tight_layout()
    plt.savefig("H_best_importance.png", dpi=220, bbox_inches="tight")
    plt.show()
    print("Saved: H_best_importance.png")

# ========== H2: SHAP (skip for GNN) ==========
if best_name == "TorchGNN":
    print("⚠ Skip SHAP for GNN (graph input).")
else:
    try:
        import shap
        print("shap available.")
    except:
        print("installing shap...")
        !pip -q install shap
        import shap
        print("shap installed.")

    N_EXPLAIN = min(20, len(X_test_tab))
    N_BG = min(100, len(X_train_tab))
    X_shap = X_test_tab.sample(n=N_EXPLAIN, random_state=42)
    background = shap.sample(X_train_tab, N_BG, random_state=42)

    # predict callable
    if isinstance(best_model, SklearnWrapper):
        pred_fn = best_model.est.predict
        model_name = type(best_model.est).__name__.lower()
    else:
        pred_fn = lambda X: best_model.predict(np.asarray(X, dtype=np.float32))
        model_name = type(best_model.model).__name__.lower()

    is_tree_like = any(k in model_name for k in ["forest", "extra", "xgb", "lgbm", "gradient", "hist", "boost"])

    print(f"X_shap: {X_shap.shape} | background: {background.shape} | tree_like={is_tree_like}")

    if is_tree_like and isinstance(best_model, SklearnWrapper):
        explainer = shap.TreeExplainer(best_model.est)
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    else:
        explainer = shap.PermutationExplainer(pred_fn, background.values)
        min_need = 2 * X_shap.shape[1] + 1
        MAX_EVALS = max(2000, min_need + 50)  # avoid "max_evals too low"
        shap_exp = explainer(X_shap.values, max_evals=MAX_EVALS)
        shap_values = shap_exp.values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=15, show=False)
    plt.title(f"SHAP Top 15 ({best_name})")
    plt.tight_layout()
    plt.savefig("H_shap_top15_bar.png", dpi=220, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, max_display=15, show=False)
    plt.title(f"SHAP Beeswarm Top 15 ({best_name})")
    plt.tight_layout()
    plt.savefig("H_shap_top15_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.show()

    print("Saved: H_shap_top15_bar.png, H_shap_top15_beeswarm.png")

# ========== H3: ROC curve for BEST model ==========
y_test_cls = (y_test >= thr_pic50).astype(int).values

if best_name == "TorchGNN":
    score = best_model.predict(smiles=df_test["SMILES"])
elif isinstance(best_model, TorchWrapper):
    score = best_model.predict(X_test_tab.values)
else:
    score = best_model.predict(X_test_tab)

auc = roc_auc_score(y_test_cls, score)
fpr, tpr, _ = roc_curve(y_test_cls, score)

plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr, tpr, label=f"{best_name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "r--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC (active if pIC50 ≥ {thr_pic50:.1f}, balanced on TRAIN)")
plt.legend()
plt.tight_layout()
plt.savefig("H_ROC_best_model.png", dpi=220, bbox_inches="tight")
plt.show()

print("Saved: H_ROC_best_model.png")
from google.colab import files
files.download("H_ROC_best_model.png")






