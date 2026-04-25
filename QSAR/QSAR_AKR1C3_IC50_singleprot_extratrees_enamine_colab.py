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
except Exception:
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

if df_raw.shape[1] < 3:
    raise ValueError("CSV must have at least 3 columns: [ChEMBL_ID, Smiles, pIC50_value].")

id_col = df_raw.columns[0]
smi_col = df_raw.columns[1]
pic50_col = df_raw.columns[2]

df = df_raw[[id_col, smi_col, pic50_col]].copy()
df = df.rename(columns={id_col: "ChEMBL_ID", smi_col: "Smiles", pic50_col: "pIC50_value"})

df["pIC50_value"] = pd.to_numeric(df["pIC50_value"], errors="coerce")
df = df.dropna(subset=["ChEMBL_ID", "Smiles", "pIC50_value"]).copy()

print("After pIC50 cleaning:", len(df), "rows")

def is_parseable(s):
    try:
        if pd.isna(s):
            return False
        return Chem.MolFromSmiles(str(s)) is not None
    except Exception:
        return False

df = df[df["Smiles"].apply(is_parseable)].copy()
print("After SMILES parse filter:", len(df), "rows")

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

def mode_first(x: pd.Series):
    vc = x.value_counts(dropna=True)
    if len(vc) == 0:
        return np.nan
    return vc.index[0]

before = len(df)
dup_id_count = (df["ChEMBL_ID"].value_counts() > 1).sum()
print("ChEMBL_ID with >=2 rows:", int(dup_id_count))

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

df_curated = df_grouped[["ChEMBL_ID", "SMILES", "pIC50_value"]].copy()

out_name = "AKR1C3_pIC50_curated_uniqueID.csv"
df_curated.to_csv(out_name, index=False)
print("Saved:", out_name)
files.download(out_name)


# =========================
# Part C: SMILES -> descriptors + ECFP4, build df_features
# =========================

from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

df_curated = df_curated.copy()

required_cols = ["ChEMBL_ID", "SMILES", "pIC50_value"]
missing = [c for c in required_cols if c not in df_curated.columns]
if missing:
    raise ValueError(f"Missing columns in df_curated: {missing}")

df_curated["Mol"] = df_curated["SMILES"].apply(Chem.MolFromSmiles)
df_curated = df_curated[df_curated["Mol"].notna()].reset_index(drop=True)

desc_names = [d[0] for d in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

def get_descriptors(mol):
    try:
        return list(calc.CalcDescriptors(mol))
    except Exception:
        return [np.nan] * len(desc_names)

def get_ecfp4(mol, radius=2, nBits=2048):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return list(fp)
    except Exception:
        return [0] * nBits

desc_values = df_curated["Mol"].apply(get_descriptors).tolist()
df_desc = pd.DataFrame(desc_values, columns=desc_names)
df_desc = df_desc.replace([np.inf, -np.inf], np.nan)
df_desc = df_desc.dropna(axis=1, how="all")
for col in df_desc.columns:
    if pd.api.types.is_numeric_dtype(df_desc[col]):
        df_desc[col] = df_desc[col].fillna(df_desc[col].median())

ecfp_values = df_curated["Mol"].apply(get_ecfp4).tolist()
df_ecfp = pd.DataFrame(ecfp_values, columns=[f"ECFP4_{i}" for i in range(2048)])

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
# Part E: Train-test split (KEEP ORIGINAL SETTING)
# - 80/20 split
# - random_state=42
# - stratify by qcut bins of pIC50_value
# =========================

from sklearn.model_selection import train_test_split

TARGET_COL = "pIC50_value"
meta_cols = ["ChEMBL_ID", "SMILES", TARGET_COL]

X_all = df_features.drop(columns=meta_cols).copy()
y_all = df_features[TARGET_COL].copy()

n_bins = min(5, y_all.nunique())
y_binned = pd.qcut(y_all, q=n_bins, labels=False, duplicates="drop")

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_binned,
)

df_train_raw = pd.concat(
    [
        df_features.loc[X_train_raw.index, ["ChEMBL_ID", "SMILES"]].reset_index(drop=True),
        X_train_raw.reset_index(drop=True),
        y_train_raw.reset_index(drop=True),
    ],
    axis=1,
)

df_test_raw = pd.concat(
    [
        df_features.loc[X_test_raw.index, ["ChEMBL_ID", "SMILES"]].reset_index(drop=True),
        X_test_raw.reset_index(drop=True),
        y_test_raw.reset_index(drop=True),
    ],
    axis=1,
)

print("\n[Split summary]")
print("Split mode : train/test")
print("test_size  : 0.2")
print("random_seed: 42")
print("stratify   : pd.qcut(y_all, q=min(5, y_all.nunique()), duplicates='drop')")
print(f"Training: {len(df_train_raw)} compounds")
print(f"Testing : {len(df_test_raw)} compounds")
print("Train y mean:", float(y_train_raw.mean()))
print("Test  y mean:", float(y_test_raw.mean()))


# =========================
# Part D: Removing Noise + scale (fit on TRAIN only; KEEP ORIGINAL SETTING)
# =========================

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

LEAK_COLS = ["IC50_value"]
meta_cols = ["ChEMBL_ID", "SMILES", TARGET_COL]

Xtr = df_train_raw.drop(columns=meta_cols, errors="ignore").copy()
ytr = df_train_raw[TARGET_COL].copy()
Xte = df_test_raw.drop(columns=meta_cols, errors="ignore").copy()
yte = df_test_raw[TARGET_COL].copy()

Xtr = Xtr.drop(columns=[c for c in LEAK_COLS if c in Xtr.columns], errors="ignore")
Xte = Xte.drop(columns=[c for c in LEAK_COLS if c in Xte.columns], errors="ignore")

train_raw_feature_cols = Xtr.columns.tolist()

vt = VarianceThreshold(threshold=0.0)
Xtr_v = vt.fit_transform(Xtr)
Xte_v = vt.transform(Xte)

kept_cols = Xtr.columns[vt.get_support()].tolist()
Xtr_v = pd.DataFrame(Xtr_v, columns=kept_cols)
Xte_v = pd.DataFrame(Xte_v, columns=kept_cols)
print("After constant filter:", Xtr_v.shape[1], "features")

corr = Xtr_v.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if (upper[col] > 0.95).any()]

Xtr_f = Xtr_v.drop(columns=to_drop)
Xte_f = Xte_v.drop(columns=to_drop)
print(f"After correlation filter: {Xtr_f.shape[1]} features (dropped {len(to_drop)})")

scaler = StandardScaler()
Xtr_s = pd.DataFrame(scaler.fit_transform(Xtr_f), columns=Xtr_f.columns)
Xte_s = pd.DataFrame(scaler.transform(Xte_f), columns=Xte_f.columns)

df_train = pd.concat(
    [
        df_train_raw[["ChEMBL_ID", "SMILES"]].reset_index(drop=True),
        Xtr_s.reset_index(drop=True),
        ytr.reset_index(drop=True),
    ],
    axis=1,
)

df_test = pd.concat(
    [
        df_test_raw[["ChEMBL_ID", "SMILES"]].reset_index(drop=True),
        Xte_s.reset_index(drop=True),
        yte.reset_index(drop=True),
    ],
    axis=1,
)

print("Processed train shape:", df_train.shape)
print("Processed test  shape:", df_test.shape)

df_train.to_csv("AKR1C3_train_processed.csv", index=False)
df_test.to_csv("AKR1C3_test_processed.csv", index=False)
print("Saved: AKR1C3_train_processed.csv, AKR1C3_test_processed.csv")
files.download("AKR1C3_train_processed.csv")
files.download("AKR1C3_test_processed.csv")


# =========================
# Part D2: Load Enamine SDF and preprocess with TRAIN-fitted pipeline
# - This does NOT change the original train/test split
# - It is an external validation / screening set only
# =========================

import os
import gzip
from rdkit.Chem import PandasTools

print("\nUpload external validation SDF, e.g. Enamine_advanced_collection_202512.sdf")
uploaded_sdf = files.upload()
sdf_filename = list(uploaded_sdf.keys())[0]
print("Validation SDF:", sdf_filename)


def load_sdf_records(sdf_path):
    records = []

    if sdf_path.lower().endswith(".gz"):
        supplier = Chem.ForwardSDMolSupplier(gzip.open(sdf_path), removeHs=False)
    else:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)

    fallback_counter = 1
    id_priority = [
        "Molecule ChEMBL ID", "ChEMBL_ID", "Molecule_ChEMBL_ID",
        "ID", "CatalogID", "Catalog ID", "Compound_ID", "Name"
    ]

    for mol in supplier:
        if mol is None:
            continue

        smiles = Chem.MolToSmiles(mol, canonical=True)
        mol_id = None
        for key in id_priority:
            if mol.HasProp(key):
                val = mol.GetProp(key).strip()
                if val:
                    mol_id = val
                    break

        if mol_id is None and mol.HasProp("_Name"):
            val = mol.GetProp("_Name").strip()
            if val:
                mol_id = val

        if mol_id is None:
            mol_id = f"ENAMINE_{fallback_counter:07d}"
        fallback_counter += 1

        records.append({
            "Molecule ChEMBL ID": mol_id,
            "Smiles": smiles,
            "Mol": mol,
        })

    return pd.DataFrame(records)


df_val_sdf = load_sdf_records(sdf_filename)
if df_val_sdf.empty:
    raise ValueError("No valid molecules were read from the uploaded SDF file.")

print("Loaded validation molecules:", len(df_val_sdf))
print(df_val_sdf.head(3))

val_desc_values = df_val_sdf["Mol"].apply(get_descriptors).tolist()
df_val_desc = pd.DataFrame(val_desc_values, columns=desc_names)
df_val_desc = df_val_desc.replace([np.inf, -np.inf], np.nan)
df_val_desc = df_val_desc.dropna(axis=1, how="all")

for col in df_val_desc.columns:
    if pd.api.types.is_numeric_dtype(df_val_desc[col]):
        fill_val = df_val_desc[col].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        df_val_desc[col] = df_val_desc[col].fillna(fill_val)

val_ecfp_values = df_val_sdf["Mol"].apply(get_ecfp4).tolist()
df_val_ecfp = pd.DataFrame(val_ecfp_values, columns=[f"ECFP4_{i}" for i in range(2048)])

df_val_features_raw = pd.concat(
    [
        df_val_sdf[["Molecule ChEMBL ID", "Smiles"]].reset_index(drop=True),
        df_val_desc.reset_index(drop=True),
        df_val_ecfp.reset_index(drop=True),
    ],
    axis=1,
)

# Align to original TRAIN raw feature columns before VT / correlation / scaling
X_val_raw = df_val_features_raw.drop(columns=["Molecule ChEMBL ID", "Smiles"], errors="ignore").copy()
X_val_raw = X_val_raw.reindex(columns=train_raw_feature_cols, fill_value=0.0)
X_val_raw = X_val_raw.drop(columns=[c for c in LEAK_COLS if c in X_val_raw.columns], errors="ignore")

X_val_v = vt.transform(X_val_raw)
X_val_v = pd.DataFrame(X_val_v, columns=kept_cols)
X_val_f = X_val_v.drop(columns=to_drop, errors="ignore")
X_val_s = pd.DataFrame(scaler.transform(X_val_f), columns=X_val_f.columns)

df_val_processed = pd.concat(
    [
        df_val_features_raw[["Molecule ChEMBL ID", "Smiles"]].reset_index(drop=True),
        X_val_s.reset_index(drop=True),
    ],
    axis=1,
)

print("Processed validation shape:", df_val_processed.shape)


# =========================
# Part F: Train ONLY ExtraTrees (settings unchanged)
# - Keep original ExtraTrees settings exactly the same
# - Keep output filename F_compare_20models.csv for compatibility
# =========================

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor

RANDOM_STATE = 42
TARGET_COL = "pIC50_value"
META_COLS = ["ChEMBL_ID", "SMILES"]

X_train_tab = df_train.drop(columns=META_COLS + [TARGET_COL]).copy()
y_train = df_train[TARGET_COL].astype(float).copy()
X_test_tab = df_test.drop(columns=META_COLS + [TARGET_COL]).copy()
y_test = df_test[TARGET_COL].astype(float).copy()
X_val_tab = df_val_processed.drop(columns=["Molecule ChEMBL ID", "Smiles"], errors="ignore").copy()

print("Train X:", X_train_tab.shape, " Train y:", y_train.shape)
print("Test  X:", X_test_tab.shape,  " Test  y:", y_test.shape)
print("Valid X:", X_val_tab.shape)


def pick_balanced_threshold(y: pd.Series, decimals=1):
    y = pd.to_numeric(y, errors="coerce").dropna().astype(float)
    vals = np.unique(np.round(y.values, decimals))
    vals = np.sort(vals)
    best = None
    for t in vals:
        pos = int((y >= t).sum())
        neg = int((y < t).sum())
        diff = abs(pos - neg)
        if best is None or diff < best[0]:
            best = (diff, float(t), pos, neg)
    return best


diff, thr_pic50, pos_n, neg_n = pick_balanced_threshold(y_train, decimals=1)
print(f"Balanced threshold chosen on TRAIN: {thr_pic50:.1f} | pos={pos_n}, neg={neg_n}, diff={diff}")

extra_model = ExtraTreesRegressor(
    n_estimators=1200,
    random_state=RANDOM_STATE,
    max_features="sqrt",
)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(extra_model, X_train_tab, y_train, cv=cv, scoring="r2", n_jobs=-1)

extra_model.fit(X_train_tab, y_train)
y_tr_pred = extra_model.predict(X_train_tab)

fitted_models = {"ExtraTrees": extra_model}
best_name = "ExtraTrees"
best_model = extra_model

df_F = pd.DataFrame([
    {
        "model": "ExtraTrees",
        "cv_r2_mean(5fold,sklearn_only)": float(cv_scores.mean()),
        "cv_r2_std(5fold,sklearn_only)": float(cv_scores.std()),
        "train_r2": float(r2_score(y_train, y_tr_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_tr_pred))),
        "train_mae": float(mean_absolute_error(y_train, y_tr_pred)),
        "note": "only ExtraTrees kept; parameters unchanged",
    }
])

print("\nF summary:")
print(df_F)

df_F.to_csv("F_compare_20models.csv", index=False)
print("Saved: F_compare_20models.csv")
files.download("F_compare_20models.csv")


# =========================
# Part G: External validation on TEST + Enamine ranking CSV
# - Keep original G output filename for compatibility
# - New extra CSV for Enamine external set
# =========================

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.base import clone

y_test_cls = (y_test >= thr_pic50).astype(int).values

y_pred_test = best_model.predict(X_test_tab)

df_G = pd.DataFrame([
    {
        "model": "ExtraTrees",
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        f"test_auc(thr={thr_pic50:.1f})": float(roc_auc_score(y_test_cls, y_pred_test)),
        f"test_acc(thr={thr_pic50:.1f})": float(accuracy_score(y_test_cls, (y_pred_test >= thr_pic50).astype(int))),
        f"test_f1(thr={thr_pic50:.1f})": float(f1_score(y_test_cls, (y_pred_test >= thr_pic50).astype(int))),
    }
])

print("\nG summary:")
print(df_G)

df_G.to_csv("G_test_20models.csv", index=False)
print("Saved: G_test_20models.csv")
files.download("G_test_20models.csv")

print("\nBest model by TEST R²:", best_name)

# ---- Y-randomization on best model ----
N_PERM = 50
rand_r2 = []
print("\nRunning Y-randomization on ExtraTrees...")
for i in range(N_PERM):
    y_shuf = y_train.sample(frac=1, random_state=i).reset_index(drop=True).values
    est = clone(best_model)
    est.fit(X_train_tab, y_shuf)
    y_perm = est.predict(X_test_tab)
    rand_r2.append(r2_score(y_test, y_perm))

rand_r2 = np.array(rand_r2, dtype=float)
orig_r2 = float(df_G.loc[0, "test_r2"])
print(f"Randomized TEST R² (mean): {rand_r2.mean():.4f} ± {rand_r2.std():.4f}")
print(f"Original  TEST R²:         {orig_r2:.4f}")
print("✓ Passed Y-randomization (on TEST)" if orig_r2 > rand_r2.max() else "⚠ Warning: close to randomized")

# ---- Train/Test predicted vs experimental plot ----
y_tr_pred_best = best_model.predict(X_train_tab)
y_te_pred_best = y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_train, y_tr_pred_best, alpha=0.6, edgecolors="k", linewidth=0.5)
mn, mx = float(y_train.min()), float(y_train.max())
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal")
axes[0].set_xlabel("Experimental pIC50")
axes[0].set_ylabel("Predicted pIC50")
axes[0].set_title(f"Train (ExtraTrees) R²={r2_score(y_train, y_tr_pred_best):.4f}")
axes[0].legend()
axes[0].set_aspect("equal", adjustable="box")

axes[1].scatter(y_test, y_te_pred_best, alpha=0.6, edgecolors="k", linewidth=0.5)
mn, mx = float(y_test.min()), float(y_test.max())
axes[1].plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal")
axes[1].set_xlabel("Experimental pIC50")
axes[1].set_ylabel("Predicted pIC50")
axes[1].set_title(f"Test (ExtraTrees) R²={r2_score(y_test, y_te_pred_best):.4f}")
axes[1].legend()
axes[1].set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig("predicted_vs_experimental_best.png", dpi=220, bbox_inches="tight")
plt.show()
print("Saved: predicted_vs_experimental_best.png")
files.download("predicted_vs_experimental_best.png")

# ---- NEW: Enamine prediction + descending ranking CSV ----
y_val_pred = best_model.predict(X_val_tab)

df_enamine_ranked = df_val_processed[["Molecule ChEMBL ID", "Smiles"]].copy()
df_enamine_ranked["predicted_pIC50_value"] = y_val_pred
df_enamine_ranked = df_enamine_ranked.sort_values("predicted_pIC50_value", ascending=False).reset_index(drop=True)
df_enamine_ranked.insert(0, "rank", np.arange(1, len(df_enamine_ranked) + 1))

enamine_out = "Enamine_advanced_collection_202512_predicted_sorted.csv"
df_enamine_ranked.to_csv(enamine_out, index=False)
print("Saved:", enamine_out)
print(df_enamine_ranked.head(10))
files.download(enamine_out)


# =========================
# Part H: Interpret ExtraTrees + SHAP + ROC curve
# =========================

from sklearn.metrics import roc_curve

print("Best model:", best_name)
feature_names = list(X_train_tab.columns)
imp = np.asarray(best_model.feature_importances_, dtype=float)
imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
topk = 15
top_idx = np.argsort(imp)[::-1][:topk]

plt.figure(figsize=(10, 7))
plt.barh([feature_names[i] for i in top_idx][::-1], [imp[i] for i in top_idx][::-1])
plt.title("Best model feature importance (ExtraTrees) via feature_importances_")
plt.tight_layout()
plt.savefig("H_best_importance.png", dpi=220, bbox_inches="tight")
plt.show()
print("Saved: H_best_importance.png")
files.download("H_best_importance.png")

try:
    import shap
    print("shap available.")
except Exception:
    print("installing shap...")
    !pip -q install shap
    import shap
    print("shap installed.")

N_EXPLAIN = min(20, len(X_test_tab))
X_shap = X_test_tab.sample(n=N_EXPLAIN, random_state=42)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=15, show=False)
plt.title("SHAP Top 15 (ExtraTrees)")
plt.tight_layout()
plt.savefig("H_shap_top15_bar.png", dpi=220, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, max_display=15, show=False)
plt.title("SHAP Beeswarm Top 15 (ExtraTrees)")
plt.tight_layout()
plt.savefig("H_shap_top15_beeswarm.png", dpi=220, bbox_inches="tight")
plt.show()

print("Saved: H_shap_top15_bar.png, H_shap_top15_beeswarm.png")
files.download("H_shap_top15_bar.png")
files.download("H_shap_top15_beeswarm.png")

score = y_pred_test
auc = roc_auc_score(y_test_cls, score)
fpr, tpr, _ = roc_curve(y_test_cls, score)

plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr, tpr, label=f"ExtraTrees (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "r--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC (active if pIC50 ≥ {thr_pic50:.1f}, balanced on TRAIN)")
plt.legend()
plt.tight_layout()
plt.savefig("H_ROC_best_model.png", dpi=220, bbox_inches="tight")
plt.show()

print("Saved: H_ROC_best_model.png")
files.download("H_ROC_best_model.png")
