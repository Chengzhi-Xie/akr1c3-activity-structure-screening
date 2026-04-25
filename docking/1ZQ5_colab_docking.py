!apt-get -qq update
!apt-get -qq install -y openbabel > /dev/null

!pip -q install vina biopython pandas numpy py3Dmol tqdm requests





from google.colab import files
import os

uploaded = files.upload()

csv_candidates = [k for k in uploaded.keys() if k.lower().endswith(".csv")]
assert len(csv_candidates) >= 1, "请上传 Enamine_QSAR_Pharm.csv"
CSV_PATH = "/content/" + csv_candidates[0]
print("Using CSV:", CSV_PATH)






from pathlib import Path

# =========================
# Core settings
# =========================
PDB_ID = "1ZQ5"
TARGET_CHAIN = "A"
KEEP_COFAC_RESNAMES = {"NAP"}          # 保留 NADP
REFERENCE_LIGAND_RESNAME = "E04"       # 共晶抑制剂 EM1404
REMOVE_WATERS = True

# box settings
BOX_PADDING = 8.0
MIN_BOX_EDGE = 22.0

# docking settings
EXHAUSTIVENESS = 32
NUM_MODES = 20
ENERGY_RANGE = 4
CPU = 2
SEED = 2026

# outputs
WORKDIR = Path("/content/AKR1C3_Vina_1ZQ5_NAP_keep")
WORKDIR.mkdir(parents=True, exist_ok=True)

RAW_PDB = WORKDIR / f"{PDB_ID}.pdb"
RECEPTOR_CLEAN_PDB = WORKDIR / f"{PDB_ID}_chain{TARGET_CHAIN}_keepNAP_noE04_clean.pdb"
RECEPTOR_H_PDB = WORKDIR / f"{PDB_ID}_chain{TARGET_CHAIN}_keepNAP_noE04_H.pdb"
RECEPTOR_PDBQT = WORKDIR / f"{PDB_ID}_chain{TARGET_CHAIN}_keepNAP_noE04.pdbqt"

LIGAND_DIR = WORKDIR / "ligands"
POSE_DIR = WORKDIR / "docked_poses"
LIGAND_DIR.mkdir(exist_ok=True)
POSE_DIR.mkdir(exist_ok=True)

print("WORKDIR:", WORKDIR)







import io
import json
import requests
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select, Polypeptide

def download_pdb(pdb_id: str, outpath: Path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    outpath.write_text(r.text)
    return outpath

def get_structure(pdb_path: Path, structure_id="struct"):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, str(pdb_path))

def extract_reference_ligand_coords(structure, chain_id="A", lig_resname="E04"):
    coords = []
    found = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for res in chain:
                hetflag = res.id[0].strip()
                resname = res.get_resname().strip().upper()
                if hetflag and resname == lig_resname:
                    found.append((chain.id, res.id[1], resname))
                    for atom in res:
                        elem = (atom.element or "").strip().upper()
                        if elem != "H":
                            coords.append(atom.coord)
    if len(coords) == 0:
        raise ValueError(f"未找到 chain {chain_id} 上的参考配体 {lig_resname}")
    coords = np.array(coords, dtype=float)
    return coords, found

class ReceptorSelect(Select):
    def __init__(self, target_chain="A", keep_cofac=None, remove_waters=True):
        self.target_chain = target_chain
        self.keep_cofac = set(x.upper() for x in (keep_cofac or set()))
        self.remove_waters = remove_waters

    def accept_chain(self, chain):
        return chain.id == self.target_chain

    def accept_residue(self, residue):
        resname = residue.get_resname().strip().upper()
        hetflag = residue.id[0].strip()

        # 标准氨基酸全部保留
        if Polypeptide.is_aa(residue, standard=False):
            return True

        # 删除水
        if self.remove_waters and resname in {"HOH", "WAT", "DOD"}:
            return False

        # 只保留指定辅因子，如 NAP
        if hetflag and resname in self.keep_cofac:
            return True

        # 其余杂原子全部删除，包括 E04
        return False

# 1. 下载 PDB
download_pdb(PDB_ID, RAW_PDB)
print("Downloaded:", RAW_PDB)

# 2. 读取结构
structure = get_structure(RAW_PDB, PDB_ID)

# 3. 提取原始 E04 坐标（用于 box）
e04_coords, e04_found = extract_reference_ligand_coords(
    structure,
    chain_id=TARGET_CHAIN,
    lig_resname=REFERENCE_LIGAND_RESNAME
)
print("Reference ligand found:", e04_found)

box_center = e04_coords.mean(axis=0)
box_span = e04_coords.max(axis=0) - e04_coords.min(axis=0)
box_size = np.maximum(box_span + 2 * BOX_PADDING, MIN_BOX_EDGE)

print("Docking box center:", box_center)
print("Docking box size  :", box_size)

# 4. 输出清洗后的 receptor：保留 chain A + NAP，删除 E04/水/其他杂原子
io_obj = PDBIO()
io_obj.set_structure(structure)
io_obj.save(str(RECEPTOR_CLEAN_PDB), ReceptorSelect(
    target_chain=TARGET_CHAIN,
    keep_cofac=KEEP_COFAC_RESNAMES,
    remove_waters=REMOVE_WATERS
))

print("Saved cleaned receptor PDB:", RECEPTOR_CLEAN_PDB)

# 5. 保存 box 参数
box_info = {
    "pdb_id": PDB_ID,
    "chain": TARGET_CHAIN,
    "reference_ligand": REFERENCE_LIGAND_RESNAME,
    "keep_cofactor": sorted(list(KEEP_COFAC_RESNAMES)),
    "center_x": float(box_center[0]),
    "center_y": float(box_center[1]),
    "center_z": float(box_center[2]),
    "size_x": float(box_size[0]),
    "size_y": float(box_size[1]),
    "size_z": float(box_size[2]),
}
with open(WORKDIR / "docking_box_from_E04.json", "w") as f:
    json.dump(box_info, f, indent=2)

print(json.dumps(box_info, indent=2))








import subprocess

# 加氢
cmd1 = f'obabel "{RECEPTOR_CLEAN_PDB}" -O "{RECEPTOR_H_PDB}" -h'
print(cmd1)
subprocess.run(cmd1, shell=True, check=True)

# 转 receptor pdbqt
cmd2 = f'obabel "{RECEPTOR_H_PDB}" -O "{RECEPTOR_PDBQT}" -xr'
print(cmd2)
subprocess.run(cmd2, shell=True, check=True)

print("Receptor PDBQT ready:", RECEPTOR_PDBQT)







import pandas as pd
import numpy as np

df = pd.read_csv(CSV_PATH).copy()

required_cols = ["SMILES", "Catalog_ID"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"CSV 缺少必须列: {c}")

if "VIDA Name" not in df.columns:
    df["VIDA Name"] = df["Catalog_ID"].astype(str)

if "predicted_pIC50_value" not in df.columns:
    df["predicted_pIC50_value"] = np.nan

df["Catalog_ID"] = df["Catalog_ID"].astype(str)
df = df.dropna(subset=["SMILES", "Catalog_ID"]).reset_index(drop=True)

print("Ligands in CSV:", len(df))
display(df[["Catalog_ID", "VIDA Name", "SMILES", "predicted_pIC50_value"]].head())







import os
import subprocess
from tqdm.auto import tqdm

prep_records = []
prep_failed = []

for row in tqdm(df.itertuples(index=False), total=len(df), desc="Preparing ligands"):
    lig_id = str(row.Catalog_ID).replace("/", "_").replace("\\", "_").replace(" ", "_")
    smiles = str(row.SMILES).strip()

    smi_path = LIGAND_DIR / f"{lig_id}.smi"
    sdf_path = LIGAND_DIR / f"{lig_id}.sdf"
    pdbqt_path = LIGAND_DIR / f"{lig_id}.pdbqt"

    smi_path.write_text(smiles + f" {lig_id}\n")

    try:
        # 3D + 加氢 + 简单最小化
        cmd_sdf = (
            f'obabel -ismi "{smi_path}" -osdf -O "{sdf_path}" '
            f'--gen3d -h --minimize --ff MMFF94 --steps 250'
        )
        subprocess.run(cmd_sdf, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 转 pdbqt
        cmd_pdbqt = f'obabel "{sdf_path}" -opdbqt -O "{pdbqt_path}"'
        subprocess.run(cmd_pdbqt, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        prep_records.append({
            "Catalog_ID": lig_id,
            "sdf_file": str(sdf_path),
            "pdbqt_file": str(pdbqt_path)
        })
    except Exception as e:
        prep_failed.append({
            "Catalog_ID": lig_id,
            "SMILES": smiles,
            "reason": str(e)
        })

prep_df = pd.DataFrame(prep_records)
prep_failed_df = pd.DataFrame(prep_failed)

prep_df.to_csv(WORKDIR / "ligand_preparation_success.csv", index=False)
prep_failed_df.to_csv(WORKDIR / "ligand_preparation_failed.csv", index=False)

print("Prepared:", len(prep_df))
print("Failed  :", len(prep_failed_df))
display(prep_df.head())
if len(prep_failed_df):
    display(prep_failed_df.head())








from vina import Vina
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

results = []
dock_failed = []

# 只对成功准备的 ligand 进行 docking
dock_df = df.merge(prep_df, on="Catalog_ID", how="inner").copy()

for row in tqdm(dock_df.itertuples(index=False), total=len(dock_df), desc="Docking with Vina"):
    lig_id = str(row.Catalog_ID)
    lig_pdbqt = row.pdbqt_file
    out_pose = POSE_DIR / f"{lig_id}_vina_out.pdbqt"

    try:
        v = Vina(sf_name='vina', cpu=CPU, seed=SEED)
        v.set_receptor(str(RECEPTOR_PDBQT))
        v.compute_vina_maps(center=box_center.tolist(), box_size=box_size.tolist())

        v.set_ligand_from_file(str(lig_pdbqt))
        v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=NUM_MODES)
        energies = v.energies(n_poses=NUM_MODES)

        v.write_poses(str(out_pose), n_poses=NUM_MODES, energy_range=ENERGY_RANGE, overwrite=True)

        if energies is None or len(energies) == 0:
            dock_failed.append({
                "Catalog_ID": lig_id,
                "reason": "No energies returned"
            })
            continue

        top = energies[0]  # [affinity, rmsd_lb, rmsd_ub]
        results.append({
            "Catalog_ID": lig_id,
            "VIDA Name": getattr(row, "VIDA_Name", getattr(row, "VIDA Name", "")),
            "SMILES": row.SMILES,
            "predicted_pIC50_value": row.predicted_pIC50_value,
            "vina_affinity_kcal_mol": float(top[0]),
            "vina_rmsd_lb": float(top[1]),
            "vina_rmsd_ub": float(top[2]),
            "pose_file": str(out_pose)
        })
    except Exception as e:
        dock_failed.append({
            "Catalog_ID": lig_id,
            "reason": str(e)
        })

results_df = pd.DataFrame(results).sort_values("vina_affinity_kcal_mol", ascending=True).reset_index(drop=True)
dock_failed_df = pd.DataFrame(dock_failed)

results_df.to_csv(WORKDIR / "vina_docking_results.csv", index=False)
dock_failed_df.to_csv(WORKDIR / "vina_docking_failed.csv", index=False)

print("Docked successfully:", len(results_df))
print("Docking failed     :", len(dock_failed_df))
display(results_df.head(20))







from Bio.PDB import NeighborSearch

def load_pose_coords_from_pdbqt(pdbqt_path):
    coords = []
    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            elif line.startswith("ENDMDL"):
                break
    return np.array(coords, dtype=float)

def annotate_nearby_residues(receptor_pdb_path, pose_pdbqt_path, cutoff=4.5):
    structure = get_structure(receptor_pdb_path, "receptor")
    atoms = [a for a in structure.get_atoms()]
    ns = NeighborSearch(atoms)

    lig_xyz = load_pose_coords_from_pdbqt(pose_pdbqt_path)
    residues = set()

    for xyz in lig_xyz:
        near_atoms = ns.search(xyz, cutoff, level="A")
        for a in near_atoms:
            r = a.get_parent()
            c = r.get_parent()
            residues.add(f"{r.get_resname()}:{c.id}:{r.id[1]}")

    return ", ".join(sorted(residues))

nearby_list = []
for i, row in results_df.iterrows():
    nearby = annotate_nearby_residues(RECEPTOR_CLEAN_PDB, row["pose_file"], cutoff=4.5)
    nearby_list.append(nearby)

results_df["nearby_residues_4p5A"] = nearby_list
results_df.to_csv(WORKDIR / "vina_docking_results_with_contacts.csv", index=False)

display(results_df.head(20))






rank_df = results_df.copy()

if len(rank_df) > 0:
    # vina 越负越好
    vmin = rank_df["vina_affinity_kcal_mol"].min()
    vmax = rank_df["vina_affinity_kcal_mol"].max()
    rank_df["vina_norm"] = (vmax - rank_df["vina_affinity_kcal_mol"]) / (vmax - vmin + 1e-8)

    if rank_df["predicted_pIC50_value"].notna().sum() > 0:
        pmin = rank_df["predicted_pIC50_value"].min()
        pmax = rank_df["predicted_pIC50_value"].max()
        rank_df["pic50_norm"] = (rank_df["predicted_pIC50_value"] - pmin) / (pmax - pmin + 1e-8)
    else:
        rank_df["pic50_norm"] = 0.0

    rank_df["consensus_score"] = 0.6 * rank_df["vina_norm"] + 0.4 * rank_df["pic50_norm"]
    rank_df = rank_df.sort_values("consensus_score", ascending=False).reset_index(drop=True)

rank_df.to_csv(WORKDIR / "vina_consensus_ranked_results.csv", index=False)
display(rank_df.head(20))








import py3Dmol

raw_pdb_text = RAW_PDB.read_text()
top_pose_file = results_df.loc[0, "pose_file"] if len(results_df) else None
top_pose_text = Path(top_pose_file).read_text() if top_pose_file else None

view = py3Dmol.view(width=1000, height=700)
view.addModel(raw_pdb_text, "pdb")

# 蛋白 cartoon
view.setStyle({"chain": TARGET_CHAIN, "hetflag": False}, {"cartoon": {"color": "spectrum"}})

# NAP 显示为橙色 stick
view.setStyle({"resn": "NAP"}, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.25}})

# 原始 E04 显示为洋红
view.setStyle({"resn": "E04"}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.25}})

# docked top pose 显示为青色
if top_pose_text is not None:
    view.addModel(top_pose_text, "pdbqt")
    view.setStyle({"model": -1}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.25}})

# 画 box
cx, cy, cz = box_center
sx, sy, sz = box_size
view.addBox({
    "center": {"x": float(cx), "y": float(cy), "z": float(cz)},
    "dimensions": {"w": float(sx), "h": float(sy), "d": float(sz)},
    "color": "yellow",
    "alpha": 0.25
})

view.zoomTo({"chain": TARGET_CHAIN})
view.show()







import shutil

zip_base = str(WORKDIR)
zip_path = shutil.make_archive(zip_base, 'zip', root_dir=WORKDIR)

print("Zipped results:", zip_path)
from google.colab import files
files.download(zip_path)