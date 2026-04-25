!apt-get -qq update
!apt-get -qq install -y openbabel > /dev/null

!pip -q install vina biopython pandas numpy scipy py3Dmol tqdm requests





from google.colab import files
import os

uploaded = files.upload()

csv_candidates = [k for k in uploaded.keys() if k.lower().endswith(".csv")]
assert len(csv_candidates) >= 1, "请上传 Enamine_QSAR_Pharm.csv"

CSV_PATH = "/content/" + csv_candidates[0]
print("Using CSV:", CSV_PATH)






from pathlib import Path

# =========================
# Core target settings
# =========================
PDB_ID = "8RRJ"
TARGET_CHAIN = "A"

# keep/remove settings
KEEP_COFAC_RESNAMES = {"NAP"}          # 保留 NADP/NAP
REFERENCE_LIGAND_RESNAME = "A1H2U"     # 8RRJ 的共晶抑制剂
REMOVE_WATERS = True
REMOVE_IONS = True

ION_RESNAMES = {
    "NA","K","CL","CA","MG","ZN","MN","FE","CU","CO","NI","CD","HG"
}

# docking box
BOX_PADDING = 8.0
MIN_BOX_EDGE = 22.0

# docking parameters
EXHAUSTIVENESS = 32
NUM_MODES = 20
ENERGY_RANGE = 4
CPU = 2
SEED = 2026

# output
WORKDIR = Path("/content/AKR1C3_8RRJ_Vina_keepNAP")
WORKDIR.mkdir(parents=True, exist_ok=True)

RAW_CIF = WORKDIR / f"{PDB_ID}.cif"
RECEPTOR_CLEAN_PDB = WORKDIR / f"{PDB_ID}_chain{TARGET_CHAIN}_keepNAP_noLig_clean.pdb"
RECEPTOR_H_PDB = WORKDIR / f"{PDB_ID}_chain{TARGET_CHAIN}_keepNAP_noLig_H.pdb"
RECEPTOR_PDBQT = WORKDIR / f"{PDB_ID}_chain{TARGET_CHAIN}_keepNAP_noLig.pdbqt"

LIGAND_DIR = WORKDIR / "ligands"
POSE_DIR = WORKDIR / "docked_poses"
LIGAND_DIR.mkdir(exist_ok=True)
POSE_DIR.mkdir(exist_ok=True)

print("WORKDIR:", WORKDIR)







import json
import requests
import numpy as np
from scipy.spatial.distance import cdist

from Bio.PDB import MMCIFParser, PDBIO, Select, Polypeptide

def download_cif(pdb_id: str, outpath):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    outpath.write_text(r.text)
    return outpath

def get_structure_from_cif(cif_path, structure_id="struct"):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(structure_id, str(cif_path))

def atom_element(atom):
    elem = (atom.element or "").strip()
    if elem:
        return elem.upper()
    name = atom.get_name().strip()
    letters = "".join([c for c in name if c.isalpha()])
    return letters[:1].upper() if letters else "C"

def residue_heavy_coords(residue):
    coords = []
    for atom in residue:
        if atom_element(atom) != "H":
            coords.append(atom.coord)
    return np.array(coords, dtype=float) if len(coords) else np.zeros((0, 3), dtype=float)

def chain_heavy_coords(chain):
    coords = []
    for residue in chain:
        if Polypeptide.is_aa(residue, standard=False):
            for atom in residue:
                if atom_element(atom) != "H":
                    coords.append(atom.coord)
    return np.array(coords, dtype=float) if len(coords) else np.zeros((0, 3), dtype=float)

def residue_min_distance_to_coords(residue, coords_ref):
    a = residue_heavy_coords(residue)
    if len(a) == 0 or len(coords_ref) == 0:
        return np.inf
    return float(cdist(a, coords_ref).min())

def residue_com(residue):
    xyz = residue_heavy_coords(residue)
    if len(xyz) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return xyz.mean(axis=0)

def residue_signature(residue):
    chain = residue.get_parent().id
    resname = residue.get_resname().strip()
    hetflag, resid, icode = residue.id
    return f"{resname} chain={chain} resid={resid}{icode.strip()}"

def sanitize_id(x):
    x = str(x)
    for bad in ["/", "\\", " ", ":", ";", "|", ",", "(", ")", "[", "]", "{", "}", "'","\""]:
        x = x.replace(bad, "_")
    return x

class ResidueSelection(Select):
    def __init__(self, keep_residue_ids):
        self.keep_residue_ids = set(keep_residue_ids)

    def accept_residue(self, residue):
        return residue.full_id in self.keep_residue_ids

    def accept_atom(self, atom):
        altloc = atom.get_altloc()
        elem = atom_element(atom)
        if altloc not in (" ", "A", "1"):
            return False
        if elem == "H":
            return False
        return True

def residue_to_pdb_block(residue, new_resname="LIG", new_chain="Z", model_id=1):
    lines = ["MODEL     %4d" % model_id]
    serial = 1
    resseq = 1
    for atom in residue:
        elem = atom_element(atom)
        if elem == "H":
            continue
        name = atom.get_name().strip()
        x, y, z = atom.coord
        # 4-char atom field
        atom_name = name[:4].rjust(4)
        line = (
            f"HETATM{serial:5d} {atom_name} {new_resname:>3s} {new_chain:1s}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {elem:>2s}"
        )
        lines.append(line)
        serial += 1
    lines.append("ENDMDL")
    return "\n".join(lines) + "\n"

# -------------------------
# 1) download and parse
# -------------------------
download_cif(PDB_ID, RAW_CIF)
print("Downloaded CIF:", RAW_CIF)

structure = get_structure_from_cif(RAW_CIF, PDB_ID)
model = structure[0]

assert TARGET_CHAIN in model.child_dict, f"Target chain {TARGET_CHAIN} not found in {PDB_ID}"

protein_chain = model[TARGET_CHAIN]
protein_coords = chain_heavy_coords(protein_chain)
assert len(protein_coords) > 0, "No protein atoms found in target chain."

# -------------------------
# 2) choose nearest NAP to chain A
# -------------------------
nap_candidates = []
lig_candidates = []

for chain in model:
    for residue in chain:
        resname = residue.get_resname().strip().upper()
        if resname in KEEP_COFAC_RESNAMES:
            nap_candidates.append(residue)
        elif resname == REFERENCE_LIGAND_RESNAME.upper():
            lig_candidates.append(residue)

assert len(nap_candidates) > 0, f"No {KEEP_COFAC_RESNAMES} found in structure."
assert len(lig_candidates) > 0, f"No reference ligand {REFERENCE_LIGAND_RESNAME} found in structure."

selected_nap = min(nap_candidates, key=lambda r: residue_min_distance_to_coords(r, protein_coords))
selected_lig = min(lig_candidates, key=lambda r: residue_min_distance_to_coords(r, protein_coords))

print("Selected NAP near chain", TARGET_CHAIN, ":", residue_signature(selected_nap))
print("Selected reference ligand     :", residue_signature(selected_lig))

# -------------------------
# 3) define docking box from original A1H2U coords
# -------------------------
lig_xyz = residue_heavy_coords(selected_lig)
assert len(lig_xyz) > 0, "Selected ligand has no heavy atoms."

box_center = lig_xyz.mean(axis=0)
box_span = lig_xyz.max(axis=0) - lig_xyz.min(axis=0)
box_size = np.maximum(box_span + 2 * BOX_PADDING, MIN_BOX_EDGE)

print("Box center:", box_center)
print("Box size  :", box_size)

# -------------------------
# 4) keep: chain A protein residues + selected NAP only
# remove: selected ligand, ions, waters, all other hetero residues
# -------------------------
keep_ids = []

for residue in protein_chain:
    if Polypeptide.is_aa(residue, standard=False):
        keep_ids.append(residue.full_id)

keep_ids.append(selected_nap.full_id)

io_obj = PDBIO()
io_obj.set_structure(structure)
io_obj.save(str(RECEPTOR_CLEAN_PDB), ResidueSelection(keep_ids))

print("Saved cleaned receptor PDB:", RECEPTOR_CLEAN_PDB)

# -------------------------
# 5) save metadata
# -------------------------
meta = {
    "pdb_id": PDB_ID,
    "target_chain": TARGET_CHAIN,
    "kept_cofactor": residue_signature(selected_nap),
    "removed_reference_ligand": residue_signature(selected_lig),
    "reference_ligand_resname": REFERENCE_LIGAND_RESNAME,
    "center_x": float(box_center[0]),
    "center_y": float(box_center[1]),
    "center_z": float(box_center[2]),
    "size_x": float(box_size[0]),
    "size_y": float(box_size[1]),
    "size_z": float(box_size[2]),
}
with open(WORKDIR / "docking_setup_8RRJ.json", "w") as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))






import subprocess

cmd1 = f'obabel "{RECEPTOR_CLEAN_PDB}" -O "{RECEPTOR_H_PDB}" -h'
print(cmd1)
subprocess.run(cmd1, shell=True, check=True)

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
df["SMILES"] = df["SMILES"].astype(str)
df = df.dropna(subset=["SMILES", "Catalog_ID"]).reset_index(drop=True)

print("Ligands in CSV:", len(df))
display(df[["Catalog_ID", "VIDA Name", "SMILES", "predicted_pIC50_value"]].head())









import subprocess
from tqdm.auto import tqdm

prep_records = []
prep_failed = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing ligands"):
    lig_id = sanitize_id(row["Catalog_ID"])
    smiles = str(row["SMILES"]).strip()

    smi_path = LIGAND_DIR / f"{lig_id}.smi"
    sdf_path = LIGAND_DIR / f"{lig_id}.sdf"
    pdbqt_path = LIGAND_DIR / f"{lig_id}.pdbqt"

    smi_path.write_text(smiles + f" {lig_id}\n")

    try:
        # 1) generate 3D + add H
        cmd_sdf = f'obabel -ismi "{smi_path}" -osdf -O "{sdf_path}" --gen3d -h'
        subprocess.run(cmd_sdf, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2) optional quick minimization
        cmd_min = f'obabel "{sdf_path}" -O "{sdf_path}" --minimize --ff MMFF94 --steps 250'
        subprocess.run(cmd_min, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3) to pdbqt
        cmd_pdbqt = f'obabel "{sdf_path}" -opdbqt -O "{pdbqt_path}"'
        subprocess.run(cmd_pdbqt, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        prep_records.append({
            "Catalog_ID": row["Catalog_ID"],
            "VIDA Name": row["VIDA Name"],
            "SMILES": row["SMILES"],
            "predicted_pIC50_value": row["predicted_pIC50_value"],
            "sdf_file": str(sdf_path),
            "pdbqt_file": str(pdbqt_path)
        })
    except Exception as e:
        prep_failed.append({
            "Catalog_ID": row["Catalog_ID"],
            "SMILES": row["SMILES"],
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

dock_df = prep_df.copy()

for _, row in tqdm(dock_df.iterrows(), total=len(dock_df), desc="Docking with Vina"):
    lig_id = sanitize_id(row["Catalog_ID"])
    lig_pdbqt = row["pdbqt_file"]
    out_pose = POSE_DIR / f"{lig_id}_vina_out.pdbqt"

    try:
        v = Vina(sf_name="vina", cpu=CPU, seed=SEED)
        v.set_receptor(str(RECEPTOR_PDBQT))
        v.compute_vina_maps(
            center=box_center.tolist(),
            box_size=box_size.tolist()
        )
        v.set_ligand_from_file(str(lig_pdbqt))
        v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=NUM_MODES)

        energies = v.energies(n_poses=NUM_MODES)
        v.write_poses(
            str(out_pose),
            n_poses=NUM_MODES,
            energy_range=ENERGY_RANGE,
            overwrite=True
        )

        if energies is None or len(energies) == 0:
            dock_failed.append({
                "Catalog_ID": row["Catalog_ID"],
                "reason": "No docking energies returned"
            })
            continue

        top = energies[0]  # affinity, rmsd_lb, rmsd_ub

        results.append({
            "Catalog_ID": row["Catalog_ID"],
            "VIDA Name": row["VIDA Name"],
            "SMILES": row["SMILES"],
            "predicted_pIC50_value": row["predicted_pIC50_value"],
            "vina_affinity_kcal_mol": float(top[0]),
            "vina_rmsd_lb": float(top[1]),
            "vina_rmsd_ub": float(top[2]),
            "pose_file": str(out_pose)
        })

    except Exception as e:
        dock_failed.append({
            "Catalog_ID": row["Catalog_ID"],
            "reason": str(e)
        })

results_df = pd.DataFrame(results)
dock_failed_df = pd.DataFrame(dock_failed)

if len(results_df):
    results_df = results_df.sort_values("vina_affinity_kcal_mol", ascending=True).reset_index(drop=True)

results_df.to_csv(WORKDIR / "vina_docking_results.csv", index=False)
dock_failed_df.to_csv(WORKDIR / "vina_docking_failed.csv", index=False)

print("Docked successfully:", len(results_df))
print("Docking failed     :", len(dock_failed_df))

display(results_df.head(20))
if len(dock_failed_df):
    display(dock_failed_df.head())






from Bio.PDB import PDBParser, NeighborSearch

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
    return np.array(coords, dtype=float) if len(coords) else np.zeros((0,3), dtype=float)

def annotate_nearby_residues(receptor_pdb_path, pose_pdbqt_path, cutoff=4.5):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(receptor_pdb_path))
    atoms = [a for a in structure.get_atoms()]
    ns = NeighborSearch(atoms)

    lig_xyz = load_pose_coords_from_pdbqt(pose_pdbqt_path)
    residues = set()

    for xyz in lig_xyz:
        near_atoms = ns.search(xyz, cutoff, level="A")
        for atom in near_atoms:
            residue = atom.get_parent()
            chain = residue.get_parent()
            residues.add(f"{residue.get_resname()}:{chain.id}:{residue.id[1]}")

    return ", ".join(sorted(residues))

if len(results_df):
    nearby_list = []
    for _, row in results_df.iterrows():
        nearby = annotate_nearby_residues(RECEPTOR_CLEAN_PDB, row["pose_file"], cutoff=4.5)
        nearby_list.append(nearby)

    results_df["nearby_residues_4p5A"] = nearby_list
    results_df.to_csv(WORKDIR / "vina_docking_results_with_contacts.csv", index=False)

display(results_df.head(20))






rank_df = results_df.copy()

if len(rank_df):
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
from pathlib import Path

receptor_pdb_text = Path(RECEPTOR_CLEAN_PDB).read_text()
ref_lig_pdb_block = residue_to_pdb_block(selected_lig, new_resname="LIG", new_chain="Z", model_id=1)

view = py3Dmol.view(width=1000, height=700)

# receptor (protein A + selected NAP)
view.addModel(receptor_pdb_text, "pdb")
view.setStyle({"chain": TARGET_CHAIN, "hetflag": False}, {"cartoon": {"color": "spectrum"}})
view.setStyle({"resn": "NAP"}, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.25}})

# original reference ligand A1H2U shown as generic LIG
view.addModel(ref_lig_pdb_block, "pdb")
view.setStyle({"model": 1}, {"stick": {"colorscheme": "magentaCarbon", "radius": 0.25}})

# top docked pose
if len(results_df):
    top_pose_text = Path(results_df.loc[0, "pose_file"]).read_text()
    view.addModel(top_pose_text, "pdbqt")
    view.setStyle({"model": 2}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.25}})

# docking box
cx, cy, cz = box_center
sx, sy, sz = box_size
view.addBox({
    "center": {"x": float(cx), "y": float(cy), "z": float(cz)},
    "dimensions": {"w": float(sx), "h": float(sy), "d": float(sz)},
    "color": "yellow",
    "alpha": 0.25
})

view.zoomTo()
view.show()






import shutil
from google.colab import files

zip_base = str(WORKDIR)
zip_path = shutil.make_archive(zip_base, "zip", root_dir=WORKDIR)

print("Zipped results:", zip_path)
files.download(zip_path)