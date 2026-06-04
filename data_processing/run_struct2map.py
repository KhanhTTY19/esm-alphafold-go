from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd
import os, warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_predicted_PDB(pdbfile):
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues() if r.has_id("CA")]
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]
    if not seqs or not residues:
        return None, None
    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)
    return distances, seqs[0]

pdb_dir = "raw_data/predicted_struct_protein_data"
out_dir = "data/protein_contact_map"
os.makedirs(out_dir, exist_ok=True)

files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
print(f"Processing {len(files)} PDB files...")

for file_name in tqdm(files):
    # AF-P04637-F1-model.pdb -> P04637
    parts = file_name.split("-")
    if len(parts) < 3:
        continue
    name = parts[1]
    out_path = f"{out_dir}/{name}.txt"
    if os.path.exists(out_path):
        continue
    try:
        D, seq = load_predicted_PDB(os.path.join(pdb_dir, file_name))
        if D is None:
            continue
        A = np.double(D < 10.0)
        result = [[i, j] for i in range(len(A)) 
                         for j in range(len(A)) if A[i][j] and i != j]
        pd.DataFrame(result).to_csv(out_path, sep=" ", index=False, header=False)
    except Exception as e:
        print(f"Error {name}: {e}")

print(f"Done! Contact maps: {len(os.listdir(out_dir))}")
