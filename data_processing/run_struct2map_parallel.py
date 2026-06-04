from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd
import os, warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

def process_one(file_name):
    pdb_dir = "raw_data/predicted_struct_protein_data"
    out_dir = "data/protein_contact_map"
    
    parts = file_name.split("-")
    if len(parts) < 3:
        return False
    name = parts[1]
    out_path = f"{out_dir}/{name}.txt"
    if os.path.exists(out_path):
        return True
    try:
        pdbfile = os.path.join(pdb_dir, file_name)
        parser = PDBParser()
        structure = parser.get_structure(name, pdbfile)
        residues = [r for r in structure.get_residues() if r.has_id("CA")]
        records = SeqIO.parse(pdbfile, 'pdb-atom')
        seqs = [str(r.seq) for r in records]
        if not seqs or not residues:
            return False
        
        # Dùng numpy vectorized thay vì double loop
        coords = np.array([r["CA"].get_coord() for r in residues])
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt((diff**2).sum(axis=-1))
        
        A = distances < 10.0
        np.fill_diagonal(A, False)
        rows, cols = np.where(A)
        result = np.column_stack([rows, cols])
        pd.DataFrame(result).to_csv(out_path, sep=" ", index=False, header=False)
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    pdb_dir = "raw_data/predicted_struct_protein_data"
    out_dir = "data/protein_contact_map"
    os.makedirs(out_dir, exist_ok=True)
    
    files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    # Bỏ qua file đã xử lý
    files = [f for f in files if not os.path.exists(
        f"{out_dir}/{f.split('-')[1]}.txt")]
    print(f"Processing {len(files)} PDB files with 8 workers...")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one, f): f for f in files}
        ok, fail = 0, 0
        for future in tqdm(as_completed(futures), total=len(files)):
            if future.result():
                ok += 1
            else:
                fail += 1
    
    print(f"Done! OK: {ok}, Failed: {fail}")
    print(f"Contact maps: {len(os.listdir(out_dir))}")
