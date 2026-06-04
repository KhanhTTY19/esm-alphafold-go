from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd
import os, pickle, warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def seq2onehot(seq):
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_embed = dict(zip(chars, range(len(chars))))
    vocab_one_hot = np.zeros((len(chars), len(chars)), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1
    embed_x = [vocab_embed.get(v, vocab_embed['-']) for v in seq]
    return np.array([vocab_one_hot[j, :] for j in embed_x])

pdb_dir = "raw_data/predicted_struct_protein_data"
protein_node2onehot = {}
protein_sequence = {}

files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
print(f"Processing {len(files)} PDB files...")

for file_name in tqdm(files):
    parts = file_name.split("-")
    if len(parts) < 3:
        continue
    pid = parts[1]
    try:
        pdbfile = os.path.join(pdb_dir, file_name)
        records = list(SeqIO.parse(pdbfile, 'pdb-atom'))
        if not records:
            continue
        seq = str(records[0].seq)
        protein_sequence[pid] = seq
        protein_node2onehot[pid] = seq2onehot(seq)
    except Exception as e:
        pass

print(f"Done! {len(protein_node2onehot)} proteins")

with open('processed_data/protein_node2onehot', 'wb') as f:
    pickle.dump(protein_node2onehot, f)
with open('processed_data/protein_sequence', 'wb') as f:
    pickle.dump(protein_sequence, f)
print("Saved protein_node2onehot and protein_sequence")
