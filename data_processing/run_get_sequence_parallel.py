from Bio import SeqIO
import numpy as np
import os, pickle, warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def process_one(file_name):
    pdb_dir = "raw_data/predicted_struct_protein_data"
    parts = file_name.split("-")
    if len(parts) < 3:
        return None, None, None
    pid = parts[1]
    try:
        records = list(SeqIO.parse(os.path.join(pdb_dir, file_name), 'pdb-atom'))
        if not records:
            return None, None, None
        seq = str(records[0].seq)
        return pid, seq, seq2onehot(seq)
    except:
        return None, None, None

if __name__ == "__main__":
    pdb_dir = "raw_data/predicted_struct_protein_data"
    files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    print(f"Processing {len(files)} files with 30 workers...")

    protein_node2onehot = {}
    protein_sequence = {}

    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(process_one, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files)):
            pid, seq, onehot = future.result()
            if pid:
                protein_sequence[pid] = seq
                protein_node2onehot[pid] = onehot

    print(f"Done! {len(protein_node2onehot)} proteins")
    with open('processed_data/protein_node2onehot', 'wb') as f:
        pickle.dump(protein_node2onehot, f)
    with open('processed_data/protein_sequence', 'wb') as f:
        pickle.dump(protein_sequence, f)
    print("Saved!")
