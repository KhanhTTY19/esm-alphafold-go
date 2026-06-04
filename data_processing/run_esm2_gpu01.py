import os, pickle, gc, argparse
import torch
import torch.nn as nn
import esm
from Bio import SeqIO
import numpy as np
from tqdm import tqdm

FASTA_PATH = "raw_data/UP000005640_9606.fasta"
MODEL_PATH = "raw_data/esm2/esm2_t33_650M_UR50D.pt"
OUTPUT_PATH = "processed_data/dict_sequence_feature"
BATCH_SIZE = 16  # 2 GPU nên tăng batch size lên
MAX_SEQ_LEN = 1022

with open("data/protein_list.csv") as f:
    valid_ids = set(l.strip() for l in f if l.strip())
print(f"Valid proteins: {len(valid_ids)}")

print("Reading FASTA...")
protein_data = []
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    pid = record.id.split("|")[1] if "|" in record.id else record.id
    if pid in valid_ids:
        seq = str(record.seq)
        if len(seq) <= MAX_SEQ_LEN:
            protein_data.append((pid, seq))
print(f"Proteins to embed: {len(protein_data)}")

print("Loading ESM-2 650M...")
model, alphabet = esm.pretrained.load_model_and_alphabet_local(MODEL_PATH)
batch_converter = alphabet.get_batch_converter()
model.eval()

# Dùng GPU 0 và 1
device = torch.device("cuda:0")
model = nn.DataParallel(model, device_ids=[0, 1])
model = model.to(device)
print("Model loaded on GPU 0 and 1")

embeddings_dict = {}

# Load checkpoint nếu có
ckpt_path = OUTPUT_PATH + ".ckpt"
if os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f:
        embeddings_dict = pickle.load(f)
    print(f"Resumed from checkpoint: {len(embeddings_dict)} proteins done")

# Bỏ qua protein đã xử lý
done_ids = set(embeddings_dict.keys())
protein_data = [(pid, seq) for pid, seq in protein_data if pid not in done_ids]
print(f"Remaining: {len(protein_data)} proteins")

for i in tqdm(range(0, len(protein_data), BATCH_SIZE)):
    batch = protein_data[i:i+BATCH_SIZE]
    try:
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model.module(batch_tokens, repr_layers=[33])
        token_repr = results["representations"][33]

        for j, tokens_len in enumerate(batch_lens):
            embedding = token_repr[j, 1:tokens_len-1].mean(0)
            embeddings_dict[batch_labels[j]] = embedding.cpu().numpy()

        del batch_tokens, token_repr, results
        torch.cuda.empty_cache()
        gc.collect()

        # Checkpoint mỗi 200 batches
        if (i // BATCH_SIZE) % 200 == 0 and i > 0:
            with open(ckpt_path, "wb") as f:
                pickle.dump(embeddings_dict, f)

    except Exception as e:
        print(f"Error batch {i}: {e}")
        continue

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings_dict, f)
print(f"Done! {len(embeddings_dict)} embeddings saved")
