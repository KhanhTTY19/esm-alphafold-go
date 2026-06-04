import requests, os
from tqdm import tqdm

with open("raw_data/human_protein_ids.txt") as f:
    protein_ids = [l.strip() for l in f if l.strip()]

print(f"Total proteins: {len(protein_ids)}")
out_dir = "raw_data/predicted_struct_protein_data"
failed = []

for pid in tqdm(protein_ids[:100]):  # thử 100 cái trước
    url = f"https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v4.pdb"
    out_path = f"{out_dir}/AF-{pid}-F1-model_v4.pdb"
    if os.path.exists(out_path):
        continue
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(r.content)
        else:
            failed.append(pid)
    except Exception as e:
        failed.append(pid)

print(f"Downloaded OK. Failed: {len(failed)}")
print("Failed IDs:", failed[:10])
