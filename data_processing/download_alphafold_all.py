import requests, os, json
from tqdm import tqdm

with open("processed_data/HUMAN_protein_info.json") as f:
    go_data = json.load(f)

swissprot_ids = [pid for pid in go_data.keys() 
                 if len(pid) == 6 and pid[0] in 'PQOAB' and not pid.startswith('A0A')]
print(f"Total: {len(swissprot_ids)} proteins to download")

out_dir = "raw_data/predicted_struct_protein_data"
os.makedirs(out_dir, exist_ok=True)
failed = []

for pid in tqdm(swissprot_ids):
    out_path = f"{out_dir}/AF-{pid}-F1-model.pdb"
    if os.path.exists(out_path):
        continue
    try:
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{pid}"
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            pdb_url = r.json()[0].get('pdbUrl', '')
            if pdb_url:
                r2 = requests.get(pdb_url, timeout=60)
                if r2.status_code == 200:
                    with open(out_path, 'wb') as f:
                        f.write(r2.content)
                else:
                    failed.append(pid)
            else:
                failed.append(pid)
        else:
            failed.append(pid)
    except Exception as e:
        failed.append(pid)

downloaded = len(os.listdir(out_dir))
print(f"\nDownloaded: {downloaded}, Failed: {len(failed)}")
with open("raw_data/failed_ids.txt", "w") as f:
    f.write("\n".join(failed))
print("Failed IDs saved to raw_data/failed_ids.txt")
