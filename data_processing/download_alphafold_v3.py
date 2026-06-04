import requests, os, json
from tqdm import tqdm

with open("processed_data/HUMAN_protein_info.json") as f:
    go_data = json.load(f)

swissprot_ids = [pid for pid in go_data.keys() 
                 if len(pid) == 6 and pid[0] in 'PQOAB' and not pid.startswith('A0A')]
print(f"Total SwissProt proteins with GO: {len(swissprot_ids)}")

out_dir = "raw_data/predicted_struct_protein_data"
os.makedirs(out_dir, exist_ok=True)
failed = []

for pid in tqdm(swissprot_ids[:20]):  # test 20 trước
    out_path = f"{out_dir}/AF-{pid}-F1-model.pdb"
    if os.path.exists(out_path):
        continue
    try:
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{pid}"
        r = requests.get(api_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            pdb_url = data[0].get('pdbUrl', '')
            if pdb_url:
                r2 = requests.get(pdb_url, timeout=60)
                if r2.status_code == 200:
                    with open(out_path, 'wb') as f:
                        f.write(r2.content)
                    print(f"OK: {pid}")
                else:
                    failed.append(pid)
        else:
            failed.append(pid)
    except Exception as e:
        print(f"Error {pid}: {e}")
        failed.append(pid)

downloaded = len(os.listdir(out_dir))
print(f"\nDownloaded: {downloaded}, Failed: {len(failed)}")
print("Failed:", failed)
