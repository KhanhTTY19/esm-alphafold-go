import requests, os, json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

with open("processed_data/HUMAN_protein_info.json") as f:
    go_data = json.load(f)

swissprot_ids = [pid for pid in go_data.keys() 
                 if len(pid) == 6 and pid[0] in 'PQOAB' and not pid.startswith('A0A')]

out_dir = "raw_data/predicted_struct_protein_data"
# Bỏ qua file đã download
swissprot_ids = [pid for pid in swissprot_ids 
                 if not os.path.exists(f"{out_dir}/AF-{pid}-F1-model.pdb")]
print(f"Remaining: {len(swissprot_ids)} proteins")

failed = []

def download_one(pid):
    out_path = f"{out_dir}/AF-{pid}-F1-model.pdb"
    try:
        r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{pid}", timeout=30)
        if r.status_code == 200:
            pdb_url = r.json()[0].get('pdbUrl', '')
            if pdb_url:
                r2 = requests.get(pdb_url, timeout=60)
                if r2.status_code == 200:
                    with open(out_path, 'wb') as f:
                        f.write(r2.content)
                    return pid, True
        return pid, False
    except:
        return pid, False

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(download_one, pid): pid for pid in swissprot_ids}
    for future in tqdm(as_completed(futures), total=len(swissprot_ids)):
        pid, ok = future.result()
        if not ok:
            failed.append(pid)

print(f"Downloaded: {len(os.listdir(out_dir))}, Failed: {len(failed)}")
with open("raw_data/failed_ids.txt", "w") as f:
    f.write("\n".join(failed))
