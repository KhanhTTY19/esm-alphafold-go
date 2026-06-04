import os, pickle, warnings
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

def process_one(file_name):
    contact_dir = "data/protein_contact_map"
    out_dir = "processed_data/node2vec_features"
    
    pid = file_name.split(".")[0]
    out_path = f"{out_dir}/{pid}.txt"
    if os.path.exists(out_path):
        return pid, True

    try:
        input_file = os.path.join(contact_dir, file_name)
        G = nx.read_edgelist(input_file, nodetype=int, data=False, delimiter=" ")
        if G.number_of_nodes() == 0:
            return pid, False

        n2v = Node2Vec(G, dimensions=128, walk_length=30, num_walks=10,
                       p=0.8, q=1.2, workers=1, quiet=True)
        model = n2v.fit(window=10, min_count=1, batch_words=128, epochs=1)

        # Lưu theo thứ tự node index để np.loadtxt đọc được
        nodes = sorted([int(n) for n in model.wv.index_to_key])
        embeddings = np.array([model.wv[str(n)] for n in nodes])
        np.savetxt(out_path, embeddings)
        return pid, True
    except Exception as e:
        return pid, False

if __name__ == "__main__":
    os.makedirs("processed_data/node2vec_features", exist_ok=True)
    
    files = [f for f in os.listdir("data/protein_contact_map") if f.endswith('.txt')]
    # Bỏ qua file đã xử lý
    files = [f for f in files 
             if not os.path.exists(f"processed_data/node2vec_features/{f.split('.')[0]}.txt")]
    print(f"Processing {len(files)} proteins with 30 workers...")

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(process_one, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files)):
            pid, success = future.result()
            if success:
                ok += 1
            else:
                fail += 1

    print(f"Done! OK: {ok}, Failed: {fail}")

    # Gộp tất cả vào 1 pickle
    print("Assembling protein_node2vec pickle...")
    protein_node2vec = {}
    for f in tqdm(os.listdir("processed_data/node2vec_features")):
        pid = f.split(".")[0]
        try:
            data = np.loadtxt(f"processed_data/node2vec_features/{f}")
            protein_node2vec[pid] = data
        except:
            pass

    with open("processed_data/protein_node2vec", "wb") as f:
        pickle.dump(protein_node2vec, f)
    print(f"Saved protein_node2vec: {len(protein_node2vec)} proteins")
