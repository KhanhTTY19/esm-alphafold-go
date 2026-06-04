import gzip
import json

go_annotations = {}
with gzip.open("raw_data/goa_human.gaf.gz", "rt") as f:
    for line in f:
        if line.startswith("!"):
            continue
        cols = line.strip().split("\t")
        if len(cols) < 5:
            continue
        protein_id = cols[1]
        go_id = cols[4]
        if protein_id not in go_annotations:
            go_annotations[protein_id] = []
        go_annotations[protein_id].append(go_id)

with open("processed_data/HUMAN_protein_info.json", "w") as f:
    json.dump(go_annotations, f, indent=4)

print(f"Done! Total proteins: {len(go_annotations)}")
