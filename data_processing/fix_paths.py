import re

with open("model/labels_load.py", "r") as f:
    content = f.read()

replacements = {
    "/mnt/workspace/replicate/Struct2GO/processed_data/HUMAN_protein_info.json": "processed_data/HUMAN_protein_info.json",
    "/mnt/workspace/replicate/Struct2GO/raw_data/go.obo": "raw_data/go.obo",
    "/mnt/workspace/replicate/Struct2GO/data/protein_list.csv": "data/protein_list.csv",
    "../processed_data/protein_node2onehot": "processed_data/protein_node2onehot",
    "/mnt/workspace/replicate/Struct2GO/processed_data/protein_node2vec": "processed_data/protein_node2vec",
    "/mnt/workspace/replicate/Struct2GO/processed_data/dict_sequence_feature": "processed_data/dict_sequence_feature",
    "/mnt/workspace/replicate/Struct2GO/data/protein_contact_map": "data/protein_contact_map",
    "/mnt/workspace/replicate/Struct2GO/processed_data/": "processed_data/",
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open("model/labels_load.py", "w") as f:
    f.write(content)

print("Paths fixed!")
