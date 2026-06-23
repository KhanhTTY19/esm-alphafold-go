import torch

models = [
    'save_models/bestmodel_cc_32_0.0001_0.1.pkl',
    'save_models/bestmodel_mf_32_0.0001_0.1.pkl',
]

for path in models:
    model = torch.load(path)
    first_conv = model.convpools[0].conv1
    print(f"{path}: in={first_conv.in_feats}")