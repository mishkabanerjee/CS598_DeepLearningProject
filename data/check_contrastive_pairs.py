import os, json

acf_dir = r"C:\Users\mishka.banerjee\Documents\UIUC\Deep Learning\hirid\acf_neighbors"
valid_patients = 0

for fname in os.listdir(acf_dir):
    with open(os.path.join(acf_dir, fname), 'r') as f:
        acf_map = json.load(f)
        if any(len(v["pos"]) > 0 or len(v["neg"]) > 0 for v in acf_map.values()):
            valid_patients += 1

print(f"✅ Valid patients with usable contrastive pairs: {valid_patients}")
