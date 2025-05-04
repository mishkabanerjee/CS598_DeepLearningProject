import pandas as pd

# Path to the labels file
label_path = "data/circulatory_failure_labels.csv"

# Load it
df = pd.read_csv(label_path)

# Show basic stats
print("✅ Total patients in label file:", len(df))
print("✅ Patients labeled with circulatory failure:", df['label'].sum())
print("✅ Patients without circulatory failure:", (df['label'] == 0).sum())

# Peek at a few entries
print("\n🔍 Sample rows:")
print(df.sample(10).sort_values(by="label", ascending=False))
