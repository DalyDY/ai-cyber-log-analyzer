import pandas as pd

df = pd.read_csv(
    "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print("\nLabels:")
print(df["Label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nData types:")
print(df.dtypes.head())