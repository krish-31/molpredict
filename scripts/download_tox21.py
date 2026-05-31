"""
download_tox21.py
─────────────────
Download and validate the Tox21 CSV dataset from MoleculeNet S3.
"""
import gzip, os, shutil, urllib.request

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CSV_PATH = os.path.join(OUT_DIR, "tox21.csv")
GZ_PATH  = os.path.join(OUT_DIR, "tox21.csv.gz")

os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(CSV_PATH):
    print(f"Already downloaded: {CSV_PATH}")
else:
    print(f"Downloading Tox21 from {URL} …")
    urllib.request.urlretrieve(URL, GZ_PATH)
    with gzip.open(GZ_PATH, "rb") as f_in, open(CSV_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(GZ_PATH)
    print(f"Saved: {CSV_PATH}")

import pandas as pd
df = pd.read_csv(CSV_PATH)
print(f"Rows: {len(df):,} | Columns: {list(df.columns)}")
