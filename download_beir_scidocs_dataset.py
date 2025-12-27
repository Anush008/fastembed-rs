# /// script
# requires-python = ">=3.10"
# dependencies = ["beir"]
# ///

from beir import util
from pathlib import Path

dataset = "scidocs"
out_dir = Path("datasets")

# Download + unzip
zip_path = out_dir / f"{dataset}.zip"
data_path = util.download_and_unzip(
    str(zip_path),
    str(out_dir)
)

# Validate structure exists
root = out_dir / dataset
assert (root / "corpus.jsonl").exists()
assert (root / "queries.jsonl").exists()
assert (root / "qrels" / "test.tsv").exists()

print("Done.")
print("Corpus:", root / "corpus.jsonl")
print("Queries:", root / "queries.jsonl")
print("Qrels:", root / "qrels" / "test.tsv")
