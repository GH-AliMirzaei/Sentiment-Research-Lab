
from pathlib import Path
from datasets import load_dataset

# Load the original IMDB dataset
dataset = load_dataset("stanfordnlp/imdb")

# Split the official training data (80% / 20%)
split = dataset["train"].train_test_split(
    test_size=0.2,
    seed=42,
    stratify_by_column="label"
)

train_data = split["train"]
val_data = split["test"]
test_data = dataset["test"]

# Save datasets locally
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_data.save_to_disk(str(output_dir / "train"))
val_data.save_to_disk(str(output_dir / "validation"))
test_data.save_to_disk(str(output_dir / "test"))

print("IMDB dataset prepared successfully!")
print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Testing:", len(test_data))

print("Train label distribution:", train_data["label"].count(0),
      train_data["label"].count(1))
