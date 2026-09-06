"""One-off script to train and save models (optional)."""

from ml_pipeline import _resolve_data_dir, train_and_save

if __name__ == "__main__":
    data_dir = _resolve_data_dir()
    print(f"Using data from: {data_dir}")
    train_and_save(data_dir)
    print("Saved models to ./models/")
