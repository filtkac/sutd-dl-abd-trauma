import torch
from huggingface_hub import HfApi, HfFolder, Repository


# ----- Data Util Functions ----- #


def class_weights(data):
    """Creating class weights for loss function to address class imbalances"""

    weights = {
        "bowel": data["bowel"].value_counts()[0] / data["bowel"].value_counts()[1],
        "extravastion": data["extravastion"].value_counts()[0] / data["extravastion"].value_counts()[1],
        "kidney": data["kidney"].value_counts()[0] / data["kidney"].value_counts()[1],
        "liver": data["liver"].value_counts()[0] / data["liver"].value_counts()[1],
        "spleen": data["spleen"].value_counts()[0] / data["spleen"].value_counts()[1],
    }

    total = sum(weights.values())
    normalized = {key: round(value / total, 5) for key, value in weights.items()}
    return normalized


# ----- Model Util Functions ----- #


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")


def upload_models_hf(model_dir, user, repo_name):
    repo_id = f"{user}/{repo_name}"

    # Authenticate with Hugging Face
    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise ValueError("You must be logged in to Hugging Face. Use `huggingface-cli login`.")

    # Create a new repository on Hugging Face
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False)

    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Model successfully uploaded to {repo_id}")
