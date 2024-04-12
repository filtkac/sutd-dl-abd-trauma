import torch
from huggingface_hub import HfApi, HfFolder, Repository


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
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
