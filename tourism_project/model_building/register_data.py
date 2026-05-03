"""
Step 1 - Data Registration
Uploads the raw tourism.csv to Hugging Face Dataset Hub.
"""
import os
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_USERNAME = 'natkhatgopaljee'
DATASET_REPO = f'{HF_USERNAME}/tourism-dataset'
DATA_PATH = 'tourism.csv'  # root of repo — data folder is gitignored

login(token=HF_TOKEN)
api = HfApi()

# Create dataset repo (public)
api.create_repo(repo_id=DATASET_REPO, repo_type='dataset', exist_ok=True, private=False)

# Upload raw CSV
api.upload_file(
    path_or_fileobj=DATA_PATH,
    path_in_repo='tourism.csv',
    repo_id=DATASET_REPO,
    repo_type='dataset',
)
print(f'Dataset registered at: https://huggingface.co/datasets/{DATASET_REPO}')
