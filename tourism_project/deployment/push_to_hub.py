"""
Hosting Script — pushes app.py, requirements.txt, Dockerfile to HF Space.
"""
import os
from huggingface_hub import HfApi, login

HF_TOKEN   = os.environ.get('HF_TOKEN')
HF_USERNAME = 'natkhatgopaljee'
SPACE_REPO  = f'{HF_USERNAME}/tourism-wellness-app'

login(token=HF_TOKEN)
api = HfApi()
api.create_repo(repo_id=SPACE_REPO, repo_type='space', space_sdk='docker', exist_ok=True, private=False)

for local, remote in [
    ('tourism_project/deployment/app.py',           'app.py'),
    ('tourism_project/deployment/requirements.txt', 'requirements.txt'),
    ('tourism_project/deployment/Dockerfile',       'Dockerfile'),
]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=SPACE_REPO, repo_type='space')
    print(f'Uploaded {remote}')

print(f'App URL: https://huggingface.co/spaces/{SPACE_REPO}')
