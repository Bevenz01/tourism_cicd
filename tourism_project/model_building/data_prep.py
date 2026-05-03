"""
Step 2 - Data Preparation
Loads tourism.csv from HF Dataset Hub, cleans it, encodes categoricals,
splits into train/test, and uploads splits + encoders back to HF.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download, HfApi, login

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_USERNAME = 'natkhatgopaljee'
DATASET_REPO = f'{HF_USERNAME}/tourism-dataset'

login(token=HF_TOKEN)
api = HfApi()

# Load dataset from Hugging Face
print('Loading dataset from Hugging Face...')
file_path = hf_hub_download(repo_id=DATASET_REPO, filename='tourism.csv', repo_type='dataset')
df = pd.read_csv(file_path, index_col=0)
print(f'Loaded shape: {df.shape}')
print(f'Null counts:\n{df.isnull().sum()}')

# Drop unnecessary columns
df.drop(columns=['CustomerID'], inplace=True)

# Fill numeric nulls with median
numeric_cols = ['Age','DurationOfPitch','NumberOfFollowups','MonthlyIncome',
                'NumberOfTrips','NumberOfChildrenVisiting','PreferredPropertyStar']
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical columns
cat_cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

print(f'After cleaning shape: {df.shape}')
print(f'Target distribution:\n{df["ProdTaken"].value_counts()}')

# Train / test split
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)

os.makedirs('tourism_project/data', exist_ok=True)
train_df.to_csv('tourism_project/data/train.csv', index=False)
test_df.to_csv('tourism_project/data/test.csv',   index=False)
joblib.dump(encoders, 'tourism_project/data/encoders.pkl')

print(f'Train shape: {train_df.shape}')
print(f'Test  shape: {test_df.shape}')

# Upload splits + encoders to HF
for local, remote in [
    ('tourism_project/data/train.csv',    'train.csv'),
    ('tourism_project/data/test.csv',     'test.csv'),
    ('tourism_project/data/encoders.pkl', 'encoders.pkl'),
]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=DATASET_REPO, repo_type='dataset')
    print(f'Uploaded {remote} to {DATASET_REPO}')

print('Data preparation complete.')
