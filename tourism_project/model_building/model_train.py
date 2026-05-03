"""
Step 3 - Model Building with Experimentation Tracking
Loads train/test from HF, tunes 4 classifiers with GridSearchCV,
logs everything to MLflow, and registers the best model on HF Model Hub.
"""
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from huggingface_hub import hf_hub_download, HfApi, login

HF_TOKEN     = os.environ.get('HF_TOKEN')
HF_USERNAME  = 'natkhatgopaljee'
DATASET_REPO = f'{HF_USERNAME}/tourism-dataset'
MODEL_REPO   = f'{HF_USERNAME}/tourism-wellness-model'

login(token=HF_TOKEN)
api = HfApi()
api.create_repo(repo_id=MODEL_REPO, repo_type='model', exist_ok=True, private=False)

# Load data
print('Loading train/test data from Hugging Face...')
train_path = hf_hub_download(repo_id=DATASET_REPO, filename='train.csv', repo_type='dataset')
test_path  = hf_hub_download(repo_id=DATASET_REPO, filename='test.csv',  repo_type='dataset')
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
X_train = train_df.drop(columns=['ProdTaken'])
y_train = train_df['ProdTaken']
X_test  = test_df.drop(columns=['ProdTaken'])
y_test  = test_df['ProdTaken']
print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')

# MLflow
MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', 'mlruns')
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment('tourism_wellness_prediction')

models_params = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5]},
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {'n_estimators': [100, 200], 'max_depth': [5, 10]},
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, verbosity=0),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]},
    },
}

best_model = None
best_f1    = 0.0
best_name  = ''
results    = []

for name, config in models_params.items():
    print(f'\n--- Training {name} ---')
    with mlflow.start_run(run_name=name):
        gs = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        best_est = gs.best_estimator_
        y_pred = best_est.predict(X_test)
        y_prob = best_est.predict_proba(X_test)[:, 1]
        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'f1_score':  f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred),
            'roc_auc':   roc_auc_score(y_test, y_prob),
        }
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_est, artifact_path='model')
        print(f'  Best params : {gs.best_params_}')
        print(f'  Accuracy    : {metrics["accuracy"]:.4f}')
        print(f'  F1 Score    : {metrics["f1_score"]:.4f}')
        print(f'  ROC-AUC     : {metrics["roc_auc"]:.4f}')
        results.append({'model': name, **metrics})
        if metrics['f1_score'] > best_f1:
            best_f1   = metrics['f1_score']
            best_model = best_est
            best_name  = name

print('\n===== Model Comparison =====')
results_df = pd.DataFrame(results).set_index('model')
print(results_df.to_string())
print(f'\nBest model: {best_name}  (F1 = {best_f1:.4f})')

os.makedirs('tourism_project/model_building', exist_ok=True)
model_path   = 'tourism_project/model_building/best_model.pkl'
feature_path = 'tourism_project/model_building/feature_names.pkl'
joblib.dump(best_model, model_path)
joblib.dump(X_train.columns.tolist(), feature_path)

for local, remote in [(model_path,'best_model.pkl'), (feature_path,'feature_names.pkl')]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=MODEL_REPO, repo_type='model')
    print(f'Uploaded {remote} to {MODEL_REPO}')

print(f'Best model ({best_name}) registered at: https://huggingface.co/{MODEL_REPO}')
