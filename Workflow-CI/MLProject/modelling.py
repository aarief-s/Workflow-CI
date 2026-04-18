"""
modelling.py  (MLProject version)
==================================
Script training untuk MLflow Project + GitHub Actions CI.
Mendukung environment variable dan argumen CLI untuk konfigurasi fleksibel.

Penulis : Lufthi Arief Syabana
Dataset : Smartphone Usage and Addiction Analysis
Task    : Binary Classification — addicted_label (0/1)
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report,
    roc_curve, average_precision_score, precision_recall_curve
)
import joblib

# ─── Argumen CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train RF model — Lufthi Arief Syabana')
parser.add_argument('--train_path',      type=str,   default='smartphone_usage_preprocessing/train.csv')
parser.add_argument('--test_path',       type=str,   default='smartphone_usage_preprocessing/test.csv')
parser.add_argument('--n_estimators',    type=int,   default=100)
parser.add_argument('--max_depth',       type=int,   default=10)
parser.add_argument('--min_samples_split', type=int, default=2)
parser.add_argument('--max_features',    type=str,   default='sqrt')
parser.add_argument('--random_state',    type=int,   default=42)
args = parser.parse_args()

TARGET_COL = 'addicted_label'

# ─── Setup MLflow Tracking ─────────────────────────────────────────────────────
# Mendukung DagsHub via environment variable MLFLOW_TRACKING_URI
tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'mlruns')
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment('Smartphone_Addiction_CI')

print("=" * 55)
print("  CI TRAINING — Lufthi Arief Syabana")
print(f"  MLflow Tracking: {tracking_uri}")
print("=" * 55)

# ─── Load Data ────────────────────────────────────────────────────────────────
df_train = pd.read_csv(args.train_path)
df_test  = pd.read_csv(args.test_path)

X_train = df_train.drop(columns=[TARGET_COL])
y_train = df_train[TARGET_COL]
X_test  = df_test.drop(columns=[TARGET_COL])
y_test  = df_test[TARGET_COL]

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ─── Training dengan MLflow Manual Logging ─────────────────────────────────────
os.makedirs('artifacts_ci', exist_ok=True)

with mlflow.start_run(run_name='CI_RandomForest'):

    # ── Params ──
    params = {
        'n_estimators':     args.n_estimators,
        'max_depth':        args.max_depth,
        'min_samples_split': args.min_samples_split,
        'max_features':     args.max_features,
        'random_state':     args.random_state,
    }
    mlflow.log_params(params)
    mlflow.log_param('train_samples', len(X_train))
    mlflow.log_param('test_samples',  len(X_test))
    mlflow.log_param('n_features',    X_train.shape[1])

    # ── Fit ──
    model = RandomForestClassifier(**params, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_prob       = model.predict_proba(X_test)[:, 1]
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]

    # ── Metrics ──
    metrics = {
        'test_accuracy':          accuracy_score(y_test, y_pred),
        'test_precision':         precision_score(y_test, y_pred, zero_division=0),
        'test_recall':            recall_score(y_test, y_pred, zero_division=0),
        'test_f1_score':          f1_score(y_test, y_pred, zero_division=0),
        'test_roc_auc':           roc_auc_score(y_test, y_prob),
        'test_log_loss':          log_loss(y_test, y_prob),
        'test_average_precision': average_precision_score(y_test, y_prob),
        'train_accuracy':         accuracy_score(y_train, y_train_pred),
        'train_f1_score':         f1_score(y_train, y_train_pred, zero_division=0),
        'train_roc_auc':          roc_auc_score(y_train, y_train_prob),
    }
    metrics['overfit_gap_accuracy'] = metrics['train_accuracy'] - metrics['test_accuracy']
    metrics['overfit_gap_f1']       = metrics['train_f1_score'] - metrics['test_f1_score']
    mlflow.log_metrics(metrics)

    # ── Artefak 1: Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Tidak Adiksi','Adiksi'],
                yticklabels=['Tidak Adiksi','Adiksi'])
    ax.set_title('Confusion Matrix — CI Run', fontweight='bold')
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
    plt.tight_layout()
    cm_path = 'artifacts_ci/confusion_matrix.png'
    plt.savefig(cm_path, dpi=110, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path='plots')

    # ── Artefak 2: ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f"AUC = {metrics['test_roc_auc']:.4f}")
    ax.plot([0,1],[0,1],'--', color='navy', lw=1.5)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve — CI Run', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    roc_path = 'artifacts_ci/roc_curve.png'
    plt.savefig(roc_path, dpi=110, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(roc_path, artifact_path='plots')

    # ── Artefak 3: Feature Importance ──
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='gray')
    ax.set_title('Feature Importance — CI Run', fontweight='bold')
    plt.tight_layout()
    fi_path = 'artifacts_ci/feature_importance.png'
    plt.savefig(fi_path, dpi=110, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(fi_path, artifact_path='plots')

    # ── Artefak 4: Classification Report JSON ──
    cr = classification_report(y_test, y_pred,
         target_names=['Tidak Adiksi','Adiksi'], output_dict=True)
    cr_path = 'artifacts_ci/classification_report.json'
    with open(cr_path, 'w') as f:
        json.dump(cr, f, indent=2)
    mlflow.log_artifact(cr_path, artifact_path='reports')

    # ── Log Model ──
    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(model, artifact_path='model',
                             signature=signature,
                             input_example=X_test.iloc[:3])

    # ── Simpan model lokal (untuk upload ke storage) ──
    os.makedirs('model_output', exist_ok=True)
    model_pkl = 'model_output/model.pkl'
    joblib.dump(model, model_pkl)
    mlflow.log_artifact(model_pkl, artifact_path='model_pkl')

    run_id = mlflow.active_run().info.run_id
    print(f"\n  Run ID  : {run_id}")
    print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  F1-Score: {metrics['test_f1_score']:.4f}")
    print(f"  ROC-AUC : {metrics['test_roc_auc']:.4f}")

print("\n✅ CI Training selesai!")

# Simpan run_id untuk digunakan oleh step berikutnya di GitHub Actions
with open('model_output/run_id.txt', 'w') as f:
    f.write(run_id)
print(f"   Run ID disimpan ke: model_output/run_id.txt")
