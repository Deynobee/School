###############################################################################
## Purpose        :    Make pipelines and define early detection for Cognitive
## Input.         :    Processed data in excel format from ADNI
## Date           :    19/03/2026
## Authors        :    Dennis
## Email          :
## #################################################################

##Libraries
import os
import random
import joblib
import cv2
import shap
import pydicom
from sklearn.impute import SimpleImputer
from collections import Counter
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from tabulate import tabulate
from lime import lime_image
from scipy.ndimage import label as nd_label
from skimage.transform import resize as sk_resize
from skimage.segmentation import slic
from matplotlib.colors import LinearSegmentedColormap
from pydicom import dcmread
#from tf_explain.core.grad_cam import GradCAM
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras.backend as K
from keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Model as KerasModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE


# 1. Load the clinical data with baseline,CDR,Neuropath, and demography datasets
##################################################
# Natural Continum CN → SMC → EMCI → LMCI → AD
# full spectrum : CN, SMC, EMCI, LMCI, AD
# Progression Modelling : CN → EMCI → LMCI → AD
# Early Detection ; CN vs. EMCI vs. LMCI
# [CN vs SMC+EMCI], [ CN vs EMCI+LMCI] ,[CN vs MCI]
###################################################

print("1. Loading and cleaning data...")

# 1. Load Excel files
df1 = pd.read_excel("/Users/dennisboateng/Documents/GitHub/School/Alzy/Cognitive/Input/All_subjects.xlsx")
df2 = pd.read_excel("/Users/dennisboateng/Documents/GitHub/School/Alzy/Cognitive/Input/CDR.xlsx")
df3 = pd.read_excel("/Users/dennisboateng/Documents/GitHub/School/Alzy/Cognitive/Input/NEUROPATH.xlsx")
df4 = pd.read_excel("/Users/dennisboateng/Documents/GitHub/School/Alzy/Cognitive/Input/PTDEMOG.xlsx")
df5 = pd.read_excel("/Users/dennisboateng/Documents/GitHub/School/Alzy/Cognitive/Input/MMSE.xlsx")
df6 = pd.read_excel("/Users/dennisboateng/Documents/GitHub/School/Alzy/Cognitive/Input/MOCA.xlsx")

def safe_merge(left, right):
    overlap = [c for c in right.columns if c in left.columns and c != "PTID"]
    right = right.drop(columns=overlap)
    return left.merge(right, on="PTID", how="inner")

merged = df1
for df in [df2, df3, df4, df5, df6]:
    merged = safe_merge(merged, df)

print("Merged shape:", merged.shape)

merged = merged[merged['VISCODE'].isin(['sc', 'f'])]

print("\nUnique values in entry_outcomegrp:")
print(merged['entry_outcomegrp'].unique())

print("\nCounts of each group:")
print(merged['entry_outcomegrp'].value_counts(dropna=False))

# --- Target creation ---
merged['CDR_outcome'] = merged['entry_outcomegrp'].replace({
    'CN': 'CN',
    'MCI': 'MCI',
    'EMCI': 'MCI',
    'LMCI': 'MCI',
    'SMC': 'MCI',
    'AD': 'AD'
})

print("\nCounts of each group:")
print(merged['CDR_outcome'].value_counts(dropna=False))

merged['CDR_grp'] = merged['CDR_outcome'].map({'CN': 0, 'MCI': 1, 'AD': 2})
merged = merged.dropna(subset=['CDR_grp'])

# --- Intended predictors only (NO automatic numeric sweep) ---
features = ['PTGENDER', 'PTHAND', 'PTEDUCAT', 'PTMARRY', 'MMSCORE']

merged['Age_Group'] = pd.cut(
    merged['entry_age'],
    bins=[0, 60, 70, 80, 100],
    labels=['<60', '60-70', '70-80', '>80']
)

merged['MMSE_Group'] = pd.cut(
    merged['MMSCORE'],
    bins=[0, 21, 24, 28, 30],
    labels=['Severe', 'Mild', 'Normal_Low', 'Normal_High']
)

extended_features = features + ['Age_Group', 'MMSE_Group']

# Drop rows missing in predictors + target
merged = merged.dropna(subset=extended_features + ['CDR_grp']).copy()

# Encode categorical columns
categorical_cols = ['PTGENDER', 'PTHAND', 'PTMARRY', 'Age_Group', 'MMSE_Group']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    merged[col] = le.fit_transform(merged[col].astype(str))
    le_dict[col] = le

# Build X and y using ONLY intended predictors
X = merged[extended_features].values
y = merged['CDR_grp'].astype(int).values

print("\nFinal modeling columns (NO leakage):", extended_features)
print("Shape X:", X.shape, "Shape y:", y.shape)
print("Class distribution:", Counter(y))
print("Any NaNs left? ->", np.isnan(X).any())

# ----------------------------
# Cross-validation setup
# ----------------------------
class_counts = Counter(y)
min_class_count = min(class_counts.values())
n_splits = min(5, min_class_count)
print(f"\nMinimum class has {min_class_count} samples; using {n_splits}-fold CV")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ----------------------------
# Define models
# ----------------------------
def make_smote(k):
    return SMOTE(random_state=42, k_neighbors=k)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42, class_weight='balanced'
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42
    ),
    "SVM": SVC(
        kernel='rbf', C=10, gamma='scale', probability=True,
        class_weight='balanced', random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05
    )
}

# Store fold results
results = {name: {"acc": [], "bacc": [], "f1_macro": [], "auc_ovr": []} for name in models.keys()}

# ----------------------------
# ✅ Confusion-matrix storage (summed across folds)
# ----------------------------
labels = [0, 1, 2]  # fixed order: CN, MCI, AD
label_names = ["CN", "MCI", "AD"]

cm_total = {name: np.zeros((len(labels), len(labels)), dtype=int) for name in models.keys()}

# OPTIONAL: store per-fold confusion matrices if you want later
cm_per_fold = {name: [] for name in models.keys()}

# If True, prints confusion matrix + report for each fold (can be a lot of output)
PRINT_PER_FOLD_CM = True

# ----------------------------
# CV loop
# ----------------------------
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n=== Fold {fold}/{n_splits} ===")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Dynamic SMOTE k_neighbors: must be < min_class_count in training
    train_counts = Counter(y_train)
    min_train_class = min(train_counts.values())
    if min_train_class < 2:
        print("Skipping fold: minority class <2 samples in training.")
        continue
    k_neighbors = min(5, min_train_class - 1)

    for name, clf in models.items():

        # Scale BEFORE SMOTE (SMOTE uses distances)
        pipe = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("smote", make_smote(k_neighbors)),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        # ----------------------------
        # ✅ Confusion metrics (per fold + accumulated)
        # ----------------------------
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_total[name] += cm
        cm_per_fold[name].append(cm)

        if PRINT_PER_FOLD_CM:
            print(f"\n{name} | Confusion Matrix (Fold {fold}) [rows=true, cols=pred] (CN/MCI/AD):")
            print(cm)

            print(f"\n{name} | Classification Report (Fold {fold}):")
            print(classification_report(
                y_test, y_pred,
                labels=labels,
                target_names=label_names,
                digits=3,
                zero_division=0
            ))

        # Proba for AUC if available
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        else:
            auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")

        results[name]["acc"].append(acc)
        results[name]["bacc"].append(bacc)
        results[name]["f1_macro"].append(f1m)
        results[name]["auc_ovr"].append(auc)

        print(f"{name:16s} | acc={acc:.3f} | bacc={bacc:.3f} | f1_macro={f1m:.3f} | auc_ovr={auc:.3f}")

# ----------------------------
# Summary: mean ± sd
# ----------------------------
print("\n\n==================== FINAL CV SUMMARY (mean ± sd) ====================")
for name in results:
    acc_m, acc_s = np.mean(results[name]["acc"]), np.std(results[name]["acc"])
    b_m, b_s     = np.mean(results[name]["bacc"]), np.std(results[name]["bacc"])
    f1_m, f1_s   = np.mean(results[name]["f1_macro"]), np.std(results[name]["f1_macro"])
    auc_vals = [v for v in results[name]["auc_ovr"] if not np.isnan(v)]
    auc_m, auc_s = (np.mean(auc_vals), np.std(auc_vals)) if len(auc_vals) else (np.nan, np.nan)

    print(f"{name:16s} | "
          f"ACC {acc_m:.3f}±{acc_s:.3f} | "
          f"BACC {b_m:.3f}±{b_s:.3f} | "
          f"F1 {f1_m:.3f}±{f1_s:.3f} | "
          f"AUC {auc_m:.3f}±{auc_s:.3f}")

# ----------------------------
# ✅ Confusion matrix summary (summed + normalized)
# ----------------------------
print("\n\n==================== CONFUSION MATRICES (Summed Across CV Folds) ====================")
for name in models.keys():
    print(f"\n{name} - Total confusion matrix [rows=true, cols=pred] (CN/MCI/AD):")
    print(cm_total[name])

    # Row-normalized confusion matrix = per-class recall pattern
    row_sums = cm_total[name].sum(axis=1, keepdims=True)

    # Avoid divide by zero (just in case a class never appears in test folds)
    cm_norm = np.divide(cm_total[name], row_sums, where=row_sums != 0)

    print(f"\n{name} - Row-normalized confusion matrix (per-class recall):")
    print(np.round(cm_norm, 3))

print("\nDone.")