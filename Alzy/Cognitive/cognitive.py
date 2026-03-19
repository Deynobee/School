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
import numpy as np
import pandas as pd
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score
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

# 2. Merge on PTID
merged = (
    df1.merge(df2, on="PTID", how="inner")
       .merge(df3, on="PTID", how="inner")
       .merge(df4, on="PTID", how="inner")
)

print(merged.head())
print(merged.shape)

# 3. Show all column names so we know the exact diagnosis column name
print("\nColumns in merged dataset:")
print(merged.columns.tolist())

# 4. Count the original groups EXACTLY as they appear
print("\nUnique values in entry_outcomegrp:")
print(merged['entry_outcomegrp'].unique())

print("\nCounts of each group:")
print(merged['entry_outcomegrp'].value_counts(dropna=False))

##Defining early setup Grp1
# CN vs (SMC + EMCI)
merged['group_early'] = merged['entry_outcomegrp'].replace({
    'CN': 'CN',
    'SMC': 'Early',
    'EMCI': 'Early'
})

# Everything else becomes NaN (not used in this grouping)
merged.loc[~merged['entry_outcomegrp'].isin(['CN', 'SMC', 'EMCI']), 'group_early'] = None

print("\nCounts of each group:")
print(merged['group_early'].value_counts(dropna=False))



#Grp2: CN vs (EMCI + LMCI)
merged['group_prodromal'] = merged['entry_outcomegrp'].replace({
    'CN': 'CN',
    'EMCI': 'Prodromal',
    'LMCI': 'Prodromal'
})

merged.loc[~merged['entry_outcomegrp'].isin(['CN', 'EMCI', 'LMCI']), 'group_prodromal'] = None

print("\nCounts of each group:")
print(merged['group_prodromal'].value_counts(dropna=False))

#Group3: CN vs MCI
merged['group_mci'] = merged['entry_outcomegrp'].replace({
    'CN': 'CN',
    'MCI': 'MCI'
})

merged.loc[~merged['entry_outcomegrp'].isin(['CN', 'MCI']), 'group_mci'] = None
print("\nCounts of each group:")
print(merged['group_mci'].value_counts(dropna=False))