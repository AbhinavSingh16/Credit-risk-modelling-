📋 PHASE 1: SETUP & IMPORT (15 MINUTES)
Copy this into a NEW cell and run it:
python
# Install libraries
!pip install kaggle pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn -q

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score
import warnings
warnings.filterwarnings('ignore')

print("✅ ALL LIBRARIES INSTALLED!")


PHASE 2: DOWNLOAD DATA 
In python
# Setup Kaggle API
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d mlg-ulb/creditcardfraud -p ./data --unzip

print("✅ DATA DOWNLOADED!")
