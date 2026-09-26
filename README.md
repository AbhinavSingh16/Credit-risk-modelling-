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

What you did: You told Colab to download all the tools you need.

If it fails: Run it again. Kaggle sometimes needs 2 tries.
