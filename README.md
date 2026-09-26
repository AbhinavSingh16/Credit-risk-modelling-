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


PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) (45 mins)
Why this matters for interview:

"Conduct thorough understanding of the data science lifecycle including data exploration..." — that's straight from the JD. Show them you explore before you model.

Step 2.1: Class Imbalance Analysis
python
# Check class distribution
print("Class distribution:")
print(df['Class'].value_counts())
print("\nClass proportions:")
print(df['Class'].value_counts(normalize=True))

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Countplot
df['Class'].value_counts().plot(kind='bar', ax=ax[0], color=['green', 'red'])
ax[0].set_title('Class Distribution (Count)', fontsize=12, fontweight='bold')
ax[0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
ax[0].set_ylabel('Count')
ax[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)

# Pie chart
df['Class'].value_counts().plot(kind='pie', ax=ax[1], labels=['Legitimate', 'Fraud'], 
                                 colors=['green', 'red'], autopct='%1.2f%%')
ax[1].set_title('Class Proportion (%)', fontsize=12, fontweight='bold')
ax[1].set_ylabel('')

plt.tight_layout()
plt.show()

# Key insight to remember for interview:
fraud_pct = (df['Class'].sum() / len(df)) * 100
print(f"\n⚠️  CRITICAL: Only {fraud_pct:.2f}% of transactions are fraud!")
print("This is why accuracy alone is misleading. A model that predicts 'never fraud' would be 99.8% accurate but worthless.")

Interview talking point: "This dataset has extreme class imbalance. If I just predict everything as 'not fraud,' I'd get 99.8% accuracy but catch zero fraud. That's why we use Precision-Recall curves, not accuracy, and techniques like SMOTE to balance the data during training."

Step 2.2: Transaction Amount Analysis
python
# Amount statistics
print("Transaction Amount Statistics:")
print(df['Amount'].describe())
print("\nFraud vs Legitimate Transaction Amounts:")
print(df.groupby('Class')['Amount'].describe())

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
df.boxplot(column='Amount', by='Class', ax=ax[0])
ax[0].set_title('Transaction Amount by Class', fontsize=12, fontweight='bold')
ax[0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
ax[0].set_ylabel('Amount ($)')
ax[0].get_figure().suptitle('')  # Remove automatic title

# Histogram
for class_val in [0, 1]:
    data = df[df['Class'] == class_val]['Amount']
    label = 'Legitimate' if class_val == 0 else 'Fraud'
    ax[1].hist(data, bins=50, alpha=0.6, label=label)

ax[1].set_xlabel('Amount ($)')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Distribution of Transaction Amounts', fontsize=12, fontweight='bold')
ax[1].legend()
ax[1].set_xlim(0, 500)  # Focus on bulk of data

plt.tight_layout()
plt.show()

print("\n💡 Insight: Fraud transactions tend to have different amount distributions than legitimate ones.")
