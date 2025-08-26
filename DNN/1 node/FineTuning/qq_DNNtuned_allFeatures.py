# Implementing a Binary DNN Classifier for Signal vs Background Classification
# gg vs qq channel | all features

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from pathlib import Path
import sys
import pickle
import mplhep as hep
plt.style.use(hep.style.ATLAS)
FilePath='/data/dust/user/vtinari/'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
import keras
from tensorflow.keras import layers, models

#Training functions
import hyperparameter_tuning as HT

print("✅ All libraries imported successfully")
######################################

# Load DataFrame from pickle file
start_time = time.time()
df = pd.read_pickle(FilePath+'df_3classes.pkl')

# Load the weight array from the saved numpy file
weightarr = np.load(FilePath+'weightarr.npy')

# Add weights to dataframe
df['weights_MC_NOSYS']=weightarr

# Filter positive weights
mask = df['weights_MC_NOSYS'] > 0
df_cut = df[mask]

# Selecting columns associated to features
features = list(df_cut.columns[:13])

# Creating a new df only with gg and qq
mask_qq = (df_cut['prod_type'] == 0) | (df_cut['prod_type'] == 2)
df_qq=df_cut[mask_qq]
df_qq['prod_type'][df_qq['prod_type']==2]=1


# Normalizing features with StandardScaler for better scaling
df_norm = df_qq.copy()
scaler = StandardScaler()
df_norm[features] = scaler.fit_transform(df_norm[features])

# Splitting the dataset into proper components for training, validation and testing (60:20:20)
X_train_pd, X_tv, Y_train_pd, y_tv,W_train_pd,w_tv = train_test_split(df_norm[features], df_norm['prod_type'],df_norm['weights_MC_NOSYS'], train_size=0.6, shuffle=True, random_state=1234,stratify=df_norm['prod_type'])
X_test_pd, X_vali_pd, Y_test_pd, Y_vali_pd, W_test_pd,W_vali_pd = train_test_split(X_tv, y_tv,w_tv, train_size=0.5, shuffle=True, random_state=1234,stratify=y_tv)
print("Len of test set: ",len(X_test_pd), file=sys.stderr)

# Rescale weights for all datasets
train_weights, _, _ = HT.rescale_weights_by_class(W_train_pd, np.array(Y_train_pd))
vali_weights, _, _ = HT.rescale_weights_by_class(W_vali_pd, np.array(Y_vali_pd))
test_weights, _, _ = HT.rescale_weights_by_class(W_test_pd, np.array(Y_test_pd))

# Convert to numpy arrays with proper dtypes
X_train = np.array(X_train_pd, dtype=np.float32)
X_vali = np.array(X_vali_pd, dtype=np.float32)
X_test = np.array(X_test_pd, dtype=np.float32)

Y_train = np.array(Y_train_pd, dtype=np.float32)
Y_vali = np.array(Y_vali_pd, dtype=np.float32)
Y_test = np.array(Y_test_pd, dtype=np.float32)

# Print dataset information before hyperparameter tuning
print(f"Dataset shapes - X_train: {X_train.shape}, X_vali: {X_vali.shape}, X_test: {X_test.shape}")
print(f"Label shapes - Y_train: {Y_train.shape}, Y_vali: {Y_vali.shape}, Y_test: {Y_test.shape}")
print(f"Weight shapes - train_weights: {train_weights.shape}, vali_weights: {vali_weights.shape}, test_weights: {test_weights.shape}")

# Run hyperparameter tuning
start_time = time.time()

print("\n################ Starting the training ################", file=sys.stderr)
best_model, best_config, tuning_results = HT.hyperparameter_tuning(
    X_train, Y_train, X_vali, Y_vali, X_test, Y_test, train_weights, vali_weights, test_weights
)

tuning_time = time.time() - start_time
print('\n################ Training completed in '+ str(tuning_time) + ' seconds ################', file=sys.stderr)

# Plot comprehensive results
top_configs = HT.plot_tuning_results(tuning_results)

# Final model evaluation with best configuration
# Get final predictions on test set
test_predictions = best_model.predict(X_test, verbose=0)
test_auc = roc_auc_score(Y_test, test_predictions, sample_weight=test_weights)

# Compute and plot ROC curve for best model
fpr, tpr, thresholds = roc_curve(Y_test, test_predictions, sample_weight=test_weights)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=3, label=f'ROC Curve (AUC = {test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', alpha=0.8, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve - Best Tuned Model', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(FilePath+'Plots/DNN1node/gg_qq_ROCcurve_best_model.png')
plt.close()

# FINAL SUMMARY
print("="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Total elapsed time: {tuning_time:.2f} seconds")
print(f"Best configuration found:")
print(f"  Architecture: {best_config['hidden_units']}")
print(f"  Dropout: {best_config['dropout_rate']}")
print(f"  Learning Rate: {best_config['learning_rate']}")
print(f"  Batch Size: {best_config['batch_size']}")
print(f"  Best AUC: {best_config['test_auc']:.4f}")
print("="*80)
