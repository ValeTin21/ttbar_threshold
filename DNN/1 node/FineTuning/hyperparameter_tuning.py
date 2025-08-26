# Hyperparameter Tuning for DNN Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score, roc_curve, auc
from itertools import product
import time
FilePath='filepath'

# Function to rescale weights separately for signal and background within a dataset
def rescale_weights_by_class(weights, labels):
    """
    Rescale weights separately for signal (0) and background (1) classes
    so that each class has weights summing to 1.0
    """
    rescaled_weights =np.zeros(len(weights), dtype=np.float32)
    
    # Create masks for signal (0) and background (1) classes
    signal_mask = (labels == 0)
    background_mask = (labels == 1)
    
    # Calculate sum of weights for each class
    signal_weight_sum = weights[signal_mask].sum()
    background_weight_sum = weights[background_mask].sum()
    
    # Rescale weights: weight/sum(weights) for each class separately
    if signal_weight_sum > 0:
        rescaled_weights[signal_mask] = np.array(weights[signal_mask] / signal_weight_sum,dtype=np.float32)
    if background_weight_sum > 0:
        rescaled_weights[background_mask] = np.array(weights[background_mask] / background_weight_sum,dtype=np.float32)

    return rescaled_weights, signal_weight_sum, background_weight_sum


def build_dnn_model(input_shape, hidden_units, dropout_rate=0.2):
    """
    Build DNN model with specified architecture
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_shape,)))
    model.add(layers.BatchNormalization())
    
    for units in hidden_units:
        model.add(layers.Dense(units, 
                              activation='relu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    # Binary output for AUC computation
    model.add(layers.Dense(1, activation='sigmoid', name='predictions'))
    return model


def hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, train_weights, val_weights, test_weights):
    """
    Comprehensive hyperparameter tuning
    """
    #Define hyperparameter space
    hidden_units_options = [[32, 32], [64, 32], [64, 64], [128, 64], [128, 64, 32], [256, 128, 64],
                            [256, 128, 64,32],[256, 128, 64,32,16]]
    dropout_options = [0.1, 0.2, 0.3]
    learning_rate_options = [1e-3, 1e-4, 1e-5]
    batch_size_options = [ 32,64, 128,256]
       
    # Generate all combinations
    all_combinations = list(product(
        hidden_units_options, dropout_options, 
        learning_rate_options, batch_size_options
    ))
    
    print(f"Starting tuning - {len(all_combinations)} configurations to test")
    
    results = []
    best_auc = 0
    best_config = None
    best_model = None
    
    input_shape = X_train.shape[1]
    
    for i, (hidden_units, dropout_rate, learning_rate, batch_size) in enumerate(all_combinations):
        try:
            # Build and compile model
            model = build_dnn_model(input_shape, hidden_units, dropout_rate)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'],
                weighted_metrics=['accuracy']
            )
            
            # Train model for 20 epochs
            model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=batch_size,
                validation_data=(X_val, y_val, val_weights),
                sample_weight=train_weights,
                verbose=0
            )
            
            # Get predictions and compute AUC on test set
            test_predictions = model.predict(X_test, verbose=0)
            test_auc = roc_auc_score(y_test, test_predictions, sample_weight=test_weights)
            
            # Store results
            config = {
                'hidden_units': hidden_units,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'test_auc': test_auc,
                'epochs_trained': 10
            }
            results.append(config)
            
            # Update best model
            if test_auc > best_auc:
                best_auc = test_auc
                best_config = config.copy()
                best_model = model

        except Exception as e:
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return best_model, best_config, results_df

def plot_tuning_results(results_df):
    """
    Visualize hyperparameter tuning results
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
    
    # AUC vs Learning Rate
    lr_auc = results_df.groupby('learning_rate')['test_auc'].agg(['mean', 'std', 'max'])
    axes[0, 0].errorbar(lr_auc.index, lr_auc['mean'], yerr=lr_auc['std'], 
                       marker='o', capsize=5, capthick=2)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Test AUC')
    axes[0, 0].set_title('AUC vs Learning Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC vs Dropout Rate
    dropout_auc = results_df.groupby('dropout_rate')['test_auc'].agg(['mean', 'std', 'max'])
    axes[0, 1].errorbar(dropout_auc.index, dropout_auc['mean'], yerr=dropout_auc['std'],
                       marker='o', capsize=5, capthick=2)
    axes[0, 1].set_xlabel('Dropout Rate')
    axes[0, 1].set_ylabel('Test AUC')
    axes[0, 1].set_title('AUC vs Dropout Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC vs Batch Size
    batch_auc = results_df.groupby('batch_size')['test_auc'].agg(['mean', 'std', 'max'])
    axes[1, 0].errorbar(batch_auc.index, batch_auc['mean'], yerr=batch_auc['std'],
                       marker='o', capsize=5, capthick=2)
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Test AUC')
    axes[1, 0].set_title('AUC vs Batch Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Top 10 configurations
    top_configs = results_df.nlargest(10, 'test_auc')
    axes[1, 1].barh(range(len(top_configs)), top_configs['test_auc'])
    axes[1, 1].set_yticks(range(len(top_configs)))
    axes[1, 1].set_yticklabels([f"Config {i+1}" for i in range(len(top_configs))])
    axes[1, 1].set_xlabel('Test AUC')
    axes[1, 1].set_title('Top 10 Configurations')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FilePath+'Plots/DNN1node/hyperparameter_tuning_results.png')
    plt.close()
    
    return top_configs

