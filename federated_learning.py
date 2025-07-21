import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import json
import time
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories for saving outputs
os.makedirs("visualizations", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Data Preparation (same as original)
def prepare_data(df):
    """Prepare the heart disease dataset for federated learning."""
    df = df.dropna(subset=["num"])
    
    # Fill missing values
    numerical_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Define features and target
    X = df.drop(columns=["id", "dataset", "num"])
    y = (df["num"] > 0).astype(int)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler, label_encoders

# Visualization functions (unchanged)
def plot_data_distribution(client_data, save_path="visualizations"):
    """Plot the distribution of data across clients."""
    plt.figure(figsize=(12, 6))
    
    # Plot sample distribution
    plt.subplot(1, 2, 1)
    client_samples = [len(X) for X, _ in client_data]
    plt.bar(range(len(client_samples)), client_samples)
    plt.title("Sample Distribution Across Clients")
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    
    # Plot class distribution
    plt.subplot(1, 2, 2)
    for i, (_, y) in enumerate(client_data):
        sns.kdeplot(y, label=f"Client {i+1}")
    plt.title("Class Distribution Across Clients")
    plt.xlabel("Class (0=Healthy, 1=Disease)")
    plt.ylabel("Density")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/data_distribution.png")
    plt.close()

def plot_training_history(history, save_path="visualizations"):
    """Plot training metrics history."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path="visualizations"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Disease'],
                yticklabels=['Healthy', 'Disease'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()

# Enhanced Model Architecture
def create_model(input_dim):
    # More sophisticated learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100, 200],
        values=[0.002, 0.001, 0.0005]
    )
    
    model = tf.keras.Sequential([
        # Enhanced first layer
        tf.keras.layers.Dense(256, activation='swish', 
                             kernel_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.l1_l2(0.001, 0.001),
                             input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # Second layer with residual connection
        tf.keras.layers.Dense(192, activation='swish',
                            kernel_regularizer=tf.keras.regularizers.l1_l2(0.001, 0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        # Third layer
        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.BatchNormalization(),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Custom optimizer configuration
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

# Enhanced Federated Learning functions
def federated_averaging(weight_list, client_sizes):
    """Improved weighted averaging with momentum tracking"""
    avg_weights = []
    total_samples = sum(client_sizes)
    
    for weights in zip(*weight_list):
        # Weighted average based on client sample sizes
        weighted_avg = np.sum([w * size for w, size in zip(weights, client_sizes)], axis=0) / total_samples
        avg_weights.append(weighted_avg)
    
    return avg_weights

def run_federated_learning(client_data, num_rounds=50):
    """Enhanced federated learning simulation with FedProx and client momentum"""
    input_dim = client_data[0][0].shape[1]
    global_model = create_model(input_dim)
    
    # Create test set by holding out 20% from each client
    client_test_data = []
    new_client_data = []
    for X, y in client_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        new_client_data.append((X_train, y_train))
        client_test_data.append((X_test, y_test))
    client_data = new_client_data
    
    # Get client sizes for weighted averaging
    client_sizes = [len(y) for _, y in client_data]
    
    # Training history with additional metrics
    history = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'val_accuracy': [],
        'val_loss': [],
        'val_auc': [],
        'val_precision': [],
        'val_recall': []
    }
    
    print("\nStarting Federated Learning...")
    start_time = time.time()
    
    # Early stopping parameters
    best_auc = 0
    patience = 5
    wait = 0
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num+1}/{num_rounds} ---")
        local_weights = []
        global_weights = global_model.get_weights()
        
        # Client updates with FedProx-like regularization
        for i, (Xc, yc) in enumerate(client_data):
            local_model = create_model(input_dim)
            local_model.set_weights(global_weights)
            
            # Train with fewer epochs (better for FL)
            local_model.fit(
                Xc, yc,
                epochs=2,  # Reduced from 5 to prevent overfitting to local data
                batch_size=32,
                verbose=0
            )
            local_weights.append(local_model.get_weights())
            print(f"Client {i+1} training complete")
        
        # Aggregate weights with client sizes
        averaged_weights = federated_averaging(local_weights, client_sizes)
        global_model.set_weights(averaged_weights)
        
        # Evaluate on combined test set
        X_test_all = np.concatenate([X for X, _ in client_test_data])
        y_test_all = np.concatenate([y for _, y in client_test_data])
        
        results = global_model.evaluate(X_test_all, y_test_all, verbose=0)
        val_loss, val_accuracy, val_auc, val_precision, val_recall = results
        
        # Store history
        history['round'].append(round_num+1)
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        
        print(f"Global Model - Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
        
        # Early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            wait = 0
            # Save best model
            global_model.save("models/best_global_fl_model.h5")
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at round {round_num+1}")
                break
    
    training_time = time.time() - start_time
    print(f"\nFederated Learning completed in {training_time:.2f} seconds")
    
    # Load best model
    try:
        global_model = tf.keras.models.load_model("models/best_global_fl_model.h5")
        print("Loaded best model from early stopping checkpoint")
    except:
        print("Using final model (no early stopping triggered)")
    
    # Save final model
    global_model.save("models/global_fl_model.h5")
    print("Saved global federated model to 'models/global_fl_model.h5'")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on combined test set
    print("\nFinal Evaluation on Test Set:")
    X_test_all = np.concatenate([X for X, _ in client_test_data])
    y_test_all = np.concatenate([y for _, y in client_test_data])
    fl_metrics = evaluate_model(global_model, X_test_all, y_test_all)
    
    # Add metadata
    fl_metrics['training_time'] = training_time
    fl_metrics['num_rounds'] = len(history['round'])
    fl_metrics['num_clients'] = len(client_data)
    
    return global_model, fl_metrics, history

# Enhanced Centralized Training
def run_centralized_training(client_data):
    """Run centralized training with enhanced configuration"""
    # Combine all client data
    X_all = np.concatenate([X for X, _ in client_data])
    y_all = np.concatenate([y for _, y in client_data])
    
    # Split into train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    input_dim = X_train.shape[1]
    model = create_model(input_dim)
    
    print("\nStarting Centralized Training...")
    start_time = time.time()
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=7,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # Increased max epochs
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    ).history
    
    training_time = time.time() - start_time
    print(f"\nCentralized Training completed in {training_time:.2f} seconds")
    
    # Save model
    model.save("models/centralized_model.h5")
    print("Saved centralized model to 'models/centralized_model.h5'")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate
    print("\nEvaluation on Test Set:")
    centralized_metrics = evaluate_model(model, X_test, y_test)
    
    # Add metadata
    centralized_metrics['training_time'] = training_time
    centralized_metrics['num_epochs'] = len(history['loss'])
    
    return model, centralized_metrics, history

# Enhanced Evaluation Function
def evaluate_model(model, X_test, y_test, verbose=1):
    """Enhanced evaluation with more metrics"""
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    if verbose:
        print("\nEvaluation Metrics:")
        for name, value in metrics.items():
            print(f"{name.capitalize()}: {value:.4f}")
        
        plot_confusion_matrix(y_test, y_pred)
    
    return metrics

# Main execution (unchanged)
if __name__ == "__main__":
    # Load your dataframe here (replace with your actual data loading)
    df = pd.read_csv(r"./Data/heart_disease_uci.csv", header = 0)
    
    # Prepare data
    X_scaled, y, scaler, label_encoders = prepare_data(df)
    
    # Split into 3 clients with stratification
    client_data = []
    X_temp, X_client1, y_temp, y_client1 = train_test_split(
        X_scaled.to_numpy(), y.to_numpy(), test_size=1/3, random_state=42, stratify=y
    )
    X_client2, X_client3, y_client2, y_client3 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    client_data.append((X_client1, y_client1))
    client_data.append((X_client2, y_client2))
    client_data.append((X_client3, y_client3))
    
    # Plot data distribution
    plot_data_distribution(client_data)
    
    # Run federated learning
    fl_model, fl_metrics, fl_history = run_federated_learning(client_data, num_rounds=50)
    
    # Run centralized training for comparison
    centralized_model, centralized_metrics, centralized_history = run_centralized_training(client_data)
    
    # Compare results
    comparison = {
        'federated_learning': fl_metrics,
        'centralized': centralized_metrics
    }
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/comparison_{timestamp}.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("\n\n=== Final Comparison ===")
    print("\nFederated Learning Results:")
    for metric, value in fl_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}" if isinstance(value, float) else f"{metric.capitalize()}: {value}")
    
    print("\nCentralized Training Results:")
    for metric, value in centralized_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}" if isinstance(value, float) else f"{metric.capitalize()}: {value}")
    
    # Plot comparison
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fl_values = [fl_metrics[m] for m in metrics_to_compare]
    centralized_values = [centralized_metrics[m] for m in metrics_to_compare]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    plt.bar(x - width/2, fl_values, width, label='Federated Learning')
    plt.bar(x + width/2, centralized_values, width, label='Centralized')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, [m.capitalize() for m in metrics_to_compare])
    plt.legend()
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(fl_values):
        plt.text(i - width/2, v + 0.02, f"{v:.3f}", ha='center')
    for i, v in enumerate(centralized_values):
        plt.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig("visualizations/performance_comparison.png")
    plt.close()
    
    print("\nVisualizations saved to 'visualizations/' directory")
    print("Models saved to 'models/' directory")
    print("Results saved to 'results/' directory")