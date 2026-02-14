# ========================================
# PRODUCTION CNN FOR DIGIT RECOGNITION
# Kaggle Notebook - Ready to Run
# Expected Accuracy: 99%+
# Training Time: 15-20 minutes
# ========================================

# STEP 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ========================================
# STEP 2: Load and Prepare Data
# ========================================

print("\n📊 Loading MNIST Dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")
print(f"Image shape: {X_train.shape[1:]}")

# Visualize sample digits
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle("Sample MNIST Digits", fontsize=16)
plt.tight_layout()
plt.show()

# Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(10, 4))
plt.bar(unique, counts, color='steelblue')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.title('Training Data Distribution')
plt.xticks(range(10))
for i, count in enumerate(counts):
    plt.text(i, count, str(count), ha='center', va='bottom')
plt.show()

# ========================================
# STEP 3: Preprocess Data
# ========================================

print("\n🔧 Preprocessing data...")

# Reshape and normalize
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f"✓ Train shape: {X_train.shape}")
print(f"✓ Test shape: {X_test.shape}")
print(f"✓ Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")

# ========================================
# STEP 4: Data Augmentation
# CRITICAL: This helps model handle real user drawings
# ========================================

print("\n🎨 Setting up data augmentation...")

datagen = ImageDataGenerator(
    rotation_range=10,       # Rotate images by ±10 degrees
    width_shift_range=0.1,   # Shift horizontally by 10%
    height_shift_range=0.1,  # Shift vertically by 10%
    zoom_range=0.1,          # Zoom in/out by 10%
    shear_range=0.1          # Shear transformation
)

datagen.fit(X_train)

# Visualize augmented samples
print("Sample augmented images:")
sample_img = X_train[0:1]
plt.figure(figsize=(12, 3))
for i in range(10):
    augmented = datagen.flow(sample_img, batch_size=1)
    batch = next(augmented)
    plt.subplot(2, 5, i+1)
    plt.imshow(batch[0].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Data Augmentation Examples", fontsize=16)
plt.tight_layout()
plt.show()

# ========================================
# STEP 5: Build CNN Model
# ========================================

print("\n🏗️ Building CNN architecture...")

def create_production_cnn():
    """
    Production-grade CNN for digit recognition
    Architecture based on best practices for MNIST
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
        layers.BatchNormalization(name='bn1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(name='bn1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.BatchNormalization(name='bn2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.BatchNormalization(name='bn2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.4, name='dropout3'),
        
        # Dense Layers
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn4'),
        layers.Dropout(0.5, name='dropout4'),
        
        # Output layer
        layers.Dense(10, activation='softmax', name='output')
    ])
    
    return model

model = create_production_cnn()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model architecture
model.summary()

# Count parameters
total_params = model.count_params()
print(f"\n📊 Total parameters: {total_params:,}")

# ========================================
# STEP 6: Setup Callbacks
# ========================================

print("\n⚙️ Setting up training callbacks...")

callbacks = [
    # Early stopping - stops training when validation accuracy plateaus
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_digit_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ========================================
# STEP 7: Train Model
# ========================================

print("\n🚀 Starting training...")
print("=" * 60)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    epochs=30,  # Will stop early if optimal
    steps_per_epoch=len(X_train) // 128,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

print("=" * 60)
print("✅ Training completed!")

# ========================================
# STEP 8: Visualize Training History
# ========================================

print("\n📈 Visualizing training history...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================
# STEP 9: Evaluate Model
# ========================================

print("\n📊 Evaluating model performance...")

# Load best model
model = keras.models.load_model('best_digit_model.h5')

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*60}")
print(f"🎯 FINAL TEST ACCURACY: {test_accuracy*100:.2f}%")
print(f"🎯 FINAL TEST LOSS: {test_loss:.4f}")
print(f"{'='*60}")

# Get predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Detailed classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=[str(i) for i in range(10)]))

# ========================================
# STEP 10: Confusion Matrix
# ========================================

print("\n🔍 Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10),
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

# Add accuracy per class
for i in range(10):
    accuracy = (cm[i, i] / cm[i].sum()) * 100
    plt.text(10.5, i+0.5, f'{accuracy:.1f}%', 
             ha='center', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# Per-class accuracy
print("\n📊 Per-Class Accuracy:")
print("-" * 40)
for digit in range(10):
    digit_acc = (cm[digit, digit] / cm[digit].sum()) * 100
    digit_count = cm[digit].sum()
    print(f"Digit {digit}: {digit_acc:5.2f}% ({cm[digit, digit]}/{digit_count})")
print("-" * 40)

# ========================================
# STEP 11: Confidence Analysis
# ========================================

print("\n🔍 Analyzing prediction confidence...")

confidences = np.max(y_pred, axis=1)

print(f"Mean confidence: {np.mean(confidences)*100:.2f}%")
print(f"Median confidence: {np.median(confidences)*100:.2f}%")
print(f"Min confidence: {np.min(confidences)*100:.2f}%")
print(f"Max confidence: {np.max(confidences)*100:.2f}%")

# Confidence distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(confidences, bins=50, color='steelblue', edgecolor='black')
plt.xlabel('Confidence', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
plt.axvline(0.95, color='red', linestyle='--', label='95% threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# Low confidence examples
low_conf_indices = np.where(confidences < 0.95)[0]
print(f"\nPredictions with <95% confidence: {len(low_conf_indices)} ({len(low_conf_indices)/len(y_test)*100:.2f}%)")

# Show some low confidence examples
plt.subplot(1, 2, 2)
if len(low_conf_indices) > 0:
    n_examples = min(9, len(low_conf_indices))
    for i in range(n_examples):
        idx = low_conf_indices[i]
        plt.subplot(3, 3, i+1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}\nConf: {confidences[idx]:.2%}")
        plt.axis('off')
    plt.suptitle('Low Confidence Examples', fontsize=14, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'All predictions have >95% confidence!', 
             ha='center', va='center', fontsize=14)
    plt.axis('off')

plt.tight_layout()
plt.show()

# ========================================
# STEP 12: Error Analysis
# ========================================

print("\n❌ Analyzing misclassified examples...")

errors = y_test != y_pred_classes
error_indices = np.where(errors)[0]
print(f"Total errors: {len(error_indices)} / {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")

if len(error_indices) > 0:
    # Show first 9 errors
    plt.figure(figsize=(12, 9))
    n_errors = min(9, len(error_indices))
    for i in range(n_errors):
        idx = error_indices[i]
        plt.subplot(3, 3, i+1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_classes[idx]}\n"
                  f"Confidence: {confidences[idx]:.2%}")
        plt.axis('off')
    plt.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Most common confusion pairs
    print("\nMost confused digit pairs:")
    confusion_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((i, j, cm[i, j]))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (true_digit, pred_digit, count) in enumerate(confusion_pairs[:5], 1):
        print(f"{i}. {true_digit} → {pred_digit}: {count} times")

# ========================================
# STEP 13: Save Model (FIXED FOR KERAS 3)
# ========================================

print("\n💾 Saving model in multiple formats...")

# Format 1: Keras H5 format (best for Python deployment)
model.save('digit_recognition_model.h5')
print("✓ Saved: digit_recognition_model.h5")

# Format 2: Keras native format (Keras 3 default)
model.save('digit_recognition_model.keras')
print("✓ Saved: digit_recognition_model.keras")

# Format 3: Convert to TensorFlow Lite (for mobile/edge)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('digit_recognition_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("✓ Saved: digit_recognition_model.tflite")

# Model size comparison
import os
h5_size = os.path.getsize('digit_recognition_model.h5') / (1024 * 1024)
keras_size = os.path.getsize('digit_recognition_model.keras') / (1024 * 1024)
tflite_size = len(tflite_model) / (1024 * 1024)
print(f"\n📦 Model sizes:")
print(f"   H5 format: {h5_size:.2f} MB")
print(f"   Keras format: {keras_size:.2f} MB")
print(f"   TFLite format: {tflite_size:.2f} MB (optimized)")

# ========================================
# STEP 14: Test Model Loading
# ========================================

print("\n🧪 Testing model loading...")

# Test loading H5
loaded_model = keras.models.load_model('digit_recognition_model.h5')
test_pred = loaded_model.predict(X_test[0:1], verbose=0)
print(f"✓ H5 model loads correctly")
print(f"   Sample prediction: {np.argmax(test_pred)} (confidence: {np.max(test_pred):.2%})")

# ========================================
# STEP 15: Generate Report
# ========================================

print("\n" + "="*60)
print("📊 FINAL REPORT")
print("="*60)
print(f"Model: Production CNN")
print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Final accuracy: {test_accuracy*100:.2f}%")
print(f"Total errors: {len(error_indices)}")
print(f"Model size (H5): {h5_size:.2f} MB")
print(f"Model size (TFLite): {tflite_size:.2f} MB")
print(f"Training epochs: {len(history.history['accuracy'])}")
print("="*60)

# Check if accuracy meets production standard
if test_accuracy >= 0.99:
    print("\n✅ SUCCESS! Model meets production standard (≥99%)")
    print("✅ Ready for deployment!")
else:
    print(f"\n⚠️ WARNING: Accuracy is {test_accuracy*100:.2f}%")
    print("   Target is ≥99% for production")
    print("   Consider:")
    print("   - Training for more epochs")
    print("   - Adjusting learning rate")
    print("   - Adding more data augmentation")

print("\n🎉 Training complete! Download 'digit_recognition_model.h5' for deployment.")
print("\n📝 Next steps:")
print("   1. Download the .h5 model file")
print("   2. Build preprocessing pipeline")
print("   3. Create FastAPI backend")
print("   4. Build frontend interface")
print("   5. Deploy to cloud (Render/HuggingFace)")

# ========================================
# BONUS: Quick Inference Test
# ========================================

print("\n🎮 Quick inference test (10 random samples):")
print("-" * 60)

test_indices = np.random.choice(len(X_test), 10, replace=False)
for idx in test_indices:
    pred = model.predict(X_test[idx:idx+1], verbose=0)
    pred_class = np.argmax(pred)
    conf = np.max(pred)
    correct = "✓" if pred_class == y_test[idx] else "✗"
    print(f"{correct} True: {y_test[idx]} | Predicted: {pred_class} | Confidence: {conf:.2%}")

print("-" * 60)
print("✅ All done!")
