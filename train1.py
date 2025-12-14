import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# --- Konfigurasi ---
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
EPOCHS = 50 # Cukup cepat dengan Numpy untuk MLP
LEARNING_RATE = 0.01
DATASET_PATH = 'dataset/'
CLASSES = ['no', 'yes']

def get_image_paths_and_labels(path):
    """Memuat semua path gambar dan memberikan label."""
    image_paths, labels = [], []
    patterns = {'yes': 'yes/*.jpg', 'no': 'no/*.jpg'}
    for label_idx, (class_name, pattern) in enumerate(patterns.items()):
        files = glob.glob(os.path.join(path, pattern))
        image_paths.extend(files)
        labels.extend([label_idx] * len(files))
        print(f"Ditemukan {len(files)} gambar untuk kelas '{class_name}'")
    
    if not image_paths:
        print("Error: Tidak ada gambar yang ditemukan.", file=sys.stderr)
        sys.exit(1)
        
    # Acak data
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    
    image_paths = np.array(image_paths)[indices]
    labels = np.array(labels)[indices]
    
    return image_paths, labels

def load_batch(paths, labels, batch_size):
    """Generator untuk memuat batch gambar."""
    num_samples = len(paths)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_paths = paths[batch_indices]
        batch_labels = labels[batch_indices]
        
        images = []
        valid_labels = []
        for i, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_arr = np.array(img, dtype=np.float32) / 255.0
                images.append(img_arr.flatten()) # Flatten: (64*64*3,)
                valid_labels.append(batch_labels[i])
            except Exception as e:
                continue
                
        if not images:
            continue

        x = np.array(images) # (B, 12288)
        y = np.array(valid_labels).reshape(-1, 1) # (B, 1)
        yield x, y

# --- Model MLP Numpy Sederhana ---
class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inisialisasi bobot (He initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2)) # Sigmoid
        return self.a2

    def backward(self, x, y, output, learning_rate):
        m = x.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0) # ReLU derivative
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        
    def load(self, path):
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

def compute_loss(y_true, y_pred):
    # Binary Cross Entropy, added epsilon for stability
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_accuracy(y_true, y_pred):
    predictions = (y_pred > 0.5).astype(int)
    return np.mean(predictions == y_true)

def main():
    print("1. Memuat path gambar dan label...")
    all_paths, all_labels = get_image_paths_and_labels(DATASET_PATH)
    
    split_idx = int(len(all_paths) * TRAIN_SPLIT)
    train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
    
    print(f"   Sampel training: {len(train_paths)}, Sampel validasi: {len(val_paths)}")

    print("\n2. Inisialisasi Model Numpy (MLP)...")
    input_size = IMG_SIZE[0] * IMG_SIZE[1] * 3
    model = SimpleMLP(input_size, 128, 1)

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    print(f"\n3. Memulai training ({EPOCHS} epoch)...")
    
    for epoch in range(EPOCHS):
        # Training
        train_loss_sum = 0
        train_acc_sum = 0
        train_batches = 0
        
        for X, Y in load_batch(train_paths, train_labels, BATCH_SIZE):
            out = model.forward(X)
            loss = compute_loss(Y, out)
            acc = compute_accuracy(Y, out)
            
            model.backward(X, Y, out, LEARNING_RATE)
            
            train_loss_sum += loss
            train_acc_sum += acc
            train_batches += 1
            
        avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0
        avg_train_acc = train_acc_sum / train_batches if train_batches > 0 else 0

        # Validation
        val_loss_sum = 0
        val_acc_sum = 0
        val_batches = 0
        
        for X, Y in load_batch(val_paths, val_labels, BATCH_SIZE):
            out = model.forward(X)
            loss = compute_loss(Y, out)
            acc = compute_accuracy(Y, out)
            
            val_loss_sum += loss
            val_acc_sum += acc
            val_batches += 1
            
        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0
        avg_val_acc = val_acc_sum / val_batches if val_batches > 0 else 0

        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"loss: {avg_train_loss:.4f} - acc: {avg_train_acc:.4f} - "
              f"val_loss: {avg_val_loss:.4f} - val_acc: {avg_val_acc:.4f}")
        
        history['loss'].append(avg_train_loss)
        history['accuracy'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(avg_val_acc)

    print("\n4. Menyimpan model...")
    model.save("best_model.npz")
    print("   Model disimpan di 'best_model.npz'")

    print("\n5. Menyimpan riwayat training dan plot...")
    df_history = pd.DataFrame(history)
    df_history.to_csv("training_history.csv", index=False)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(df_history['loss'], label='Loss Training')
    plt.plot(df_history['val_loss'], label='Loss Validasi')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(df_history['accuracy'], label='Akurasi Training')
    plt.plot(df_history['val_accuracy'], label='Akurasi Validasi')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_and_accuracy_plot.png')
    
    print("   Riwayat & plot disimpan.")

if __name__ == "__main__":
    main()