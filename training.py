import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# --- Konfigurasi ---
IMG_SIZE = (32, 32) # Turunkan ke 32x32 agar training NumPy manual tidak lemot
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
EPOCHS = 20 # CNN belajar lebih cepat daripada MLP
LEARNING_RATE = 0.01
DATASET_PATH = 'dataset/'

def get_image_paths_and_labels(path):
    image_paths, labels = [], []
    patterns = {'yes': 'yes/*.jpg', 'no': 'no/*.jpg'}
    for label_idx, (class_name, pattern) in enumerate(patterns.items()):
        files = glob.glob(os.path.join(path, pattern))
        image_paths.extend(files)
        labels.extend([label_idx] * len(files))
    
    if not image_paths:
        print("Error: Tidak ada gambar.", file=sys.stderr)
        sys.exit(1)
        
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    return np.array(image_paths)[indices], np.array(labels)[indices]

# --- Lapisan CNN Manual (NumPy) ---
class ConvLayer:
    """
    Lapisan Konvolusi Sederhana.
    Menggunakan filter acak tetap (Random Feature Extractor) untuk kecepatan & stabilitas training manual.
    Ini menangkap fitur visual (tepi, tekstur) tanpa perlu backprop yang sangat berat di layer ini.
    """
    def __init__(self, num_filters, filter_size, input_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Inisialisasi filter acak (He Initialization style)
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels) * np.sqrt(2.0 / (filter_size * filter_size * input_channels))

    def forward(self, inputs):
        # inputs: (Batch, Height, Width, Channels)
        B, H, W, C = inputs.shape
        H_out = H - self.filter_size + 1
        W_out = W - self.filter_size + 1
        
        output = np.zeros((B, H_out, W_out, self.num_filters))
        
        # Implementasi vektorisasi loop untuk kecepatan
        for i in range(H_out):
            for j in range(W_out):
                # Ambil patch gambar: (Batch, 3, 3, Channels)
                patch = inputs[:, i:i+self.filter_size, j:j+self.filter_size, :]
                # Dot product dengan filter: (Batch, Filters)
                # Kita sum di sumbu 1,2,3 (H_f, W_f, C)
                # Filters: (NumFilter, 3, 3, C)
                # Hasil konvolusi di posisi (i,j) untuk semua batch dan semua filter
                for f in range(self.num_filters):
                    output[:, i, j, f] = np.sum(patch * self.filters[f], axis=(1, 2, 3))
                    
        return output # (Batch, 30, 30, 8)

class MaxPoolLayer:
    def forward(self, inputs):
        # inputs: (Batch, H, W, Channels)
        B, H, W, C = inputs.shape
        new_H = H // 2
        new_W = W // 2
        output = np.zeros((B, new_H, new_W, C))
        
        for i in range(new_H):
            for j in range(new_W):
                h_start, h_end = i*2, i*2+2
                w_start, w_end = j*2, j*2+2
                patch = inputs[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
                
        return output # (Batch, 15, 15, 8)

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, dz, learning_rate):
        # dz: gradient dari layer setelahnya
        m = self.input.shape[0]
        dw = np.dot(self.input.T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        dx = np.dot(dz, self.weights.T)
        
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        return dx

# --- Model Wrapper ---
class CNNModel:
    def __init__(self):
        # Input: 32x32x3
        self.conv = ConvLayer(num_filters=8, filter_size=3, input_channels=3) 
        # Out Conv: 30x30x8
        self.pool = MaxPoolLayer()
        # Out Pool: 15x15x8 = 1800
        
        self.flatten_dim = 15 * 15 * 8
        self.dense1 = DenseLayer(self.flatten_dim, 64)
        self.dense2 = DenseLayer(64, 1)

    def forward(self, x):
        # 1. Conv
        x = self.conv.forward(x)
        x = np.maximum(0, x) # ReLU
        
        # 2. Pool
        x = self.pool.forward(x)
        
        # 3. Flatten
        self.batch_size = x.shape[0]
        x = x.reshape(self.batch_size, -1)
        
        # 4. Dense 1
        self.d1_out = self.dense1.forward(x)
        self.d1_act = np.maximum(0, self.d1_out) # ReLU
        
        # 5. Dense 2 (Output)
        self.d2_out = self.dense2.forward(self.d1_act)
        self.prob = 1 / (1 + np.exp(-self.d2_out)) # Sigmoid
        
        return self.prob

    def backward(self, y_true, learning_rate):
        # Backprop hanya untuk Dense Layers (Transfer Learning style)
        # Ini membuat training jauh lebih cepat & stabil tanpa library autograd
        
        # Gradient Output Layer (Sigmoid + BCE) -> (Prob - y)
        dz2 = self.prob - y_true
        
        # Gradient Dense 2
        dz1 = self.dense2.backward(dz2, learning_rate)
        
        # Gradient ReLU Dense 1
        dz1[self.d1_out <= 0] = 0
        
        # Gradient Dense 1
        self.dense1.backward(dz1, learning_rate)

    def save(self, path):
        np.savez(path, 
                 conv_filters=self.conv.filters,
                 d1_w=self.dense1.weights, d1_b=self.dense1.bias,
                 d2_w=self.dense2.weights, d2_b=self.dense2.bias)

def train_step(model, x, y):
    out = model.forward(x)
    model.backward(y, LEARNING_RATE)
    
    # Metrics
    preds = (out > 0.5).astype(int)
    acc = np.mean(preds == y)
    epsilon = 1e-15
    out = np.clip(out, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(out) + (1 - y) * np.log(1 - out))
    return loss, acc

def main():
    print("1. Memuat data (Resize ke 32x32).")
    paths, labels = get_image_paths_and_labels(DATASET_PATH)
    
    # Split
    split = int(len(paths) * TRAIN_SPLIT)
    train_paths, val_paths = paths[:split], paths[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    
    print(f"   Training: {len(train_paths)}, Validasi: {len(val_paths)}")
    
    model = CNNModel()
    
    print("\n2. Memulai Training CNN Manual...")
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        # --- Training ---
        train_losses, train_accs = [], []
        # Shuffle batch
        indices = np.arange(len(train_paths))
        np.random.shuffle(indices)
        
        for start in range(0, len(train_paths), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(train_paths))
            batch_idx = indices[start:end]
            
            # Load images on the fly
            batch_imgs = []
            for p in train_paths[batch_idx]:
                try:
                    img = Image.open(p).convert('RGB').resize(IMG_SIZE)
                    batch_imgs.append(np.array(img, dtype=np.float32) / 255.0)
                except: continue
            
            if not batch_imgs: continue
            
            x_batch = np.array(batch_imgs)
            y_batch = train_labels[batch_idx].reshape(-1, 1)
            
            loss, acc = train_step(model, x_batch, y_batch)
            train_losses.append(loss)
            train_accs.append(acc)
            
            print(f"\r   Epoch {epoch+1} - Batch {start//BATCH_SIZE+1} - Loss: {loss:.4f}", end="")
            
        # --- Validation ---
        val_losses, val_accs = [], []
        for start in range(0, len(val_paths), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_paths))
            # Load images
            batch_imgs = []
            for p in val_paths[start:end]:
                try:
                    img = Image.open(p).convert('RGB').resize(IMG_SIZE)
                    batch_imgs.append(np.array(img, dtype=np.float32) / 255.0)
                except: continue
            
            if not batch_imgs: continue
            x_val = np.array(batch_imgs)
            y_val = val_labels[start:end].reshape(-1, 1)
            
            out = model.forward(x_val)
            # Val loss
            epsilon = 1e-15
            out = np.clip(out, epsilon, 1 - epsilon)
            v_loss = -np.mean(y_val * np.log(out) + (1 - y_val) * np.log(1 - out))
            v_acc = np.mean((out > 0.5).astype(int) == y_val)
            
            val_losses.append(v_loss)
            val_accs.append(v_acc)

        avg_t_loss = np.mean(train_losses)
        avg_t_acc = np.mean(train_accs)
        avg_v_loss = np.mean(val_losses)
        avg_v_acc = np.mean(val_accs)
        
        history['loss'].append(avg_t_loss)
        history['acc'].append(avg_t_acc)
        history['val_loss'].append(avg_v_loss)
        history['val_acc'].append(avg_v_acc)
        
        print(f"\rEpoch {epoch+1}/{EPOCHS} | Loss: {avg_t_loss:.4f} | Acc: {avg_t_acc:.4f} | Val Loss: {avg_v_loss:.4f} | Val Acc: {avg_v_acc:.4f}")

    print("\n3. Menyimpan Model CNN...")
    model.save("cnn_model.npz")
    print("   Disimpan sebagai 'cnn_model.npz'")
    
    # Save Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1); plt.plot(history['loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(history['acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Accuracy'); plt.legend()
    plt.savefig('cnn_training.png')

if __name__ == "__main__":
    main()
