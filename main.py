import os
import sys
import numpy as np
from PIL import Image

# --- Konfigurasi ---
IMG_SIZE = (32, 32) # Harus sama dengan training!
MODEL_PATH = "cnn_model.npz"
CLASSES = ['Tidak Ada Tumor', 'Tumor Terdeteksi']

# --- Arsitektur CNN (Copy-Paste dari training.py) ---
class ConvLayer:
    def __init__(self, num_filters, filter_size, input_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        # Placeholder, nanti di-load
        self.filters = np.zeros((num_filters, filter_size, filter_size, input_channels))

    def forward(self, inputs):
        B, H, W, C = inputs.shape
        H_out = H - self.filter_size + 1
        W_out = W - self.filter_size + 1
        output = np.zeros((B, H_out, W_out, self.num_filters))
        
        for i in range(H_out):
            for j in range(W_out):
                patch = inputs[:, i:i+self.filter_size, j:j+self.filter_size, :]
                for f in range(self.num_filters):
                    output[:, i, j, f] = np.sum(patch * self.filters[f], axis=(1, 2, 3))
        return output

class MaxPoolLayer:
    def forward(self, inputs):
        B, H, W, C = inputs.shape
        new_H = H // 2
        new_W = W // 2
        output = np.zeros((B, new_H, new_W, C))
        for i in range(new_H):
            for j in range(new_W):
                patch = inputs[:, i*2:i*2+2, j*2:j*2+2, :]
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
        return output

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((input_size, output_size))
        self.bias = np.zeros((1, output_size))
        
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

class CNNModel:
    def __init__(self):
        # Struktur harus sama persis dengan training.py
        self.conv = ConvLayer(num_filters=8, filter_size=3, input_channels=3)
        self.pool = MaxPoolLayer()
        
        # Perhitungan dimensi flatten:
        # Input 32 -> Conv(3x3) -> 30 -> Pool(2x2) -> 15
        # Output shape: 15x15x8
        self.flatten_dim = 15 * 15 * 8
        
        self.dense1 = DenseLayer(self.flatten_dim, 64)
        self.dense2 = DenseLayer(64, 1)

    def load(self, path):
        try:
            data = np.load(path)
            self.conv.filters = data['conv_filters']
            self.dense1.weights = data['d1_w']
            self.dense1.bias = data['d1_b']
            self.dense2.weights = data['d2_w']
            self.dense2.bias = data['d2_b']
        except Exception as e:
            raise RuntimeError(f"Gagal memuat bobot model: {e}")

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
        d1 = self.dense1.forward(x)
        d1 = np.maximum(0, d1) # ReLU
        
        # 5. Dense 2
        d2 = self.dense2.forward(d1)
        prob = 1 / (1 + np.exp(-d2)) # Sigmoid
        return prob

def predict(image_path):
    print(f"1. Memuat model CNN dari '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    try:
        model = CNNModel()
        model.load(MODEL_PATH)
    except Exception as e:
        print(f"Error saat load model: {e}", file=sys.stderr)
        sys.exit(1)

    print("2. Memproses gambar...")
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        
        # Normalisasi & Batch dimension
        img_array = np.array(img, dtype=np.float32) / 255.0
        # (32, 32, 3) -> (1, 32, 32, 3)
        input_data = np.expand_dims(img_array, 0)

    except Exception as e:
        print(f"Error memproses gambar: {e}", file=sys.stderr)
        return

    print("3. Melakukan prediksi (Konvolusi)...")
    try:
        score = model.forward(input_data)[0][0]
    except Exception as e:
        print(f"Error saat prediksi: {e}", file=sys.stderr)
        return

    # Hasil
    print("\n--- Hasil Prediksi CNN ---")
    print(f"Gambar: {os.path.basename(image_path)}")
    if score < 0.5:
        predicted_class = CLASSES[0]
        confidence = 1 - score
    else:
        predicted_class = CLASSES[1]
        confidence = score
    
    print(f"Hasil: {predicted_class}")
    print(f"Keyakinan: {confidence:.2%}")
    print("--------------------------")
    print(f"Raw Score: {score:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_gambar>")
        sys.exit(1)
    
    predict(sys.argv[1])
