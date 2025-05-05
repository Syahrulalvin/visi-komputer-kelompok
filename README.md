# visi-komputer-kelompok

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Folder dataset dan referensi
dataset_folder = 'images'
ref_image_path = 'mangga-hijau.jpg'  # ganti jika perlu

# ======== Fungsi Histogram HSV & RGB ==========
def calculate_hsv_histogram(image_path, bins=64):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def calculate_rgb_histogram(image_path, bins=64):
    image = cv2.imread(image_path)
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [bins], [0, 256])
        features.append(hist)
    features = [cv2.normalize(h, h).flatten() for h in features]
    return features  # list of B, G, R histograms

# ======== Visualisasi Histogram ==========
def plot_histogram_set(hist_list, color_list, title, bins=64):
    for hist, color in zip(hist_list, color_list):
        plt.plot(hist, color=color)
    plt.title(title)
    plt.xlabel("Bin")
    plt.ylabel("Frequency")
    plt.grid(True)

# ======== Bangun Index Gambar Dataset ==========
image_filenames = [f for f in os.listdir(dataset_folder)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg')) and f != os.path.basename(ref_image_path)]

hist_index = {}
for filename in image_filenames:
    path = os.path.join(dataset_folder, filename)
    hist = calculate_hsv_histogram(path)
    hist_index[filename] = hist

# ======== Histogram Referensi ==========
ref_hist = calculate_hsv_histogram(ref_image_path)

# ======== Cari Kemiripan Histogram ==========
similarities = {}
for filename, hist in hist_index.items():
    sim = cosine_similarity([ref_hist], [hist])[0][0]
    similarities[filename] = sim

sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
top_match_name = sorted_similarities[0][0]

# ======== Load Gambar Referensi dan Hasil ==========
ref_img = cv2.imread(ref_image_path)
ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
top_match_path = os.path.join(dataset_folder, top_match_name)
top_img = cv2.imread(top_match_path)
top_img_rgb = cv2.cvtColor(top_img, cv2.COLOR_BGR2RGB)

# ======== Tampilkan Gambar dan Histogram HSV & RGB ==========
plt.figure(figsize=(14, 10))

# -- Gambar Referensi dan Histogramnya
plt.subplot(3, 2, 1)
plt.imshow(ref_img_rgb)
plt.title("Gambar Referensi")
plt.axis("off")

plt.subplot(3, 2, 2)
ref_hsv = calculate_hsv_histogram(ref_image_path)
ref_rgb = calculate_rgb_histogram(ref_image_path)
plot_histogram_set(ref_rgb, ['r', 'g', 'b'], "Histogram RGB Referensi")

# -- Gambar Paling Mirip dan Histogramnya
plt.subplot(3, 2, 3)
plt.imshow(top_img_rgb)
plt.title(f"Paling Mirip: {top_match_name}")
plt.axis("off")

plt.subplot(3, 2, 4)
top_rgb = calculate_rgb_histogram(top_match_path)
plot_histogram_set(top_rgb, ['r', 'g', 'b'], "Histogram RGB Paling Mirip")

# -- Histogram HSV (gabungan H, S, V) Visualisasi
plt.subplot(3, 2, 5)
ref_h = ref_hsv[:64]
ref_s = ref_hsv[64:128]
ref_v = ref_hsv[128:]
plot_histogram_set([ref_h, ref_s, ref_v], ['magenta', 'cyan', 'yellow'], "Histogram HSV Referensi")

plt.subplot(3, 2, 6)
top_hsv = calculate_hsv_histogram(top_match_path)
top_h = top_hsv[:64]
top_s = top_hsv[64:128]
top_v = top_hsv[128:]
plot_histogram_set([top_h, top_s, top_v], ['magenta', 'cyan', 'yellow'], "Histogram HSV Paling Mirip")

plt.tight_layout()
plt.show()

# ======== Tampilkan Histogram Index Semua Gambar Dataset (RGB) ==========
print("\nHistogram Index (RGB) untuk semua gambar dataset:")
