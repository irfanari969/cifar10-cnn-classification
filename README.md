# Model Deep Learning: Klasifikasi Gambar CIFAR-10 🖼️

Model **Convolutional Neural Network (CNN)** ini dilatih untuk mengklasifikasikan gambar dari dataset **CIFAR-10** ke dalam 10 kategori yang berbeda. Proyek ini mendemonstrasikan langkah-langkah lengkap dari pemuatan data, pra-pemrosesan, pembangunan arsitektur CNN, pelatihan, hingga evaluasi model.

* **Nama Proyek:** Project Camp 03: Membuat Model Deep Learning (CNN)
* **Tanggal:** 23/09/2025

## 🧐 Sekilas Tentang Dataset (CIFAR-10)

Dataset CIFAR-10 terdiri dari **60.000 gambar berwarna** (32x32 piksel) dalam 10 kelas, dengan 6.000 gambar per kelas.

| Indeks | Nama Kelas |
| :---: | :---: |
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |



---

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python
* **Deep Learning Framework:** TensorFlow 2.x (dengan Keras API)
* **Pustaka Pendukung:** NumPy, Matplotlib

---

## 🧠 Arsitektur Model CNN

Model ini menggunakan arsitektur CNN sekuensial dasar yang efektif untuk klasifikasi gambar.

| Lapisan | Jenis Lapisan | Output Shape | Parameters | Fungsi |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Conv2D` (32 filters, 3x3 kernel, ReLU) | (None, 30, 30, 32) | 896 | Mengekstrak fitur dasar. |
| 2 | `MaxPooling2D` (2x2) | (None, 15, 15, 32) | 0 | Mengurangi dimensi (downsampling). |
| 3 | `Conv2D` (64 filters, 3x3 kernel, ReLU) | (None, 13, 13, 64) | 18,496 | Mengekstrak fitur yang lebih kompleks. |
| 4 | `MaxPooling2D` (2x2) | (None, 6, 6, 64) | 0 | Mengurangi dimensi. |
| 5 | `Conv2D` (64 filters, 3x3 kernel, ReLU) | (None, 4, 4, 64) | 36,928 | Mengekstrak fitur tingkat tinggi. |
| 6 | `Flatten` | (None, 1024) | 0 | Mengubah output 2D menjadi vektor 1D. |
| 7 | `Dense` (64 neurons, ReLU) | (None, 64) | 65,600 | Lapisan *fully connected* pertama. |
| 8 | **`Dense` (10 neurons, Softmax)** | (None, 10) | 650 | **Lapisan Output** (Probabilitas 10 kelas). |

**Total Parameters:** 122,570

---

## ⚙️ Pra-pemrosesan Data

Sebelum pelatihan, data dipersiapkan dengan dua langkah utama:

1.  **Pemuatan Data:** Menggunakan `datasets.cifar10.load_data()` untuk mendapatkan data latih (`train`) dan data uji (`test`).
2.  **Normalisasi:** Nilai piksel diubah dari rentang **\[0, 255\]** menjadi **\[0, 1\]** dengan membagi semua *image array* dengan **255.0**. Hal ini penting untuk performa dan stabilitas pelatihan.

```python
train_images, test_images = train_images / 255.0, test_images / 255.0
