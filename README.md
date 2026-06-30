# Cara Menjalankan Proyek

Karena adanya batas ukuran file pada GitHub dan Vercel, dataset serta model tidak disertakan di dalam repository. Ikuti langkah-langkah berikut untuk menjalankan proyek.

## pre-requirement (sebelum mulai)
pastikan telah terinstall
```
joblib
pickle
matplotlib
flask
sklearn
numpy
cv2
```
## 1. Download Dataset

Unduh dataset Helmet Detection dari Kaggle:

https://www.kaggle.com/datasets/andrewmvd/helmet-detection

Setelah selesai, ubah nama folder hasil ekstraksi menjadi:

```text
dataset/
```

---

## 2. Preprocessing Dataset

Jalankan seluruh cell pada notebook:

```text
Image_Cropping.ipynb
```

Setelah proses selesai, akan terbentuk folder baru:

```text
clean_dataset/
```

---

## 3. Training Model

Jalankan seluruh cell pada notebook:

```text
Helmet_Detection.ipynb
```

Apabila proses berhasil, folder `output/` akan berisi file berikut:

```text
output/
├── encoder.pkl
├── models.pkl
├── pca.pkl
└── scaler.pkl
```

Jika seluruh file di atas berhasil dibuat, berarti proses training telah selesai dengan benar.

<p align="center">
  <img width="631" alt="Output Training" src="https://github.com/user-attachments/assets/9203ba01-b4da-46f2-8ab3-c26d55503643">
</p>

---

## 4. Salin File Model

Pindahkan seluruh file `.pkl` dari folder `output/` ke folder:

```text
helmet_detection_flask/
```

Sehingga struktur folder menjadi seperti berikut:

<p align="center">
  <img width="642" alt="Folder Flask" src="https://github.com/user-attachments/assets/3179c00e-6744-4196-a973-682f06a33e89">
</p>

---

## 5. Menjalankan Aplikasi

Masuk ke folder `helmet_detection_flask`, kemudian jalankan:

```bash
python app.py
```

atau

```bash
flask run
```

sesuai dengan konfigurasi proyek Anda.

---
## 6. Hasil / Result
Jika sudah berjalan dapat dibuka di port 5000
<p align="center">
<img width="1600" height="957" alt="image" src="https://github.com/user-attachments/assets/9037e693-a1b1-454c-8661-16d1406dfadd" />
</p>

## Catatan

> **Repository ini tidak menyertakan dataset maupun file model (`.pkl`)** karena keterbatasan ukuran file pada GitHub dan batas maksimal ukuran deployment di Vercel (500 MB). Oleh karena itu, proses preprocessing dan training model perlu dilakukan secara lokal sebelum aplikasi dapat dijalankan.
