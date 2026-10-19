Mission-06A — Linear Regression on ESP32
Predicting Machine Temperature with MicroPython
Mata kuliah: Machine Learning
Topik: Regresi linear sederhana, training, inference, dan evaluasi pada embedded system
Perangkat: ESP32 (firmware MicroPython), kabel USB, laptop dengan Thonny IDE
Perangkat tambahan: Tidak diperlukan sensor, breadboard, atau koneksi internet
Estimasi durasi: 80–100 menit
Prasyarat: Dasar Python, regresi linear, MAE, dan penggunaan Thonny untuk menjalankan MicroPython pada ESP32

Misi: Biasanya Anda melatih dan menguji model machine learning di laptop. Dalam praktikum ini, jalankan prediksi dan perhitungan koefisien regresi langsung di ESP32. Semua nilai temperatur adalah data sintetik, bukan hasil pembacaan sensor.

1. Tujuan pembelajaran
Setelah praktikum, mahasiswa mampu:

Menjelaskan perbedaan training dan inference.
Mengimplementasikan model regresi linear sederhana pada ESP32 tanpa numpy, pandas, atau scikit-learn.
Menghitung koefisien slope dan intercept dari data training langsung pada ESP32.
Menguji model pada data yang tidak digunakan saat training dan menghitung Mean Absolute Error (MAE).
Mengidentifikasi risiko ekstrapolasi serta keterbatasan model regresi linear.
2. Studi kasus
Sebuah mesin hipotetis mengalami kenaikan temperatur seiring bertambahnya waktu operasi. Kita ingin memperkirakan temperatur berdasarkan waktu operasi.

Variabel	Simbol	Satuan	Peran
Waktu operasi	x	menit	Fitur/input
Temperatur mesin	y	°C	Target/output
Model regresi linear sederhana:

y
^
=
b
0
+
b
1
x
y
^
​
 =b 
0
​
 +b 
1
​
 x

Keterangan:

b
0
b 
0
​
 : intercept, temperatur prediksi saat 
x
=
0
x=0.
b
1
b 
1
​
 : slope, perubahan temperatur prediksi untuk tambahan satu menit operasi.
x
x: waktu operasi mesin (menit).
y
^
y
^
​
 : temperatur hasil prediksi model (°C).
Penting: Hubungan linear ini hanya ilustrasi pembelajaran. Temperatur mesin nyata dapat berubah secara nonlinear dan dipengaruhi kondisi lingkungan, beban, dan pendinginan.

Dataset sintetik
Berikut data training yang disimpan langsung di dalam kode ESP32:

Waktu (menit)	Temperatur (°C)
0	28.0
1	29.7
2	31.5
3	32.8
4	34.9
5	36.1
6	38.2
7	39.6
8	41.4
Data testing (tidak digunakan untuk menghitung koefisien):

Waktu (menit)	Temperatur (°C)
1.5	30.6
3.5	33.9
5.5	37.4
7.5	40.5
Dataset disusun sebagai simulasi pembelajaran, bukan bukti pengukuran mesin. Pemisahan train-test dipakai untuk mengevaluasi prediksi pada titik waktu yang belum dilihat model. Karena ini dataset kecil, MAE belum membuktikan kemampuan generalisasi pada mesin nyata.

3. Persiapan
Sambungkan ESP32 ke laptop melalui USB.
Jalankan Thonny dan pilih interpreter MicroPython (ESP32) serta port serial yang benar.
Pastikan Shell Thonny dapat menjalankan print("Hello ESP32").
Buat berkas baru untuk setiap eksperimen. Jalankan dari Thonny atau simpan berkas ke ESP32.
Untuk menghentikan while True, tekan Stop di Thonny atau Ctrl+C pada Shell.
Catatan: Kode di bawah hanya menggunakan fungsi bawaan Python/MicroPython. Tidak diperlukan instalasi paket ML pada ESP32.

4. Eksperimen A — Inference dengan koefisien yang sudah diketahui (15 menit)
Pada eksperimen ini kita belum melakukan training di ESP32. Anggap model linear berikut sudah diperoleh dari proses training sebelumnya:

T
^
=
28.0
+
1.7
t
T
^
 =28.0+1.7t

Buat berkas 01_inference.py:

# Mission-06A, Eksperimen A
# Inference pada ESP32 (koefisien sudah diberikan)

b0 = 28.0       # intercept (derajat C)
b1 = 1.7        # slope (derajat C per menit)

def predict(minutes):
    return b0 + b1 * minutes

for minutes in [0, 2, 4, 6, 8]:
    predicted_temp = predict(minutes)
    print("Time:", minutes, "min | Prediction:",
          round(predicted_temp, 2), "C")

Amati: ESP32 dapat menghasilkan prediksi tanpa melakukan training atau membaca sensor.

Pertanyaan cepat: Apakah b0 = 28.0 dan b1 = 1.7 dalam kode di atas dipelajari ESP32 dari dataset? Tidak. Keduanya sengaja diberikan sebagai parameter model pada eksperimen A.

5. Eksperimen B — Training regresi linear langsung di ESP32 (25 menit)
Sekarang ESP32 sendiri menghitung koefisien regresi dengan metode ordinary least squares (OLS) menggunakan data training.

Untuk pasangan data 
(
x
i
,
y
i
)
(x 
i
​
 ,y 
i
​
 ):

b
1
=
∑
i
(
x
i
−
x
ˉ
)
(
y
i
−
y
ˉ
)
∑
i
(
x
i
−
x
ˉ
)
2
b 
1
​
 = 
∑ 
i
​
 (x 
i
​
 − 
x
ˉ
 ) 
2
 
∑ 
i
​
 (x 
i
​
 − 
x
ˉ
 )(y 
i
​
 − 
y
ˉ
​
 )
​
 

b
0
=
y
ˉ
−
b
1
x
ˉ
b 
0
​
 = 
y
ˉ
​
 −b 
1
​
  
x
ˉ
 

Buat berkas 02_training.py:

# Mission-06A, Eksperimen B
# Training regresi linear sederhana langsung di ESP32

train_data = [
    (0, 28.0),
    (1, 29.7),
    (2, 31.5),
    (3, 32.8),
    (4, 34.9),
    (5, 36.1),
    (6, 38.2),
    (7, 39.6),
    (8, 41.4)
]

def fit_linear_regression(data):
    n = len(data)
    if n < 2:
        raise ValueError("Need at least two training samples")

    mean_x = sum(row[0] for row in data) / n
    mean_y = sum(row[1] for row in data) / n

    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in data
    )
    denominator = sum(
        (x - mean_x) ** 2 for x, y in data
    )

    if denominator == 0:
        raise ValueError("All training x values are identical")

    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x
    return b0, b1

def predict(x, b0, b1):
    return b0 + b1 * x

b0, b1 = fit_linear_regression(train_data)

print("Intercept:", round(b0, 4))
print("Slope:", round(b1, 4))

for minutes in [0, 2, 4, 6, 8]:
    prediction = predict(minutes, b0, b1)
    print("Time:", minutes, "min | Prediction:",
          round(prediction, 2), "C")

Diskusikan:

Bandingkan b0 dan b1 dari eksperimen B dengan angka yang diberikan dalam eksperimen A.
Mengapa hasilnya tidak harus identik?
Pada kode ini, bagian manakah yang merupakan training dan bagian manakah yang merupakan inference?
6. Eksperimen C — Mengukur error pada data testing (15 menit)
Jalankan kode 02_training.py terlebih dahulu, lalu tambahkan bagian berikut ke akhir berkas yang sama. Jangan menggunakan data testing untuk menghitung b0 dan b1.

# Tambahkan di akhir 02_training.py

test_data = [
    (1.5, 30.6),
    (3.5, 33.9),
    (5.5, 37.4),
    (7.5, 40.5)
]

absolute_errors = []

print("\n--- TEST RESULTS ---")
for minutes, actual in test_data:
    predicted = predict(minutes, b0, b1)
    error = abs(actual - predicted)
    absolute_errors.append(error)

    print("t =", minutes,
          "| Actual =", actual,
          "| Predicted =", round(predicted, 2),
          "| Abs error =", round(error, 2))

mae = sum(absolute_errors) / len(absolute_errors)
print("MAE test:", round(mae, 4), "C")

Rumus MAE:

M
A
E
=
1
n
∑
i
=
1
n
∣
y
i
−
y
^
i
∣
MAE= 
n
1
​
  
i=1
∑
n
​
 ∣y 
i
​
 − 
y
^
​
  
i
​
 ∣

Tugas mahasiswa: catat koefisien model, nilai prediksi tiap titik testing, dan MAE. Jelaskan makna satuan °C pada MAE.

Jawaban:
b
1
=
I
n
t
e
r
c
e
p
t
:
28.0222
b 
1
​
 =Intercept:28.0222

b
0
=
S
l
o
p
e
:
1.6667
b 
0
​
 =Slope:1.6667

Time	Prediction
0 min	28.02 °C
2 min	31.36 °C
4 min	34.69 °C
6 min	38.02 °C
8 min	41.36 °C
TEST RESULTS
t (min)	Actual	Predicted	Abs Error
1.5	30.6	30.52	0.08
3.5	33.9	33.86	0.04
5.5	37.4	37.19	0.21
7.5	40.5	40.52	0.02
MAE test: 0.0889 C

Makna satuan °C pada MAE adalah Satuan °C menunjukkan bahwa rata-rata kesalahan (error) prediksi memiliki unit pengukuran yang sama dengan data aslinya (suhu). Nilai MAE = 0.0889 °C berarti secara rata-rata, hasil prediksi model menyimpang sebesar 0,0889 °C dari suhu aktual.

Skala Linier: Berbeda dengan MSE yang menggunakan satuan kuadrat (
°
C
2
°C 
2
 ), MAE tetap dalam satuan asli (
°
C
°C).
Interpretasi Fisik: Nilai error dapat langsung dipahami sebagai selisih derajat Celsius di dunia nyata tanpa perlu konversi tambahan.
7. Eksperimen D — Prediksi interaktif dari terminal serial (10 menit)
Masih di 02_training.py, boleh tambahkan bagian berikut di akhir setelah evaluasi. ESP32 akan menerima input waktu operasi melalui Shell Thonny.

# Input manual dari Shell Thonny
# Ketik q untuk keluar

while True:
    user_input = input("Waktu operasi (menit), atau q: ")

    if user_input.strip().lower() == "q":
        print("Program selesai")
        break

    try:
        minutes = float(user_input)
        if minutes < 0:
            print("Waktu tidak boleh negatif")
            continue

        predicted = predict(minutes, b0, b1)
        print("Predicted temperature:",
              round(predicted, 2), "C")

        if minutes > 8:
            print("WARNING: extrapolation beyond training range")

    except ValueError:
        print("Masukkan angka atau q")

Coba input: 2.5, 6.5, 8, 12, dan 20 menit. Apakah prediksi pada 20 menit dapat langsung dipercaya? Tidak. Prediksi tersebut berada di luar rentang training (0–8 menit), sehingga bersifat ekstrapolasi dan belum divalidasi.

Jawaban:
Waktu Operasi (menit)	Prediksi Suhu (°C)	Status / Keterangan
2.5	32.19 °C	Interpolasi (Dalam rentang training)
6.5	38.86 °C	Interpolasi (Dalam rentang training)
8.0	41.36 °C	Batas Maksimum Training
12.0	48.02 °C	WARNING: Extrapolation beyond training range
20.0	61.36 °C	WARNING: Extrapolation beyond training range
🔍 Analisis Prediksi Interaktif & Ekstrapolasi
Pertanyaan:
Apakah prediksi pada 20 menit (61.36 °C) dapat langsung dipercaya?

Jawaban:
Tidak. Hasil prediksi pada 20 menit tidak dapat langsung dipercaya.

Penjelasan & Analisis:

Ekstrapolasi Data: Nilai 20 menit berada jauh di luar rentang data pelatihan (training range 0–8 menit). Model regresi linier hanya meneruskan garis lurus secara matematis berdasarkan persamaan 
y
=
28.0222
+
1.6667
x
y=28.0222+1.6667x.
Batasan Fisik Sistem: Dalam kondisi riil, suhu perangkat tidak akan naik secara linier selamanya. Perangkat akan melepas panas ke lingkungan hingga mencapai titik jenuh/saturasi termal (steady-state).
Validitas Model: Model ini hanya teruji akurat pada rentang 0–8 menit (dengan MAE = 0.0889 °C). Untuk durasi operasi di atas 8 menit, diperlukan pengumpulan data latih baru atau penggunaan model non-linier.
8. Eksperimen E — Tantangan modifikasi dataset (15 menit)
Ubah hanya data training, lalu lakukan training dan evaluasi ulang dengan test_data yang sama. Hasilnya dicatat dalam tabel.

Skenario 1: Gunakan 9 data training asli.
Skenario 2: Hapus data training pada t = 2, t = 4, dan t = 6.
Skenario 3: Gunakan data training asli, tetapi ubah satu nilai temperatur, yaitu pada t = 4 dari 34.9 menjadi 45.0 °C (outlier buatan).
Skenario	Jumlah data training	Intercept (b0)	Slope (b1)	MAE testing (°C)
1 — Asli	9	…	…	…
2 — Data dikurangi	6	…	…	…
3 — Satu outlier	9	…	…	…
Analisis: Apakah dataset yang lebih banyak selalu meningkatkan kinerja? Mengapa satu outlier dapat memengaruhi hasil OLS? Jelaskan berdasarkan hasil, bukan asumsi.

9. Ringkasan: Apa yang benar-benar dijalankan di ESP32?
Bagian	Eksperimen A	Eksperimen B–E
Dataset training di ESP32	Tidak diperlukan	Ya
Menghitung b0 dan b1 di ESP32	Tidak	Ya
Inference/prediksi di ESP32	Ya	Ya
Evaluasi pada test set	Tidak	Ya (C dan E)
Sensor fisik	Tidak	Tidak
Library machine learning	Tidak	Tidak
Catatan ilmiah: Praktikum ini mendemonstrasikan on-device training skala kecil dan on-device inference pada ESP32. Ini adalah pengantar embedded ML/TinyML. Tidak berarti metode OLS dengan data besar atau semua jenis training ML cocok dilakukan pada ESP32.

10. Laporan singkat mahasiswa
Kumpulkan laporan 1–2 halaman yang memuat:

Foto/screenshot keluaran Thonny dari eksperimen B, C, dan D.
Persamaan regresi hasil training ESP32 serta MAE testing.
Tabel hasil eksperimen E.
Jawaban tiga pertanyaan refleksi berikut:
Apa perbedaan training dan inference dalam praktikum ini?
Apakah sebuah prediksi tetap bisa dihitung meskipun berada di luar rentang training? Mengapa prediksi itu perlu diwaspadai?
Apa keuntungan dan keterbatasan melakukan training model sederhana langsung di ESP32 dibanding hanya melakukan inference?
11. Kriteria keberhasilan
 ESP32 menjalankan prediksi menggunakan koefisien tetap.
 ESP32 berhasil menghitung slope dan intercept dari data training.
 ESP32 menghasilkan MAE menggunakan data testing yang terpisah.
 Mahasiswa mencoba input interaktif dan mengidentifikasi ekstrapolasi.
 Mahasiswa membandingkan hasil tiga skenario dataset.



Mission-06A — Linear Regression on ESP32
Predicting Machine Temperature with MicroPython
1. Tujuan pembelajaran
2. Studi kasus
Dataset sintetik
3. Persiapan
4. Eksperimen A — Inference dengan koefisien yang sudah diketahui (15 menit)
5. Eksperimen B — Training regresi linear langsung di ESP32 (25 menit)
6. Eksperimen C — Mengukur error pada data testing (15 menit)
Jawaban:
TEST RESULTS
7. Eksperimen D — Prediksi interaktif dari terminal serial (10 menit)
Jawaban:
🔍 Analisis Prediksi Interaktif & Ekstrapolasi
8. Eksperimen E — Tantangan modifikasi dataset (15 menit)
9. Ringkasan: Apa yang benar-benar dijalankan di ESP32?
10. Laporan singkat mahasiswa
11. Kriteria keberhasilan
