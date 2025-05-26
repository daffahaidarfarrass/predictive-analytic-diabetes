# Laporan Proyek Machine Learning - Daffa Haidar Farras
## Domain Proyek (Kesehatan)
Diabetes melitus adalah salah satu penyakit tidak menular dengan tingkat penderita yang terus meningkat di dunia, termasuk Indonesia. Berdasarkan data dari International Diabetes Federation (IDF), jumlah penderita diabetes di Indonesia sudah mencapai 19,5 juta pada tahun 2021, dengan jumlah ini Indonesia menjadi salah satu negara dengan angka penderita diabetes tertinggi di dunia [1]. Diabetes menjadi penyebab utama komplikasi kesehatan serius, seperti penyakit  gagal ginjal, kardiovaskular, dan neuropati, yang dapat memberikan dampak pada kualitas hidup individu serta dapat memberi beban pada sistem layanan kesehatan nasional [2]. 
Peningkatan penderita diabetes di Indonesia disebabkan oleh beberapa faktor, termasuk perubahan gaya hidup, pola makan, kurangnya aktivitas fisik, dan kurangnya kesadaran petingnya pencegahan diabetes [3]. Diabetes yang merupakan penyakit kronis memerlukan penanganan jangka panjang, penanganannya berupa diet, gula darah, olahraga, dan pengobatan rutin. 

Pada jurnal “Projection of diabetes morbidity and mortality till 2045 in Indonesia based on risk factors and NCD prevention and control programs” yang ditulis oleh Wahidin, mugi dkk. Menurut penelitian yang mengambil data dari Riset Kesehatan Dasar (Riskesdas), BPJS Kesehatan, program Penyakit Tidak Menular (PTM), serta Kementerian Kesehatan. Penderita diabetes di Indonesia diperkirakan akan meningkat dari 9,19% pada tahun 2020 (setara dengan 18,69 juta kasus) menjadi 16,09% pada tahun 2045 (sekitar 40,7 juta kasus)[4]. 

Melihat fenomena ini diperlukannya inovasi yang bisa mencegah dan mendeteksi diabetes sejak dini. Salah satu pendekatan nya yaitu, dengan menggunakan machine learning untuk memprediksi dan mengidentifikasi potensi risiko penyakit diabetes berdasarkan data kesehatan individu. Pada projek ini menggunakan beberapa pendekatan algoritma machine learning antara lain : Random Forest, Decision Tree, AdaBoosting, SVM, dan KNN. 

## Business Understanding
### Problem Statements
Rumusan masalah dari masalah latar belakang diatas adalah :
1. Bagaimana memprediksi penyakit diabetes sejak dini berdasarkan data kesehatan individu?
2. Bagaimana menidentifikasi fitur atau faktor utama yang bisa berkontribusi terhadap kemungkinana seseorang dapat menderita penyakit diabetes?  
3. Bagaimana membangun model machine learning yang bisa diandalkan untuk bisa membantu pengambilan keputusan diagnosis?
### Goals
Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini : 
1. Mengembangkan model machine learning yang bisa mengklasifikasi seseorang menderita diabetes atau tidak.
2. Mengetahui fitur atau faktor yang bisa berkontribusi terhadap kemungkinana seseorang dapat menderita penyakit diabetes.
3. Menemukan model terbaik yang bisa mengklasifikasi penderita diabetes.

### Solution Statement
1. Membangun dan mengembangkan model machine learning yang dapat mengklasifikasikan risiko penyakit diabetes.
2. Melakukan sebuah analisis pada data untuk bisa memahami fitur-fitur yang mempengaruhi seseorang dapat terkena penyakit diabetes, dengan menerapkan teknik visualisasi data dan deskripsi statistik data mengetahui korelasi antar fitur dan memahami hubungan antara data target (label) dan fitur lainnya.
3. Menggunakan confusion matrix dan f1 score pada masing-masing model machine learning untuk menemukan model terbaik berdasarkan akurasi tertinggi.

## Data Understanding
Dataset yang digunakan berisi informasi kesehatan individu yang dikumpulkan untuk membantu mengembangkan model untuk memprediksi penderita diabetes. Dataset ini diambil dari platform [Kaggle](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes) Dataset ini memiliki 10 fitur (termasuk label/target), dengan masing-masing baris mewakili satu entri individu.

| Kolom | Deskripsi |
| ---- |----- |
| **Id** | Identitas unik untuk setiap entri data. |
| **Pregnancies** | Jumlah kehamilan yang pernah dialami (indikator faktor risiko bagi wanita).|
| **Glucose** | Konsentrasi glukosa plasma selama 2 jam dalam tes toleransi glukosa oral. |
| **BloodPressure** | Tekanan darah diastolik (dalam mm Hg). |
| **SkinThickness** | Ketebalan lipatan kulit triceps (dalam mm). |
| **Insulin** | Kadar insulin serum selama 2 jam (mu U/ml). |
| **BMI** | Indeks massa tubuh (berat badan dalam kg dibagi tinggi badan dalam m²). |
| **DiabetesPedigreeFunction** | Nilai fungsi riwayat keluarga/genetik terhadap risiko diabetes. |
| **Age** | Usia individu (dalam tahun). |
| **Outcome** | Label hasil klasifikasi: 1 = positif diabetes, 0 = negatif diabetes.        |


Dengan : 
| Jumlah Baris | Jumlah Kolom |
| --- | --- |
| 2768 | 10 |

### Memeriksa duplicate value
![cek_dupikat](https://github.com/user-attachments/assets/fb86118c-112e-4b66-9219-c34b19ed7573)


Dari hasil di atas, terlihat bahwa tidak ada data yang terduplikasi.
### Memeriksa Missing Value
![cek_missing_value](https://github.com/user-attachments/assets/2f08fc41-830b-441a-b279-44f1bad8fe1f)


dari output diatas diketahui bahwa tidak terdapat missing value pada dataset yang digunakan, tetapi harus dicek apakah terdapat nilai nol pada tiap kolom karena tidak mungkin nilai pada kolom Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age. Karena bisa saja missing value nya diubah menjadi 0, hal ini dapat mempengaruhi peforma machine learning.

## Deskripsi Statistik dari Data
|index|Id|Pregnancies|Glucose|BloodPressure|SkinThickness|Insulin|BMI|DiabetesPedigreeFunction|Age|Outcome|
|---|---|---|---|---|---|---|---|---|---|---|
|count|2768\.0|2768\.0|2768\.0|2768\.0|2768\.0|2768\.0|2768\.0|2768\.0|2768\.0|2768\.0|
|mean|1384\.5|3\.7427745664739884|121\.8901180711016|72\.25643032259681|26\.843041030438705|80\.1278901734104|32\.590194462832876|0\.4711925578034682|33\.13222543352601|0\.3439306358381503|
|std|799\.1970970918251|3\.3238009672683835|30\.501026868115638|12\.007473411734374|9\.812960210906155|112\.30193298150999|7\.103630403759228|0\.32566883299525706|11\.777229987737929|0\.475104095143688|
|min|1\.0|0\.0|44\.0|24\.0|7\.0|0\.0|18\.2|0\.078|21\.0|0\.0|
|25%|692\.75|1\.0|99\.0|64\.0|20\.82442196531792|0\.0|27\.575000000000003|0\.244|24\.0|0\.0|
|50%|1384\.5|3\.0|118\.0|72\.0|23\.0|37\.0|32\.2|0\.375|29\.0|0\.0|
|75%|2076\.25|6\.0|141\.0|80\.0|32\.0|130\.0|36\.625|0\.624|40\.0|1\.0|
|max|2768\.0|17\.0|199\.0|122\.0|110\.0|846\.0|80\.6|2\.42|81\.0|1\.0|

## Univariate Analysis

Dari dataset yang digunakan diketahui bahwa hanya terdapat fitur bertipe data float dan int sehingga kita bisa memvisualisasikan datanya tanpa perlu memisahkan fitur-fitur yang memiliki tipedata yang berbeda

### Pregnancies

Top 10 dengan persentase paling banyak
|Pregnancies|jumlah sampel|persentase|
|---|---|---|
|1|491|17\.7|
|0|412|14\.9|
|2|387|14\.0|
|3|270|9\.8|
|4|259|9\.4|
|5|198|7\.2|
|6|181|6\.5|
|7|145|5\.2|
|8|134|4\.8|
|9|98|3\.5|

![Pregnancies](https://github.com/user-attachments/assets/48034493-fd29-4eb7-8ad9-631f2e50828d)

Insight :
- Ada 491 individu yang memiliki 1 kehamilan, dan 412 individu yang tidak pernah mengalami kehamilan (0 kehamilan)
- 1 kehamilan merupakan 17.7% dari total dataset, sedangkan 0 kehamilan adalah 14.9% dari total.
- Distribusi Menurun


### Glucose

Top 10 dengan persentase paling banyak
|Glucose|jumlah sampel|persentase|
|---|---|---|
|99\.0|66|2\.4|
|100\.0|61|2\.2|
|102\.0|52|1\.9|
|129\.0|51|1\.8|
|106\.0|50|1\.8|
|95\.0|49|1\.8|
|112\.0|49|1\.8|
|111\.0|47|1\.7|
|105\.0|47|1\.7|
|125\.0|46|1\.7|

![Glucose](https://github.com/user-attachments/assets/43e72654-be2f-4ee7-9ca6-da58bbcc135e)

Insight :
- Terdapat 66 individu dengan kadar glukosa 99.0.
- Nilai glukosa yang paling sering ditemukan memiliki frekuensi lebih tinggi, seperti 99.0 (2.4%), 100.0 (2.2%), dan 102.0 (1.9%)
- Distribusi Menyebar


### BloodPressure

Top 10 dengan persentase paling banyak
|BloodPressure|jumlah sampel|persentase|
|---|---|---|
|70\.0|201|7\.3|
|74\.0|197|7\.1|
|78\.0|173|6\.2|
|68\.0|170|6\.1|
|64\.0|163|5\.9|
|72\.0|162|5\.9|
|80\.0|138|5\.0|
|76\.0|132|4\.8|
|60\.0|129|4\.7|
|62\.0|128|4\.6|

![BloodPressure](https://github.com/user-attachments/assets/83fd6ee3-8233-4a2b-8018-1ebe9714fce3)

Insight:
- Terdapat 201 individu dengan tekanan darah diastolik sebesar 70.0 mmHg.
- Nilai tekanan darah 70.0 mmHg ditemukan pada 7.3% dari total populasi dalam dataset.
- Distribusi Menurun

### SkinThickness

Top 10 dengan persentase paling banyak
|SkinThickness|jumlah sampel|persentase|
|---|---|---|
|20\.82442196531792|800|28\.9|
|32\.0|114|4\.1|
|30\.0|102|3\.7|
|23\.0|82|3\.0|
|27\.0|81|2\.9|
|28\.0|74|2\.7|
|18\.0|74|2\.7|
|33\.0|71|2\.6|
|39\.0|70|2\.5|
|31\.0|69|2\.5|

![SkinThickness](https://github.com/user-attachments/assets/7cb5ebea-fa8b-45f2-8b28-ddb43c486c7e)

Insight:
- Ada 800 individu dengan nilai ketebalan kulit 0 mm.
- 28.9% dari total populasi memiliki nilai ketebalan kulit 0 mm.
- Nilai "0" kemungkinan besar mencerminkan data yang hilang, bukan pengukuran sebenarnya.

### Insulin

Top 10 dengan persentase paling banyak
|Insulin|jumlah sampel|persentase|
|---|---|---|
|0|1330|48\.0|
|105|42|1\.5|
|140|33|1\.2|
|130|31|1\.1|
|180|30|1\.1|
|120|29|1\.0|
|100|27|1\.0|
|94|24|0\.9|
|135|23|0\.8|
|76|22|0\.8|

![Insulin](https://github.com/user-attachments/assets/5235e630-f9bc-4ede-8054-bd1ba91b1e24)

Insight:
- Ada 1330 individu dengan nilai ketebalan kulit 0 mm.
- 48% dari total populasi memiliki nilai ketebalan kulit 0 mm.
- Nilai "0" kemungkinan besar mencerminkan data yang hilang, bukan pengukuran sebenarnya.

### BMI

Top 10 dengan persentase paling banyak
|BMI|jumlah sampel|persentase|
|---|---|---|
|32\.0|46|1\.7|
|31\.2|45|1\.6|
|31\.6|41|1\.5|
|32\.13739161849711|39|1\.4|
|33\.3|37|1\.3|
|32\.4|35|1\.3|
|32\.8|34|1\.2|
|30\.8|33|1\.2|
|32\.9|33|1\.2|
|30\.1|31|1\.1|

![BMI](https://github.com/user-attachments/assets/8fe7725f-67c9-4810-9382-d6c5c1536664)

Insight:
- Ada 46 individu memiliki nilai BMI 32.0.
- 1.7% dari total populasi memiliki BMI 32.0.
- Nilai BMI 0.0 muncul sebanyak 39 sampel (1.4%). Nilai ini tidak realistis dan kemungkinan besar mencerminkan missing values (data yang hilang).

### DiabetesPedigreeFunction

Top 10 dengan persentase paling banyak
|DiabetesPedigreeFunction|jumlah sampel|persentase|
|---|---|---|
|0\.258|22|0\.8|
|0\.207|20|0\.7|
|0\.268|18|0\.7|
|0\.238|18|0\.7|
|0\.261|18|0\.7|
|0\.259|17|0\.6|
|0\.284|16|0\.6|
|0\.52|16|0\.6|
|0\.292|16|0\.6|
|0\.551|16|0\.6|

![DiabetesPedigreeFunction](https://github.com/user-attachments/assets/9fa01b23-3833-471b-964c-811e3fe383e5)

Insight:
- Ada 22 individu memiliki nilai DPF sebesar 0.258.
- 0.8% dari total populasi memiliki nilai DPF 0.258. 
- Distribusi data yang tersebar.

### Age

Top 10 dengan persentase paling banyak
|Age|jumlah sampel|persentase|
|---|---|---|
|22|264|9\.5|
|21|229|8\.3|
|25|182|6\.6|
|24|168|6\.1|
|23|141|5\.1|
|28|133|4\.8|
|26|117|4\.2|
|27|113|4\.1|
|29|99|3\.6|
|31|82|3\.0|

![Age](https://github.com/user-attachments/assets/f95e0a01-f15f-49b6-84d6-0c54fbc2cde4)

Insight:
- Ada 264 individu yang berusia 22 tahun.
- 22 tahun mencakup 9.5% dari total populasi.
- Distribusi pola data menurun

## Multivariate Analysis
### Umur vs Kondisi Diabetes

![Umur_Diabetes](https://github.com/user-attachments/assets/20536ff4-e675-4728-973e-c67a253d563e)


Orang yang menderita diabetes cenderung berusia lebih tua, namun faktor usia bukan satu-satunya penentu karena ada banyak variasi usia di kedua kelompok. Analisis ini menunjukkan pentingnya mempertimbangkan usia sebagai salah satu variabel penting dalam deteksi risiko diabetes, meskipun tidak cukup berdiri sendiri. Terlihat banyak outlier usia tinggi pada kelompok yang tidak menderita (usia di atas 60 tahun). Bisa menunjukkan bahwa tidak semua lansia menderita diabetes, ada faktor lain yang mungkin melindungi (gaya hidup, genetik, dll).

### BMI vs Kondisi Diabetes

![BMI_Diabetes](https://github.com/user-attachments/assets/22f6f4d7-fff6-4e49-ae58-47c2cb511efc)


Terdapat kecenderungan bahwa individu yang menderita diabetes memiliki BMI yang lebih tinggi dibanding yang tidak menderita.


### Glukosa vs Kondisi Diabetes



Nilai glukosa yang tinggi sangat berkorelasi dengan kondisi diabetes. Distribusinya menunjukkan bahwa penderita diabetes cenderung memiliki glukosa darah yang lebih tinggi secara signifikan.

### Pairplot berdasarkan Outcome

![Pairplot berdasarkan Outcome](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/Pairplot_berdasarkan_Outcome.png)

1. Diagonal (Garis diagonal: Distribusi masing-masing fitur)
   - Distribusi Glucose dan BMI terlihat lebih tinggi pada kelompok yang menderita diabetes (warna oranye).
   - Distribusi Insulin sangat menyebar dan memiliki banyak nilai nol, terutama pada Outcome = 0.
   - Distribusi Age menunjukkan kelompok penderita diabetes cenderung berumur lebih tua.
2. Scatter plot antar fitur
   - Glucose vs. BMI: Penderita diabetes (Outcome = 1, warna oranye) cenderung memiliki nilai Glucose dan BMI yang lebih tinggi.
   - Glucose vs. Insulin: Korelasi positif terlihat (semakin tinggi Glucose, cenderung Insulin juga meningkat).
   - BMI vs. Age: Tidak terlihat pola yang kuat; namun, penderita diabetes lebih banyak berada di BMI tinggi dan usia tua.
   - BMI vs. Glucose : Terlihat bahwa penderita diabetes (Outcome = 1, warna oranye) cenderung memiliki nilai Glucose dan BMI yang tinggi.
   - Insulin vs. BMI : Distribusi agak menyebar, namun terlihat bahwa penderita diabetes dengan BMI tinggi juga cenderung memiliki nilai Insulin lebih tinggi.
   - Insulin vs. Age : Tidak tampak korelasi yang jelas antara Insulin dan Age. Namun, beberapa penderita diabetes di usia tua (60+) tampak memiliki Insulin tinggi. 

   

### Correlation Matrix

![Correlation Matrix](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/Correlation_Matrix.png)

|  |	Outcome| Interpretasi |
|---|---|--- |
|Outcome	|1.000000| |
|Glucose	|0.535222| semakin tinggi Glucose, semakin besar kemungkinan diabetes.|
|BMI	|0.317163| orang dengan BMI tinggi cenderung berisiko diabetes. |
|Age	|0.237050| usia lanjut sedikit meningkatkan risiko. |
|Pregnancies|0.223796| kehamilan berulang dapat berkaitan dengan gestational diabetes. |
|SkinThickness	|0.171925| meskipun berhubungan dengan lemak tubuh, korelasi kecil. |
|BloodPressure	|0.169353| ekanan darah tidak terlalu memengaruhi outcome secara langsung. | 
|DiabetesPedigreeFunction	|0.160664| meski ini faktor genetik, korelasinya tidak terlalu besar. |
|Insulin	|0.123646| korelasi rendah. |
|Id	|0.006298| hanya nomor identitas. | 

## Data Preparation
### Mengecek Outlier

![Mengecek Outlier](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/Mengecek_Outlier.png)

Hasilnya adalah :
1. Pregnancies
   Dapat dilihat mayoritas berada di angka 1-6. Terdapat outlier yaitu diatas 13. Tetapi, outlier ini tidak akan dihapus karena dalam beberapa kasus bisa saja wanita melahirkan lebih dari 13 kali.
2. Glucose
   Pada grafik Glucose terlihat normal tidak adanya outlier
3. BloodPressure
   Terdapat nilai nol, yang tidak masuk akal sebagai tekanan darah, kemungkinan data tidak valid atau hilang. Maka outlier ini akan diisi dengan nila rata-rata dari fiturnya.
4. SkinThickness
   Ada nilai nol dan outlier besar di atas 100 mm, yang jarang terjadi dalam praktik medis. Karena rentang rata-rata dalam dunia medis adalah 0-80 mm. oleh karena itu outlier ini akan diisi dengan nila rata-rata dari fiturnya.
5. Insulin
    Banyak nilai outlier sangat tinggi (hingga lebih dari 800). 
6. BMI
    Ada outlier di atas 50, bahkan mendekati 80. Karena BMI > 70  saja biasanya sudah dianggap sangat tinggi. Maka outlier ini akan diisi dengan nila rata-rata dari fiturnya.
7. DiabetesPedigreeFunction
    Terdapat Outlier di atas 2 menunjukkan risiko genetik tinggi. Maka outlier ini akan diisi dengan nila rata-rata dari fiturnya.
8. Age
    Tidak terlihat adanya keanehan dari distribusi data di fitur Age.
   
### Menangani Outlier
![Cek Nilai 0](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/cek_nilai_0.png)

Setelah dicek ternyata terdapat banyak sekali yang bernilai 0, karena terdapat banyak sekali nilai 0 dan akan sangat berpengaruh ke peforma machine learning jika semuanya di drop. Maka, nilai 0 disini akan di ganti dengan nilai rata-rata pada setiap fitur. Terutama pada fitur SkinThickness, BMI, Glucose, dan BloodPressure yang tidak mungkin bernilai 0.

### Mengganti nilai 0 dengan nilai rata-rata
![Mengganti nilai 0](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/ganti_nilai_0.png)

lalu selanjutnya adalah menangangi kesalah nilai yang dimana itu adalah hal yang tidak mungkin di dunia medis

![Mengganti nilai yang tidak masuk akal](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/nilai_tidak_masuk_akal.png)



### Data Spliting
Setelah melakukan analisis terhadap data yang akan digunakan, selanjutnya adalah melakukan data preparation. Pertama kita akan melakukan splitting data train dan data test sebesar 80:20. 

![Data Spliting](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/Data_Spliting.png)

### SMOTE dan Standardisasi Fitur
Karena terlihat persebaran data train 0 dan 1 terlihat tidak seimbang (imbalanced data) maka saya melakukan SMOTE untuk menyeimbangkan data train. Lalu, selanjutnya saya melakukan Standardisasi Fitur untuk digunakan pada beberapa model.

![SMOTE dan Standardisasi Fitur](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/smote_Standardisasi_Fitur.png)


## Modeling
Ada 5 algoritma Machine Learning yang digunakan untuk membuat model, yaitu sebagai berikut.
### Random Forest
#### Apa itu?
Random Forest adalah algoritma ensemble learning berbasis pohon keputusan. Ia bekerja dengan membuat banyak pohon keputusan saat pelatihan dan menggabungkan hasilnya (dengan voting untuk klasifikasi atau rata-rata untuk regresi).
#### Cara Kerja
- Data dilatih melalui banyak pohon keputusan (biasanya ratusan).
- Setiap pohon dilatih pada subset acak dari data (bootstrap sampling).
- Fitur juga dipilih secara acak saat setiap node dibagi (random subspace).
- Hasil akhir diambil berdasarkan mayoritas (untuk klasifikasi) atau rata-rata (untuk regresi).
#### Tahapan
- Membuat 100 decision trees.
- Melatih model pada data hasil oversampling yang menggunakan SMOTE (`X_resampled`, `y_resampled`).
- Voting dilakukan di antara semua pohon untuk klasifikasi akhir.
#### Parameter yang digunakan
- `n_estimators=100`: Jumlah pohon yang digunakan dalam ensemble.
- `random_state=42`: Menjamin reproducibility.
- parameter default: `max_depth=None`, `bootstrap=True`
### Decision Tree
#### Apa itu?
Decision Tree adalah struktur pohon di mana setiap node internal menguji fitur, setiap cabang mewakili hasil tes, dan setiap daun mewakili label.
#### Cara Kerja
- Data dibagi berdasarkan fitur yang paling mengurangi impurity (contoh: Gini Impurity, Entropy).
- Proses ini berlanjut hingga semua data terklasifikasi atau mencapai kedalaman maksimum.
- Overfitting adalah masalah umum jika tidak dipangkas.
#### Tahapan
- Membuat satu decision tree penuh.
- Melatih model pada data oversampled.
#### Parameter yang digunakan
- Menggunakan parameter default
- `criterion='gini'`: Fungsi untuk mengukur kualitas split.
- `max_depth=None`: Pohon tumbuh sampai semua daun murni.
### AdaBoosting
#### Apa itu?
AdaBoost adalah metode boosting yang menggabungkan beberapa model lemah (biasanya decision stumps) menjadi satu model kuat.
#### Cara Kerja
- Melatih model lemah secara berurutan.
- Setiap model baru fokus pada kesalahan dari model sebelumnya.
- Memberikan bobot lebih pada kesalahan untuk model selanjutnya.
- Hasil akhir adalah kombinasi tertimbang dari semua model.
#### Tahapan
- Membuat boosting dari 50 weak learners (decision stumps).
- Decision tree dasar dipakai untuk iterasi boosting.
- Latihan pada data asli yang belum di-resample.
#### Parameter yang digunakan
- `estimator=dt_model`: Base learner (bisa decision stump).
- `n_estimators=50`: Jumlah iterasi boosting.
- `random_state=42`: Konsistensi hasil eksperimen.
### SVM
#### Apa itu?
SVM adalah algoritma klasifikasi yang mencari hyperplane terbaik untuk memisahkan kelas data.
#### Cara Kerja
- Mencari hyperplane yang memaksimalkan margin antara kelas.
- Menggunakan kernel trick untuk menangani data yang tidak dapat dipisahkan secara linear.
- Cocok untuk dataset dengan dimensi tinggi.
#### Tahapan
- Membangun hyperplane optimal dalam ruang berdimensi tinggi.
- Data distandarkan (`X_train_scaled`) agar kernel bekerja optimal.
#### Parameter yang digunakan
- `kernel='rbf'`: Menggunakan Radial Basis Function kernel.
- `C=1`: Parameter regularisasi, kontrol margin vs. kesalahan.
- `gamma='scale'`: Otomatis menghitung gamma berdasarkan variansi fitur.
### KNN
#### Apa itu?
KNN adalah metode berbasis instance yang menyimpan seluruh dataset dan mengklasifikasikan berdasarkan tetangga terdekat.
#### Cara Kerja
- Hitung jarak (misalnya Euclidean) dari titik yang ingin diklasifikasikan ke semua titik dalam data pelatihan.
- Pilih k tetangga terdekat.
- Kelas mayoritas dari tetangga tersebut adalah prediksi.
#### Tahapan
- Mencari 5 tetangga terdekat untuk setiap instance uji.
- Menentukan label berdasarkan mayoritas dari 5 tetangga.
#### Parameter yang digunakan
- `n_neighbors=5`: Jumlah tetangga yang digunakan untuk voting.
- Data harus diskalakan agar perhitungan jarak adil.

### Model yang dipilih
Setelah semua model dijalankan dan menguji data menggunakan 5 model machine learning, Model AdaBoosting memberikan performa terbaik dan dapat diandalkan untuk klasifikasi kasus diabetes dibandingkan model lainnya berdasarkan skor akurasi, skor F1, dan jumlah kesalahan klasifikasi yang paling sedikit. 


## Evaluasi
### Confusion Matrix, Akurasi, dan F1-Score
#### 1. Confusion Matrix
- Menampilkan jumlah:
  - TP (True Positive): Prediksi benar terhadap kelas positif.
  - TN (True Negative): Prediksi benar terhadap kelas negatif.
  - FP (False Positive): Prediksi salah, prediksi positif tapi sebenarnya negatif.
  - FN (False Negative): Prediksi salah, prediksi negatif tapi sebenarnya positif
- Contoh Tampilan
- 
![Contoh Tampilan](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/CM_Random_Forest.png)

#### 2. Akurasi
Akurasi Mengukur proporsi total prediksi yang benar. Tidak cocok jika data tidak seimbang.

![Akurasi](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/Akurasi.png)

#### 3. F1-Score
F1-Score adalah Harmonik rata-rata dari presisi dan recall. Cocok untuk data tidak seimbang.

![F1-Score](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/F1-Score.png)

#### 4. Recall
Recall (juga dikenal sebagai Sensitivity atau True Positive Rate) adalah metrik evaluasi yang menunjukkan seberapa baik model mendeteksi semua instance positif dalam data.

![Recall](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/Recall.png)


keterangan :
- TP (True Positives): Jumlah data positif yang diprediksi benar.
- FN (False Negatives): Jumlah data positif yang diprediksi salah sebagai negatif.


### Penerapan Matriks Confusion
#### 1. Random Forest

![Confusion Matrix Random Forest](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/CM_Random_Forest.png)

Berdasarkan confusion matrix diatas:
- 360 responden diklasifikasikan benar sebagai TIDAK DIABETES (True Negative).
- 178 responden diklasifikasikan benar sebagai DIABETES (True Positive).
- 7 responden diklasifikasikan salah sebagai DIABETES, padahal sebenarnya TIDAK DIABETES (False Positive).
- 9 responden diklasifikasikan salah sebagai TIDAK DIABETES, padahal sebenarnya DIABETES (False Negative).

#### 2. Decision Tree

![Confusion Matrix Decision Tree](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/CM_Decision_Tree.png)

Berdasarkan confusion matrix diatas:
- 320 responden diklasifikasikan benar sebagai TIDAK DIABETES (True Negative).
- 63 responden diklasifikasikan benar sebagai DIABETES (True Positive).
- 47 responden diklasifikasikan salah sebagai DIABETES, padahal sebenarnya TIDAK DIABETES (False Positive).
- 124 responden diklasifikasikan salah sebagai TIDAK DIABETES, padahal sebenarnya DIABETES (False Negative).


#### 3. AdaBoosting

![Confusion Matrix AdaBoosting](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/CM_AdaBoosting.png)

Berdasarkan confusion matrix diatas:
- 365 responden diklasifikasikan benar sebagai TIDAK DIABETES (True Negative).
- 180 responden diklasifikasikan benar sebagai DIABETES (True Positive).
- 2 responden diklasifikasikan salah sebagai DIABETES, padahal sebenarnya TIDAK DIABETES (False Positive).
- 7 responden diklasifikasikan salah sebagai TIDAK DIABETES, padahal sebenarnya DIABETES (False Negative).
#### 4. SVM

![Confusion Matrix SVM](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/CM_SVM.png)

Berdasarkan confusion matrix diatas:
- 311 responden diklasifikasikan benar sebagai TIDAK DIABETES (True Negative).
- 151 responden diklasifikasikan benar sebagai DIABETES (True Positive).
- 56 responden diklasifikasikan salah sebagai DIABETES, padahal sebenarnya TIDAK DIABETES (False Positive).
- 36 responden diklasifikasikan salah sebagai TIDAK DIABETES, padahal sebenarnya DIABETES (False Negative).
#### 5. KNN

![Confusion Matrix KNN](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/CM_KNN.png)

Berdasarkan confusion matrix diatas:
- 313 responden diklasifikasikan benar sebagai TIDAK DIABETES (True Negative).
- 169 responden diklasifikasikan benar sebagai DIABETES (True Positive).
- 54 responden diklasifikasikan salah sebagai DIABETES, padahal sebenarnya TIDAK DIABETES (False Positive).
- 18 responden diklasifikasikan salah sebagai TIDAK DIABETES, padahal sebenarnya DIABETES (False Negative).

#### Penerapan Accuracy, F1-Score, Recall

| |          Model|  Accuracy  |F1-Score    |Recall|
|---|---|---|---|---|
|0  |Random Forest  |0.981949  |0.972973  |0.962567|
|1  |Decision Tree  |0.691336  |0.424242  |0.336898|
|2    |AdaBoosting  |0.983755  |0.975610  |0.962567|
|3    |        SVM  |0.833935  |0.766497  |0.807487|
|4    |        KNN  |0.870036  |0.824390  |0.903743|

## Kesimpulan

![Kesimpulan Fitur yang mempengaruhi](https://github.com/daffahaidarfarrass/predictive-analytic-diabetes/blob/main/images/Kesimpulan_fitur.png)

1. Berdasarkan data yang diperolah, menunjukan Fitur utama yang memengaruhi kemungkinan diabetes adalah `Glucose`, `BMI`, `Age`, dan `Pregnancies`. ke-4 faktor ini sangat berpengaruh pada perhitungan prediksi diabetes.
2. Setelah menguji data menggunakan 5 model machine learning, Model AdaBoosting memberikan performa terbaik dan dapat diandalkan untuk klasifikasi kasus diabetes dibandingkan model lainnya berdasarkan skor akurasi, skor F1, dan jumlah kesalahan klasifikasi yang paling sedikit.
3. Model ini dapat digunakan untuk membantu pengambilan keputusan medis awal, terutama pada tahap skrining atau pencegahan.

## Referensi
---
1. Afriani, T., & Pudiyanti, P. (2020). Peran Teknologi Informasi dalam Perawatan Diabetes Mellitus. Nursing Current: Jurnal Keperawatan. https://doi.org/10.19166/NC.V8I1.2722
2. Rahmini, J. A., & Rahayuningtyas, D. K. (2020, Nov). Inovasi Kesehatan Terkini Sebagai Strategi Efektif Pada Manajemen Diabetes Di Masa Pandemi. Jurnal Keperawatan, 5(2). https://doi.org/10.32668/jkep.v5i2.453
3. Ardila, M., S. Humolungo, D. T., Amukti, D. P., & Akrom. (2024, Jun). PROMOSI KESEHATAN PENCEGAHAN DAN PENGENDALIAN DIABETES MELITUS PADA REMAJA. J . A . I : Jurnal Abdimas Indonesia, 4(2). https://doi.org/10.53769/jai.v4i2.729
4. Wahidin, M. Achadi, A. DKK. (2024). Projection of diabetes morbidity and mortality till 2045 in Indonesia based on risk factors and NCD prevention and control programs. Scientific report. https://www.nature.com/articles/s41598-024-54563-2
