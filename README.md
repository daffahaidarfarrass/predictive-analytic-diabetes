# Laporan Proyek Machine Learning - Daffa Haidar Farras
## Domain Proyek (Kesehatan)
Diabetes melitus adalah salah satu penyakit tidak menular dengan tingkat penderita yang terus meningkat di dunia, termasuk Indonesia. Berdasarkan data dari International Diabetes Federation (IDF), jumlah penderita diabetes di Indonesia sudah mencapai 19,5 juta pada tahun 2021, dengan jumlah ini Indonesia menjadi salah satu negara dengan angka penderita diabetes tertinggi di dunia [1]. Diabetes menjadi penyebab utama komplikasi kesehatan serius, seperti penyakit  gagal ginjal, kardiovaskular, dan neuropati, yang dapat memberikan dampak pada kualitas hidup individu serta dapat memberi beban pada sistem layanan kesehatan nasional [2]. 
Peningkatan penderita diabetes di Indonesia disebabkan oleh beberapa faktor, termasuk perubahan gaya hidup, pola makan, kurangnya aktivitas fisik, dan kurangnya kesadaran petingnya pencegahan diabetes [3]. Diabetes yang merupakan penyakit kronis memerlukan penanganan jangka panjang, penanganannya berupa diet, gula darah, olahraga, dan pengobatan rutin. 

Pada jurnal “Projection of diabetes morbidity and mortality till 2045 in Indonesia based on risk factors and NCD prevention and control programs” yang ditulis oleh Wahidin, mugi dkk. Menurut penelitian yang mengambil data dari Riset Kesehatan Dasar (Riskesdas), BPJS Kesehatan, program Penyakit Tidak Menular (PTM), serta Kementerian Kesehatan. Penderita diabetes di Indonesia diperkirakan akan meningkat dari 9,19% pada tahun 2020 (setara dengan 18,69 juta kasus) menjadi 16,09% pada tahun 2045 (sekitar 40,7 juta kasus)[4]. 

Melihat fenomena ini diperlukannya inovasi yang bisa mencegah dan mendeteksi diabetes sejak dini. Salah satu pendekatan nya yaitu, dengan menggunakan machine learning untuk memprediksi dan mengidentifikasi potensi risiko penyakit diabetes berdasarkan data kesehatan individu. Pada projek ini menggunakan beberapa pendekatan algoritma machine learning antara lain : Random Forest, XGBoost, SVM, dan KNN. 

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

### Menangani duplicate value
![Cek Duplikasi](images/cek_dupikat.png)
Dari hasil di atas, terlihat bahwa tidak ada data yang terduplikasi.
### Menangani Missing Value
![Cek Missing Value](images/cek_missing_value.png)
dari output diatas diketahui bahwa tidak terdapat missing value pada dataset yang digunakan, tetapi harus dicek apakah terdapat nilai nol pada tiap kolom karena tidak mungkin nilai pada kolom Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age. Karena bisa saja missing value nya diubah menjadi 0, hal ini dapat mempengaruhi peforma machine learning.
### Menangani Outlier
![Cek Nilai 0](images/cek_nilai_0.png)
Setelah dicek ternyata terdapat banyak sekali yang bernilai 0, karena terdapat banyak sekali nilai 0 dan akan sangat berpengaruh ke peforma machine learning jika semuanya di drop. Maka, nilai 0 disini akan di ganti dengan nilai rata-rata pada setiap fitur. Terutama pada fitur Insulin, BMI, Glucose, dan SkinThickness yang tidak mungkin bernilai 0.
### Mengganti nilai 0 dengan nilai rata-rata
![Mengganti nilai 0](images/ganti_nilai_0.png)

### Mengecek Outlier
![Mengecek Outlier](images/Mengecek_Outlier.png)

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

## Deskripsi Statistik dari Data
|Id|	Pregnancies|	Glucose	|BloodPressure|	SkinThickness|	Insulin|	BMI	|DiabetesPedigreeFunction|	Age|	Outcome|
|---|---|---|---|---|---|---|---|---|---|
|count	|2768.000000|	2768.000000|	2768.000000|	2768.000000|	2.768000e+03	2768.000000	2.768000e+03	2768.000000	2768.000000	2768.000000
mean	1384.500000	3.742775	121.890118	72.256430	2.670450e+01	118.628647	3.255548e+01	0.471193	33.132225	0.343931
std	799.197097	3.323801	30.501027	12.007473	1.318295e-12	89.681553	4.121892e-13	0.325669	11.777230	0.475104
min	1.000000	0.000000	44.000000	24.000000	2.670450e+01	14.000000	3.255548e+01	0.078000	21.000000	0.000000
25%	692.750000	1.000000	99.000000	64.000000	2.670450e+01	80.127890	3.255548e+01	0.244000	24.000000	0.000000
50%	1384.500000	3.000000	118.000000	72.000000	2.670450e+01	80.127890	3.255548e+01	0.375000	29.000000	0.000000
75%	2076.250000	6.000000	141.000000	80.000000	2.670450e+01	130.000000	3.255548e+01	0.624000	40.000000	1.000000
max	2768.000000	17.000000	199.000000	122.000000	2.670450e+01	846.000000	3.255548e+01	2.420000	81.0
## Referensi
---
1. Afriani, T., & Pudiyanti, P. (2020). Peran Teknologi Informasi dalam Perawatan Diabetes Mellitus. Nursing Current: Jurnal Keperawatan. https://doi.org/10.19166/NC.V8I1.2722
2. Rahmini, J. A., & Rahayuningtyas, D. K. (2020, Nov). Inovasi Kesehatan Terkini Sebagai Strategi Efektif Pada Manajemen Diabetes Di Masa Pandemi. Jurnal Keperawatan, 5(2). https://doi.org/10.32668/jkep.v5i2.453
3. Ardila, M., S. Humolungo, D. T., Amukti, D. P., & Akrom. (2024, Jun). PROMOSI KESEHATAN PENCEGAHAN DAN PENGENDALIAN DIABETES MELITUS PADA REMAJA. J . A . I : Jurnal Abdimas Indonesia, 4(2). https://doi.org/10.53769/jai.v4i2.729
4. Wahidin, M. Achadi, A. DKK. (2024). Projection of diabetes morbidity and mortality till 2045 in Indonesia based on risk factors and NCD prevention and control programs. Scientific report. https://www.nature.com/articles/s41598-024-54563-2
