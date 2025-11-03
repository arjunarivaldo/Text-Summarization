# Project 2: Text Summarization

Proyek ini membangun dan membandingkan tiga arsitektur model Text Summarization (dan satu baseline) pada dataset Liputan6 untuk menemukan metode peringkasan terbaik.

1.  **Extractive (Kustom):** Model berbasis `indobert-base-p1` dengan _Sentence Position Embeddings_ kustom.
2.  **Abstractive (Standar):** Model T5 standar (`panggi/t5-base-indonesian-summarization-cased`).
3.  **Abstractive (Kustom):** Model T5 yang dimodifikasi dengan _Sentence Position Embeddings_.

Seluruh eksperimen dijalankan menggunakan **10% dari total data training dan 10% dari data tes** untuk iterasi cepat.

## Arsitektur

- **`src/`**: Berisi semua kode sumber Python (`.py`) yang dapat dieksekusi untuk _preprocessing_, _training_, dan _evaluasi_.
- **`notebooks/`**: Berisi _notebook_ Analisis Data Eksploratif (EDA) (`.ipynb`) yang menjadi dasar keputusan desain model.

## Hasil Akhir (pada 10% Sampel Test Set)

Perbandingan skor ROUGE-2 (kesamaan frasa) dan BERTScore (kesamaan makna).

| Model                        |  ROUGE-2  | BERTScore-F1 |
| :--------------------------- | :-------: | :----------: |
| **Abstractive (T5 Standar)** | **22.61** |  **77.30**   |
| Abstractive (T5 Kustom)      |   22.39   |    77.21     |
| Extractive (BERT Kustom)     |   21.67   |    76.13     |
| Baseline (Lead-2)            |   21.43   |    76.01     |

## Cara Menjalankan Proyek

1.  **Clone dan Setup**

    ```bash
    git clone [https://github.com/arjunarivaldo/Text-Summarization.git](https://github.com/arjunarivaldo/Text-Summarization.git)
    cd Text-Summarization
    pip install -r requirements.txt
    ```

    _Unggah `liputan6_dataset_train.csv` dan `liputan6_dataset_test.csv` ke dalam folder `Text-Summarization/`._

2.  **Langkah 1: Jalankan Semua Preprocessing (Urut)**
    _(Perintah ini akan membuat 6 folder data .arrow: `processed_...`)_

    ```bash
    python src/01_preprocess_extractive.py
    python src/02_preprocess_abstractive.py
    python src/03_preprocess_abstractive_custom.py
    ```

3.  **Langkah 2: Jalankan Semua Training (Gunakan GPU & Urut)**
    _(Gunakan `--use_sample` untuk berlatih dengan 10% data)_

    ```bash
    # (Ini akan membuat folder 'bert-summarizer-results_sample/best_model')
    python src/04_train_extractive.py --use_sample --eval_steps 100

    # (Ini akan membuat folder 'bert-abstractive-results_sample/best_model')
    python src/05_train_abstractive.py --use_sample --eval_steps 100 --batch_size 8

    # (Ini akan membuat folder 'bert-abstractive-results-custom_sample/best_model')
    python src/06_train_abstractive_custom.py --use_sample --eval_steps 100 --batch_size 8
    ```

4.  **Langkah 3: Jalankan Evaluasi Komparatif**
    _(Gunakan `--use_sample` untuk evaluasi cepat pada 10% test set)_
    ```bash
    # (Skrip ini memuat ke-3 model & baseline untuk perbandingan akhir)
    python src/07_evaluate_comparison.py --use_sample
    ```
