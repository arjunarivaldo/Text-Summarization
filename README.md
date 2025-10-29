# Proyek 2: Summarization Extractive vs. Abstractive

Proyek ini membangun dan membandingkan dua arsitektur model Text Summarization pada dataset Liputan6:

1.  **Extractive (Kustom):** Model berbasis `indobert-base-p1` dengan _Sentence Position Embeddings_ kustom untuk memilih kalimat.
2.  **Abstractive (T5):** Model berbasis `panggi/t5-base-indonesian-summarization-cased` yang di-fine-tune untuk menghasilkan ringkasan baru.

Eksperimen ini dijalankan menggunakan **10% dari total data training** untuk iterasi cepat.

## Arsitektur

- **`src/`**: Berisi semua kode sumber yang dapat dieksekusi (`.py`).
- **`notebooks/`**: Berisi notebook Analisis Data Eksploratif (`.ipynb`).

## Hasil Akhir (pada 10% Sampel Test Set)

| Model                       | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :-------------------------- | :-----: | :-----: | :-----: |
| **Model Abstractive (T5)**  |  _TBD_  |  _TBD_  |  _TBD_  |
| Model Extractive (BERT k=2) |  38.29  |  21.01  |  30.82  |
| Baseline (Lead-2)           |  37.87  |  20.77  |  30.41  |

_(Catatan: Hasil T5 akan terisi setelah langkah 2 & 3 dijalankan)_

## Cara Menjalankan Proyek

1.  **Clone dan Setup**

    ```bash
    git clone [https://github.com/USERNAME_ANDA/Text-Summarization.git](https://github.com/USERNAME_ANDA/Text-Summarization.git)
    cd Text-Summarization
    pip install -r requirements.txt
    ```

    _Unggah `liputan6_dataset_train.csv` dan `liputan6_dataset_test.csv` ke dalam folder `Text-Summarization/`._

2.  **Langkah 1: Jalankan Semua Preprocessing**

    ```bash
    # (Ini akan membuat folder 'processed_liputan6_train' & 'processed_liputan6_test')
    python src/01_preprocess_extractive.py

    # (Ini akan membuat folder 'processed_liputan6_train_abs' & 'processed_liputan6_test_abs')
    python src/03_preprocess_abstractive.py
    ```

3.  **Langkah 2: Jalankan Semua Training (Gunakan GPU)**
    _Gunakan `--use_sample` untuk berlatih dengan 10% data._

    ```bash
    # (Ini akan membuat folder 'bert-summarizer-results_sample/best_model')
    python src/02_train_extractive.py --use_sample --eval_steps 100

    # (Ini akan membuat folder 'bert-abstractive-results_sample/best_model')
    python src/04_train_abstractive.py --use_sample --eval_steps 100 --batch_size 8
    ```

4.  **Langkah 3: Jalankan Evaluasi Komparatif**
    _Gunakan `--use_sample` untuk evaluasi cepat pada 10% test set._
    _Pastikan path model di argumen default skrip sesuai dengan output training (misalnya, `bert-summarizer-results_sample/best_model`)._
    ```bash
    # (Skrip ini memuat kedua model dan baseline untuk perbandingan)
    python src/05_evaluate_comparison.py --use_sample
    ```
