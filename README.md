# Project 2: Extractive Text Summarization dengan IndoBERT

Proyek ini adalah implementasi dari model extractive summarization menggunakan arsitektur BERT kustom pada dataset Liputan6.

Tujuan utama dari eksperimen ini adalah untuk membuktikan bahwa model BERT yang telah di-fine-tune dapat mengalahkan baseline "Lead-2" yang kuat pada metrik ROUGE.

**Eksperimen ini menggunakan 10% dari total data training.**

## Arsitektur Model

Model ini menggunakan `indobert-base-p1` sebagai dasarnya. Arsitektur kustom (`BertSumClassifier`) telah dibuat untuk menambahkan **Sentence Position Embeddings** (berdasarkan temuan EDA) di atas token embeddings standar BERT.

## Hasil Akhir (pada 10% Data Test Set)

Model BERT kustom (k=2) berhasil mengalahkan baseline terkuat (Lead-2).

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :--- | :---: | :---: | :---: |
| **Model BERT (k=2)** | **38.62** | **21.45** | **31.28** |
| Baseline (Lead-2) | 38.29 | 21.25 | 30.98 |

## Cara Menjalankan Proyek

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Langkah 1: Preprocessing Data**
    (Langkah ini akan memproses `liputan6_dataset_train.csv` dan `liputan6_dataset_test.csv` menjadi format `.arrow`)
    ```bash
    python 01_preprocess.py
    ```

3.  **Langkah 2: Training Model**
    (Skrip ini akan melatih model pada 10% data dan menyimpannya ke `./bert-summarizer-best-model`)
    ```bash
    python 02_train.py --use_sample --eval_steps 100
    ```

4.  **Langkah 3: Evaluasi Model**
    (Skrip ini akan memuat model terbaik dan mengevaluasinya pada test set)
    ```bash
    python 03_evaluate.py
    ```
