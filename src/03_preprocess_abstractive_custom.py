# src/03_preprocess_abstractive.py
# (DIMODIFIKASI untuk membuat sentence_pos_ids)

import pandas as pd
import nltk
import os
import psutil
from datasets import load_dataset
from transformers import T5TokenizerFast
from functools import partial
from typing import List, Dict, Any

# Impor fungsi cleaning dari utils.py
from utils import clean_article_text

# --- Variabel Konfigurasi ---
MODEL_NAME = 'panggi/t5-base-indonesian-summarization-cased' 
MAX_LEN_ARTICLE = 512
MAX_LEN_SUMMARY = 64 

TRAIN_CSV = 'liputan6_dataset_train.csv'
TEST_CSV = 'liputan6_dataset_test.csv'

# Path BARU untuk data yang sudah diproses
PROCESSED_TRAIN_PATH = './processed_liputan6_train_abs_custom'
PROCESSED_TEST_PATH = './processed_liputan6_test_abs_custom'

def _preprocess_batch_abstractive_custom(examples: Dict[str, List[Any]], 
                                        tokenizer: T5TokenizerFast, 
                                        max_len_article: int,
                                        max_len_summary: int) -> Dict[str, List[Any]]:

    inputs_text = examples['clean_article']
    summary_text = [str(s) if s else "" for s in examples['clean_summary']] # Pastikan summary adalah string

    batch_input_ids = []
    batch_attention_mask = []
    batch_sentence_pos_ids = []

    max_sent_pos = 128 # Sesuai dengan model kustom

    # --- LOGIKA PREPROCESSING BARU (Mirip Extractive) ---
    for article_text in inputs_text:
        article_input_ids = []
        article_sent_pos_ids = []

        # 1. Tambahkan prefix T5
        prefix = "ringkas: "
        prefix_tokens = tokenizer(prefix, add_special_tokens=False).input_ids
        article_input_ids.extend(prefix_tokens)
        # Prefix dihitung sbg pos 0
        article_sent_pos_ids.extend([0] * len(prefix_tokens)) 

        # 2. Bersihkan & pecah kalimat
        cleaned_article = clean_article_text(str(article_text)) 
        try:
            sentences = nltk.sent_tokenize(cleaned_article)
        except LookupError:
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(cleaned_article)

        if not sentences:
            sentences = [] # Pastikan sentences adalah list

        current_sent_pos = 0 # Mulai dari 0

        # 3. Tokenisasi kalimat demi kalimat & buat pos_ids
        for i, sentence in enumerate(sentences):
            # Batasi jumlah kalimat
            current_sent_pos = i 
            if i >= max_sent_pos: break 

            tokenized_sent = tokenizer(sentence, add_special_tokens=False).input_ids

            # Cek jika melebihi max_len (sisakan 1 token untuk </s>)
            if (len(article_input_ids) + len(tokenized_sent)) >= max_len_article - 1:
                remaining_space = max_len_article - 1 - len(article_input_ids)
                if remaining_space > 0:
                    article_input_ids.extend(tokenized_sent[:remaining_space])
                    article_sent_pos_ids.extend([i] * remaining_space)
                break 
            else:
                article_input_ids.extend(tokenized_sent)
                article_sent_pos_ids.extend([i] * len(tokenized_sent))

        # 4. Tambahkan token akhir (EOS)
        article_input_ids.append(tokenizer.eos_token_id)
        article_sent_pos_ids.append(current_sent_pos) # Beri pos_id yg sama dgn kalimat terakhir

        # 5. Buat Attention Mask
        attention_mask = [1] * len(article_input_ids)

        # 6. Padding
        padding_length = max_len_article - len(article_input_ids)
        if padding_length > 0:
            article_input_ids.extend([tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            article_sent_pos_ids.extend([0] * padding_length) # Pad pos_ids dgn 0

        batch_input_ids.append(article_input_ids)
        batch_attention_mask.append(attention_mask)
        batch_sentence_pos_ids.append(article_sent_pos_ids)

    # --- AKHIR LOGIKA BARU ---

    # Tokenisasi ringkasan (label decoder) - tidak berubah
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text_target=summary_text,
            max_length=max_len_summary,
            truncation=True,
            padding="max_length"
        )

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": labels['input_ids'],
        "sentence_pos_ids": batch_sentence_pos_ids # <<< Data baru
    }

def run_preprocessing(csv_path: str, save_path: str, tokenizer: T5TokenizerFast):
    """Memuat CSV, memproses data abstractive kustom, dan menyimpan ke disk"""
    print(f"Memuat {csv_path}...")
    try:
        raw_dataset = load_dataset('csv', data_files={'data': csv_path})['data']
    except FileNotFoundError:
        print(f"ERROR: File {csv_path} tidak ditemukan.")
        return

    raw_dataset = raw_dataset.filter(lambda x: x['clean_article'] is not None and x['clean_summary'] is not None)

    if len(raw_dataset) == 0:
        print(f"Tidak ada data valid di {csv_path}.")
        return

    print(f"Memulai preprocessing abstractive KUSTOM untuk {csv_path}...")
    num_cpus = os.cpu_count() or 1
    print(f"Menggunakan {num_cpus} core CPU...")

    preprocess_func = partial(
        _preprocess_batch_abstractive_custom, # Panggil fungsi kustom baru
        tokenizer=tokenizer,
        max_len_article=MAX_LEN_ARTICLE,
        max_len_summary=MAX_LEN_SUMMARY
    )

    processed_dataset = raw_dataset.map(
        preprocess_func,
        batched=True,
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names,
        desc=f"Processing {os.path.basename(csv_path)}"
    )

    print(f"Preprocessing selesai. Menyimpan ke {save_path}...")
    processed_dataset.save_to_disk(save_path)
    print(f"Berhasil disimpan di {save_path}.")

if __name__ == "__main__":
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    print(f"Memuat tokenizer dari {MODEL_NAME}...")
    try:
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    except OSError as e:
        print(f"Error memuat tokenizer {MODEL_NAME}: {e}")
        exit()

    if not os.path.exists(TRAIN_CSV):
        print(f"WARNING: {TRAIN_CSV} tidak ditemukan.")
    elif not os.path.exists(PROCESSED_TRAIN_PATH):
        run_preprocessing(TRAIN_CSV, PROCESSED_TRAIN_PATH, tokenizer)
    else:
        print(f"Output path {PROCESSED_TRAIN_PATH} sudah ada, proses dilewati.")

    if not os.path.exists(TEST_CSV):
        print(f"WARNING: {TEST_CSV} tidak ditemukan.")
    elif not os.path.exists(PROCESSED_TEST_PATH):
        run_preprocessing(TEST_CSV, PROCESSED_TEST_PATH, tokenizer)
    else:
        print(f"Output path {PROCESSED_TEST_PATH} sudah ada, proses dilewati.")

    print("Semua proses preprocessing abstractive KUSTOM selesai.")