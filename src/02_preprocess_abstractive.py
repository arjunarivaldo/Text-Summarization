# src/03_preprocess_abstractive.py
import pandas as pd
import nltk
import os
import psutil
from datasets import load_dataset
from transformers import T5TokenizerFast # Menggunakan T5
from functools import partial
from typing import List, Dict, Any

# --- Variabel Konfigurasi ---
MODEL_NAME = 'panggi/t5-base-indonesian-summarization-cased'
MAX_LEN_ARTICLE = 512
MAX_LEN_SUMMARY = 64 

TRAIN_CSV = 'liputan6_dataset_train.csv'
TEST_CSV = 'liputan6_dataset_test.csv'

PROCESSED_TRAIN_PATH = './processed_liputan6_train_abs'
PROCESSED_TEST_PATH = './processed_liputan6_test_abs'

def _preprocess_batch_abstractive(
    examples: Dict[str, List[Any]], 
    tokenizer: T5TokenizerFast, 
    max_len_article: int,
    max_len_summary: int) -> Dict[str, List[Any]]:
    
    # Preprocessing untuk T5 perlu prefix
    prefix = "ringkas: "
    inputs_text = [prefix + doc for doc in examples['clean_article']]
    
    inputs = tokenizer(
        inputs_text,
        max_length=max_len_article,
        truncation=True,
        padding="max_length"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['clean_summary'],
            max_length=max_len_summary,
            truncation=True,
            padding="max_length"
        )
    
    inputs['labels'] = labels['input_ids']
    return inputs

def run_preprocessing(csv_path: str, save_path: str, tokenizer: T5TokenizerFast):
    print(f"Memuat {csv_path}...")
    raw_dataset = load_dataset('csv', data_files={'data': csv_path})['data']
    raw_dataset = raw_dataset.filter(lambda x: x['clean_article'] is not None and x['clean_summary'] is not None)

    print(f"Memulai preprocessing abstractive untuk {csv_path}...")
    num_cpus = os.cpu_count()
    print(f"Menggunakan {num_cpus} core CPU...")
    
    preprocess_func = partial(
        _preprocess_batch_abstractive,
        tokenizer=tokenizer,
        max_len_article=MAX_LEN_ARTICLE,
        max_len_summary=MAX_LEN_SUMMARY
    )
    
    processed_dataset = raw_dataset.map(
        preprocess_func,
        batched=True,
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names
    )
    
    print(f"Preprocessing selesai. Menyimpan ke {save_path}...")
    processed_dataset.save_to_disk(save_path)
    print(f"Berhasil disimpan di {save_path}.")

if __name__ == "__main__":
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    print(f"Memuat tokenizer dari {MODEL_NAME}...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        run_preprocessing(TRAIN_CSV, PROCESSED_TRAIN_PATH, tokenizer)
    else:
        print(f"Output path {PROCESSED_TRAIN_PATH} sudah ada, proses dilewati.")
        
    if not os.path.exists(PROCESSED_TEST_PATH):
        run_preprocessing(TEST_CSV, PROCESSED_TEST_PATH, tokenizer)
    else:
        print(f"Output path {PROCESSED_TEST_PATH} sudah ada, proses dilewati.")
        
    print("Semua proses preprocessing abstractive selesai.")