# 01_preprocess.py
import pandas as pd
import nltk
import ast
import os
import psutil
from datasets import load_dataset, Dataset
from transformers import BertTokenizer
from functools import partial
from typing import List, Dict, Any

# Impor fungsi cleaning dari utils.py
from utils import clean_article_text

# --- Variabel Konfigurasi ---
MODEL_NAME = 'indobenchmark/indobert-base-p1'
MAX_LEN = 512
TRAIN_CSV = 'liputan6_dataset_train.csv'
TEST_CSV = 'liputan6_dataset_test.csv'

# Path output
PROCESSED_TRAIN_PATH = './processed_liputan6_train'
PROCESSED_TEST_PATH = './processed_liputan6_test'

def _preprocess_batch(examples: Dict[str, List[Any]], 
                    tokenizer: BertTokenizer, 
                    max_len: int) -> Dict[str, List[Any]]:
    """
    Fungsi yang memproses batch data, dioptimalkan untuk datasets.map()
    """
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id

    batch_input_ids = []
    batch_token_type_ids = []
    batch_attention_mask = []
    batch_sentence_pos_ids = []
    batch_labels = []

    for article_text, summary_str in zip(examples['clean_article'], examples['extractive_summary']):
        
        # 1. Cleaning (EDA 4)
        cleaned_article = clean_article_text(article_text)
        
        try:
            sentences = nltk.sent_tokenize(cleaned_article)
        except LookupError:
            print("Mengunduh tokenizer NLTK 'punkt_tab'...")
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(cleaned_article)

        if not sentences:
            continue

        try:
            true_indices = set(ast.literal_eval(str(summary_str)))
        except:
            true_indices = set()

        article_input_ids = []
        article_token_type_ids = []
        article_sent_pos_ids = []
        article_labels = []

        segment_id = 0
        for i, sentence in enumerate(sentences):
            tokenized_sent = tokenizer.encode(sentence, add_special_tokens=False)
            tokens_to_add = [cls_token_id] + tokenized_sent + [sep_token_id]
            types_to_add = [segment_id] * len(tokens_to_add)
            pos_to_add = [i] * len(tokens_to_add) # (EDA 2)
            label = 1 if i in true_indices else 0

            if (len(article_input_ids) + len(tokens_to_add)) > max_len:
                break
            
            article_input_ids.extend(tokens_to_add)
            article_token_type_ids.extend(types_to_add)
            article_sent_pos_ids.extend(pos_to_add)
            article_labels.append(label)
            segment_id = 1 - segment_id
        
        # Padding
        padding_length = max_len - len(article_input_ids)
        attention_mask = [1] * len(article_input_ids)
        
        if padding_length > 0:
            article_input_ids.extend([pad_token_id] * padding_length)
            article_token_type_ids.extend([0] * padding_length)
            article_sent_pos_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
            
        # Buat label akhir (-100 = ignore index)
        final_labels = [-100] * max_len 
        cls_indices = [i for i, token in enumerate(article_input_ids) if token == cls_token_id]
        
        for i, cls_index in enumerate(cls_indices):
            if i < len(article_labels):
                final_labels[cls_index] = article_labels[i]

        batch_input_ids.append(article_input_ids)
        batch_token_type_ids.append(article_token_type_ids)
        batch_attention_mask.append(attention_mask)
        batch_sentence_pos_ids.append(article_sent_pos_ids)
        batch_labels.append(final_labels)

    return {
        "input_ids": batch_input_ids,
        "token_type_ids": batch_token_type_ids,
        "attention_mask": batch_attention_mask,
        "sentence_pos_ids": batch_sentence_pos_ids,
        "labels": batch_labels,
    }

def run_preprocessing(csv_path: str, save_path: str, tokenizer: BertTokenizer, max_len: int):
    """Memuat CSV, memproses, dan menyimpan ke disk"""
    print(f"Memuat {csv_path}...")
    raw_dataset = load_dataset('csv', data_files={'data': csv_path})['data']
    
    # Hapus baris yang kosong
    raw_dataset = raw_dataset.filter(lambda x: x['clean_article'] is not None and x['extractive_summary'] is not None)

    print(f"Memulai preprocessing untuk {csv_path}...")
    num_cpus = os.cpu_count()
    print(f"Menggunakan {num_cpus} core CPU...")
    
    # 'partial' untuk "membekukan" argumen tokenizer dan max_len
    preprocess_with_tokenizer = partial(_preprocess_batch, 
                                        tokenizer=tokenizer, 
                                        max_len=max_len)
    
    processed_dataset = raw_dataset.map(
        preprocess_with_tokenizer,
        batched=True,
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names
    )
    
    print(f"Preprocessing selesai. Menyimpan ke {save_path}...")
    processed_dataset.save_to_disk(save_path)
    print(f"Berhasil disimpan di {save_path}.")

if __name__ == "__main__":
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    print("Memuat tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Proses Train Set
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        run_preprocessing(TRAIN_CSV, PROCESSED_TRAIN_PATH, tokenizer, MAX_LEN)
    else:
        print(f"Output path {PROCESSED_TRAIN_PATH} sudah ada, proses dilewati.")
        
    # Proses Test Set
    if not os.path.exists(PROCESSED_TEST_PATH):
        run_preprocessing(TEST_CSV, PROCESSED_TEST_PATH, tokenizer, MAX_LEN)
    else:
        print(f"Output path {PROCESSED_TEST_PATH} sudah ada, proses dilewati.")
        
    print("Semua proses preprocessing selesai.")