# src/04_train_abstractive.py
# (DIMODIFIKASI untuk melatih model kustom)

import torch
import numpy as np
import argparse
import os
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    T5TokenizerFast, 
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# --- IMPOR MODEL KUSTOM BARU ---
from model_abstractive_kustom import T5WithSentencePosition

# --- Variabel Konfigurasi ---
MODEL_NAME = 'panggi/t5-base-indonesian-summarization-cased'
# --- GUNAKAN PATH DATA KUSTOM BARU ---
PROCESSED_TRAIN_PATH = './processed_liputan6_train_abs_custom' 
OUTPUT_DIR = './bert-abstractive-results-custom'

# --- Fungsi Utama Training ---
def main(args):
    print("Memuat dataset abstractive KUSTOM yang sudah diproses...")
    if not os.path.exists(args.data_path):
        print(f"ERROR: Folder data {args.data_path} tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan '03_preprocess_abstractive.py' (versi kustom).")
        return
    processed_dataset = load_from_disk(args.data_path)

    if args.use_sample:
        print("Menggunakan 10% sampel data untuk training cepat.")
        sampled_split = processed_dataset.train_test_split(train_size=0.1, seed=42, shuffle=True)
        sampled_dataset = sampled_split['train']
    else:
        print("Menggunakan 100% data untuk training penuh.")
        sampled_dataset = processed_dataset

    if len(sampled_dataset) < 2:
        print("ERROR: Dataset terlalu kecil.")
        return

    train_test_split = sampled_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    print(f"Jumlah data train: {len(train_dataset)}")
    print(f"Jumlah data eval: {len(eval_dataset)}")

    print(f"Memuat config, tokenizer, dan model KUSTOM T5WithSentencePosition...")
    try:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

        # --- GUNAKAN MODEL KUSTOM BARU ---
        model = T5WithSentencePosition.from_pretrained(MODEL_NAME, config=config)

    except OSError as e:
        print(f"Error memuat model/tokenizer {MODEL_NAME}: {e}")
        return

    # Data Collator tidak berubah
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100
    )

    print("Mempersiapkan Training Arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,

        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs_abs',
        logging_strategy="steps",
        logging_steps=100,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        disable_tqdm=False,
    )

    # Trainer tidak berubah
    # Trainer akan otomatis melewatkan 'sentence_pos_ids' dari batch ke model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("--- MEMULAI TRAINING ABSTRACTIVE KUSTOM ---")
    trainer.train()
    print("--- TRAINING ABSTRACTIVE KUSTOM SELESAI ---")

    best_model_final_path = os.path.join(args.output_dir, "best_model")
    print(f"Menyimpan model abstractive kustom terbaik ke {best_model_final_path}...")
    trainer.save_model(best_model_final_path)
    print("Model berhasil disimpan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih model Abstractive KUSTOM (T5 + Pos) pada dataset Liputan6.")

    parser.add_argument("--data_path", type=str, default=PROCESSED_TRAIN_PATH,
                        help="Path ke data train ABSTRACTIVE KUSTOM (.arrow)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Direktori output untuk hasil training.")
    parser.add_argument("--use_sample", action="store_true",
                        help="Gunakan 10% sampel data.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Jumlah epoch.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU (T5 butuh lebih kecil).")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Frekuensi evaluasi/penyimpanan.")

    args = parser.parse_args()

    if args.use_sample:
        args.output_dir = f"{args.output_dir}_sample"
        print(f"Output akan disimpan di: {args.output_dir}")

    main(args)