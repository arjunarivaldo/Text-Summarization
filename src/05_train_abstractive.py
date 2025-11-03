# src/04_train_abstractive.py
import torch
import numpy as np
import argparse
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    T5TokenizerFast,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# --- Variabel Konfigurasi ---
MODEL_NAME = 'panggi/t5-base-indonesian-summarization-cased'
PROCESSED_TRAIN_PATH = './processed_liputan6_train_abs' # Path data abstractive
OUTPUT_DIR = './bert-abstractive-results'
BEST_MODEL_DIR = './bert-abstractive-best-model' # Path model BARU

# --- Fungsi Utama Training ---
def main(args):
    print("Memuat dataset abstractive yang sudah diproses...")
    processed_dataset = load_from_disk(args.data_path)
    
    if args.use_sample:
        print("Menggunakan 10% sampel data untuk training cepat.")
        sampled_split = processed_dataset.train_test_split(train_size=0.1, seed=42, shuffle=True)
        sampled_dataset = sampled_split['train']
    else:
        print("Menggunakan 100% data untuk training penuh.")
        sampled_dataset = processed_dataset

    print("Membuat split train/validation (90/10)...")
    train_test_split = sampled_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    print(f"Jumlah data train: {len(train_dataset)}")
    print(f"Jumlah data eval: {len(eval_dataset)}")
    
    print("Memuat config, tokenizer, dan model Seq2Seq...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, config=config)
    
    # Data Collator sangat PENTING untuk Seq2Seq
    # Ini akan membuat label padding (-100) secara dinamis
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100 # Standar untuk mengabaikan loss pada padding
    )
    
    print("Mempersiapkan Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_abs',
        logging_strategy="steps", # Ubah ke 'epoch' jika output berantakan
        logging_steps=100,
        
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        
        load_best_model_at_end=True,
        metric_for_best_model="loss", # Monitor 'loss' krn ROUGE lambat di-eval
        greater_is_better=False,     # 'loss' lebih baik jika lebih kecil
        
        fp16=True,  # Mengoptimalkan A100
        report_to="tensorboard",
        disable_tqdm=True
    )

    # Gunakan Trainer standar, bukan CustomTrainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # Tidak pakai compute_metrics saat training
        # krn 'generate' teks lambat. Evaluasi ROUGE di skrip terpisah.
    )

    print("--- MEMULAI TRAINING ABSTRACTIVE ---")
    trainer.train()
    print("--- TRAINING ABSTRACTIVE SELESAI ---")
    
    print(f"Menyimpan model abstractive terbaik ke {BEST_MODEL_DIR}...")
    trainer.save_model(BEST_MODEL_DIR)
    print("Model berhasil disimpan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih model Abstractive (IndoBART) pada dataset Liputan6.")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=PROCESSED_TRAIN_PATH,
        help="Path ke folder data train ABSTRACTIVE (.arrow)"
    )
    parser.add_argument(
        "--use_sample",
        action="store_true",
        help="Gunakan 10% sampel data untuk lari cepat."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Jumlah epoch training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (A100 bisa 16-32).")
    parser.add_argument("--eval_steps", type=int, default=500, help="Jumlah step per evaluasi/penyimpanan.")
    
    args = parser.parse_args()
    main(args)