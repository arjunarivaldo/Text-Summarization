# 02_train.py
import torch
import numpy as np
import argparse
from datasets import load_from_disk
from transformers import (
    BertConfig,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import f1_score

# Impor arsitektur model kustom
from model_architecture import BertSumClassifier

# --- Variabel Konfigurasi ---
MODEL_NAME = 'indobenchmark/indobert-base-p1'
PROCESSED_TRAIN_PATH = 'processed_liputan6_train'
OUTPUT_DIR = './bert-summarizer-results'
BEST_MODEL_DIR = './bert-summarizer-best-model'

# (EDA 3) Bobot kelas untuk 5.71 : 1 imbalance
CLASS_WEIGHT_RATIO = 5.71 
# Pindahkan ke device ('cuda') di dalam Trainer
class_weights = torch.tensor([1.0, CLASS_WEIGHT_RATIO])


# --- Fungsi Helper ---

def compute_metrics(eval_pred):
    """(EDA 4) Menghitung F1-Score untuk kelas '1'"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    true_labels = labels.flatten()
    true_predictions = predictions.flatten()
    
    # Filter 'ignore_index' (-100)
    active_indices = true_labels != -100
    true_labels = true_labels[active_indices]
    true_predictions = true_predictions[active_indices]
    
    if len(true_labels) == 0:
        return {"f1": 0.0}

    f1 = f1_score(true_labels, true_predictions, pos_label=1, average='binary')
    return {"f1": f1}

class CustomTrainer(Trainer):
    """(EDA 3) Custom Trainer untuk menerapkan Class Weighting"""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Pindahkan bobot ke device yang sama dengan model
        weights = class_weights.to(model.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# --- Fungsi Utama Training ---

def main(args):
    print("Memuat dataset yang sudah diproses...")
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
    
    print("Memuat config dan model kustom...")
    config = BertConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2 
    )
    model = BertSumClassifier.from_pretrained(MODEL_NAME, config=config)
    
    print("Mempersiapkan Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        disable_tqdm=True,
        
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        fp16=True,  # Mengoptimalkan Tensor Cores
        report_to="tensorboard"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("--- MEMULAI TRAINING ---")
    trainer.train()
    print("--- TRAINING SELESAI ---")
    
    print(f"Menyimpan model terbaik ke {BEST_MODEL_DIR}...")
    trainer.save_model(BEST_MODEL_DIR)
    print("Model berhasil disimpan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih model BertSum pada dataset Liputan6.")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=PROCESSED_TRAIN_PATH,
        help="Path ke folder data train yang sudah diproses (.arrow)"
    )
    parser.add_argument(
        "--use_sample",
        action="store_true",
        help="Gunakan 10% sampel data untuk lari cepat."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Jumlah epoch training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device (A100 bisa 16 atau 32).")
    parser.add_argument("--eval_steps", type=int, default=500, help="Jumlah step per evaluasi/penyimpanan.")
    
    args = parser.parse_args()
    main(args)