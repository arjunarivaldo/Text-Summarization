# src/05_evaluate_comparison.py
# (DIMODIFIKASI untuk mengevaluasi 3 model + baseline)

import torch
import nltk
import evaluate
import argparse
import os
from datasets import load_dataset
from transformers import (
    BertTokenizer, 
    T5TokenizerFast, 
    AutoModelForSeq2SeqLM
)
from tqdm.auto import tqdm
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Impor SEMUA arsitektur model
from model_architecture import BertSumClassifier
from model_abstractive_kustom import T5WithSentencePosition
from utils import clean_article_text, generate_baseline_summary

# --- Konfigurasi ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path default ke model-model terbaik Anda
DEFAULT_EXTRACTIVE_PATH = './bert-summarizer-best-model'
DEFAULT_ABSTRACTIVE_PATH = './bert-abstractive-best-model'
# Path BARU untuk model kustom
DEFAULT_ABSTRACTIVE_CUSTOM_PATH = './bert-abstractive-results-custom_sample/best_model'
DEFAULT_TEST_FILE = "liputan6_dataset_test.csv"

# Tokenizer untuk masing-masing model
EXTRACTIVE_TOKENIZER_NAME = 'indobenchmark/indobert-base-p1'
ABSTRACTIVE_TOKENIZER_NAME = 'panggi/t5-base-indonesian-summarization-cased'

EXTRACTIVE_K = 2 
BASELINE_K = 2 

# --- Fungsi Prediksi Extractive (Tidak Berubah) ---
def predict_extractive_summary(article_text: str, model, tokenizer, k: int) -> str:
    # ... (Salin/Tempel fungsi lengkap dari file 05_evaluate_abstractive.py Anda)
    # (Pastikan NLTK.download('punkt') ada di 'except')
    if not isinstance(article_text, str) or not article_text: return ""
    cleaned_article = clean_article_text(article_text)
    try:
        sentences = nltk.sent_tokenize(cleaned_article)
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        sentences = nltk.sent_tokenize(cleaned_article)
    if not sentences: return ""
    cls_id, sep_id, max_len = tokenizer.cls_token_id, tokenizer.sep_token_id, 512
    input_ids, token_type_ids, sent_pos_ids = [], [], []
    segment_id = 0
    for i, sent in enumerate(sentences):
        if i >= 128: break 
        tokenized_sent = tokenizer.encode(sent, add_special_tokens=False)
        tokens_to_add = [cls_id] + tokenized_sent + [sep_id]
        types_to_add = [segment_id] * len(tokens_to_add)
        pos_to_add = [i] * len(tokens_to_add)
        if (len(input_ids) + len(tokens_to_add)) > max_len: break
        input_ids.extend(tokens_to_add); token_type_ids.extend(types_to_add); sent_pos_ids.extend(pos_to_add)
        segment_id = 1 - segment_id
    if not input_ids: return ""

    current_len = len(input_ids)
    if current_len < max_len:
        padding_needed = max_len - current_len
        input_ids.extend([tokenizer.pad_token_id] * padding_needed)
        token_type_ids.extend([0] * padding_needed)
        sent_pos_ids.extend([0] * padding_needed)

    input_ids = torch.tensor([input_ids[:max_len]]).to(DEVICE)
    token_type_ids = torch.tensor([token_type_ids[:max_len]]).to(DEVICE)
    sent_pos_ids = torch.tensor([sent_pos_ids[:max_len]]).to(DEVICE)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, sentence_pos_ids=sent_pos_ids)
        logits = outputs.logits

    cls_indices = (input_ids[0] == cls_id).nonzero(as_tuple=True)[0]
    if len(cls_indices) == 0: return "" 

    probs = torch.softmax(logits[0, cls_indices], dim=1)[:, 1]

    actual_k = min(k, len(probs))
    if actual_k <= 0: return ""

    top_k_indices = torch.topk(probs, k=actual_k).indices
    sorted_indices = sorted([idx.item() for idx in top_k_indices if idx.item() < len(sentences)])

    final_sentences = []
    seen_indices = set()
    for idx in sorted_indices:
        if idx not in seen_indices:
            final_sentences.append(sentences[idx])
            seen_indices.add(idx)

    return " ".join(final_sentences)

# --- Fungsi Prediksi Abstractive Standar (Tidak Berubah) ---
def predict_abstractive_summary(article_text: str, model, tokenizer) -> str:
    # ... (Salin/Tempel fungsi lengkap dari file 05_evaluate_abstractive.py Anda)
    # (Pastikan prefix "ringkas: " ada)
    if not isinstance(article_text, str) or not article_text: return ""
    prefix = "ringkas: "
    inputs_text = prefix + article_text
    inputs = tokenizer(
        inputs_text, max_length=512, truncation=True, return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs, num_beams=4, max_length=64, early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# --- FUNGSI PREDIKSI BARU UNTUK MODEL KUSTOM ---
def predict_abstractive_custom_summary(article_text: str, model, tokenizer) -> str:
    """
    Menghasilkan ringkasan menggunakan model T5 KUSTOM,
    termasuk preprocessing on-the-fly untuk sentence_pos_ids.
    """
    if not isinstance(article_text, str) or not article_text: return ""

    max_len_article = 512
    max_sent_pos = 128

    article_input_ids = []
    article_sent_pos_ids = []

    # 1. Tambahkan prefix T5
    prefix = "ringkas: "
    prefix_tokens = tokenizer(prefix, add_special_tokens=False).input_ids
    article_input_ids.extend(prefix_tokens)
    article_sent_pos_ids.extend([0] * len(prefix_tokens))

    # 2. Bersihkan & pecah kalimat
    cleaned_article = clean_article_text(str(article_text))
    try:
        sentences = nltk.sent_tokenize(cleaned_article)
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
        sentences = nltk.sent_tokenize(cleaned_article)
    if not sentences: sentences = []

    current_sent_pos = 0
    # 3. Tokenisasi kalimat demi kalimat & buat pos_ids
    for i, sentence in enumerate(sentences):
        current_sent_pos = i
        if i >= max_sent_pos: break
        tokenized_sent = tokenizer(sentence, add_special_tokens=False).input_ids
        if (len(article_input_ids) + len(tokenized_sent)) >= max_len_article - 1:
            remaining_space = max_len_article - 1 - len(article_input_ids)
            if remaining_space > 0:
                article_input_ids.extend(tokenized_sent[:remaining_space])
                article_sent_pos_ids.extend([i] * remaining_space)
            break
        else:
            article_input_ids.extend(tokenized_sent)
            article_sent_pos_ids.extend([i] * len(tokenized_sent))

    article_input_ids.append(tokenizer.eos_token_id)
    article_sent_pos_ids.append(current_sent_pos)

    # 4. Buat tensor (tanpa padding, .generate() akan menanganinya)
    input_ids = torch.tensor([article_input_ids]).to(DEVICE)
    attention_mask = torch.ones_like(input_ids) # Mask 1 untuk semua token asli
    sent_pos_ids = torch.tensor([article_sent_pos_ids]).to(DEVICE)

    # 5. --- TRIK GENERASI KUSTOM ---
    # Kita harus menjalankan encoder secara manual untuk memasukkan `sentence_pos_ids`
    with torch.no_grad():
        # Dapatkan embedding
        word_embeddings = model.encoder.embed_tokens(input_ids)
        sent_pos_embeds = model.sent_pos_embeddings(torch.clamp(sent_pos_ids, max=max_sent_pos - 1))
        # Gabungkan embedding
        combined_embeddings = word_embeddings + sent_pos_embeds

        # Jalankan encoder secara manual
        encoder_outputs = model.encoder(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 6. Panggil .generate() dengan encoder_outputs yang sudah dikustomisasi
        summary_ids = model.generate(
            encoder_outputs=encoder_outputs, # <<< Gunakan output kustom
            attention_mask=attention_mask,   # <<< Teruskan mask
            num_beams=4,
            max_length=64,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def main(args):
    print(f"Menggunakan device: {DEVICE}")

    # --- 1. Muat Model-Model ---
    print(f"Memuat model Extractive dari {args.extractive_path}...")
    model_extractive = BertSumClassifier.from_pretrained(args.extractive_path).to(DEVICE)
    tokenizer_extractive = BertTokenizer.from_pretrained(EXTRACTIVE_TOKENIZER_NAME)
    model_extractive.eval()

    print(f"Memuat model Abstractive (Standar) dari {args.abstractive_path}...")
    model_abstractive = AutoModelForSeq2SeqLM.from_pretrained(args.abstractive_path).to(DEVICE)
    tokenizer_abstractive = T5TokenizerFast.from_pretrained(ABSTRACTIVE_TOKENIZER_NAME)
    model_abstractive.eval()

    # --- MUAT MODEL KUSTOM BARU ---
    print(f"Memuat model Abstractive (Kustom) dari {args.abstractive_custom_path}...")
    model_abstractive_custom = T5WithSentencePosition.from_pretrained(args.abstractive_custom_path).to(DEVICE)
    # Tokenizer-nya sama dengan abstractive standar
    tokenizer_abstractive_custom = T5TokenizerFast.from_pretrained(ABSTRACTIVE_TOKENIZER_NAME) 
    model_abstractive_custom.eval()

    # ... (Kode muat test_dataset & handle sampling SAMA) ...
    print(f"Memuat test set dari {args.test_file}...")
    full_test_dataset = load_dataset('csv', data_files={'test': args.test_file})['test']
    full_test_dataset = full_test_dataset.filter(lambda x: x['clean_article'] is not None and x['clean_summary'] is not None)

    if args.use_sample:
        sample_size = int(0.1 * len(full_test_dataset))
        print(f"Menggunakan 10% sampel ({sample_size} entri) dari test set...")
        test_dataset = full_test_dataset.shuffle(seed=42).select(range(sample_size))
    else:
        test_dataset = full_test_dataset
        print(f"Menggunakan seluruh test set ({len(test_dataset)} entri)...")

    print("Memuat metrik ROUGE dan BERTScore...")
    rouge_metric = evaluate.load('rouge')
    bertscore_metric = evaluate.load('bertscore')

    preds_extractive, preds_abstractive, preds_abstractive_custom, preds_baseline, references = [], [], [], [], []

    print(f"Menjalankan prediksi komparatif pada {len(test_dataset)} artikel...")
    for example in tqdm(test_dataset, desc="Evaluating"):
        article = example['clean_article']
        reference = example['clean_summary']
        if not isinstance(article, str) or not isinstance(reference, str): continue

        preds_extractive.append(predict_extractive_summary(article, model_extractive, tokenizer_extractive, k=EXTRACTIVE_K))
        preds_abstractive.append(predict_abstractive_summary(article, model_abstractive, tokenizer_abstractive))
        # Panggil fungsi prediksi kustom baru
        preds_abstractive_custom.append(predict_abstractive_custom_summary(article, model_abstractive_custom, tokenizer_abstractive_custom))
        preds_baseline.append(generate_baseline_summary(article, k=BASELINE_K))
        references.append(reference)

    print("Menghitung skor ROUGE akhir...")
    rouge_extractive = rouge_metric.compute(predictions=preds_extractive, references=references, use_stemmer=True)
    rouge_abstractive = rouge_metric.compute(predictions=preds_abstractive, references=references, use_stemmer=True)
    rouge_abstractive_custom = rouge_metric.compute(predictions=preds_abstractive_custom, references=references, use_stemmer=True)
    rouge_baseline = rouge_metric.compute(predictions=preds_baseline, references=references, use_stemmer=True)

    print("Menghitung skor BERTScore akhir...")
    bertscore_extractive = bertscore_metric.compute(predictions=preds_extractive, references=references, lang="id")
    bertscore_abstractive = bertscore_metric.compute(predictions=preds_abstractive, references=references, lang="id")
    bertscore_abstractive_custom = bertscore_metric.compute(predictions=preds_abstractive_custom, references=references, lang="id")
    bertscore_baseline = bertscore_metric.compute(predictions=preds_baseline, references=references, lang="id")

    # Ambil rata-rata F1
    bs_extractive_f1 = np.mean(bertscore_extractive['f1'])
    bs_abstractive_f1 = np.mean(bertscore_abstractive['f1'])
    bs_abstractive_custom_f1 = np.mean(bertscore_abstractive_custom['f1'])
    bs_baseline_f1 = np.mean(bertscore_baseline['f1'])

    print("\n" + "="*50)
    print("--- SKOR AKHIR KOMPARATIF PADA TEST SET ---")
    if args.use_sample:
        print("--- (DIEVALUASI PADA 10% SAMPEL TEST SET) ---")
    print("="*50)

    def print_scores(model_name, rouge_scores, bertscore_f1):
        print(f"\n--- {model_name} ---")
        print(f"  ROUGE-1: {rouge_scores['rouge1'] * 100:.2f} | ROUGE-2: {rouge_scores['rouge2'] * 100:.2f} | ROUGE-L: {rouge_scores['rougeLsum'] * 100:.2f}")
        print(f"  BERTScore-F1: {bertscore_f1 * 100:.2f}")

    print_scores(f"Baseline (Lead-{BASELINE_K})", rouge_baseline, bs_baseline_f1)
    print_scores(f"Model Extractive (k={EXTRACTIVE_K})", rouge_extractive, bs_extractive_f1)
    print_scores("Model Abstractive (T5 Standar)", rouge_abstractive, bs_abstractive_f1)
    print_scores("Model Abstractive (T5 Kustom)", rouge_abstractive_custom, bs_abstractive_custom_f1)

    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi komparatif model summarization.")
    parser.add_argument("--extractive_path", type=str, default=DEFAULT_EXTRACTIVE_PATH)
    parser.add_argument("--abstractive_path", type=str, default=DEFAULT_ABSTRACTIVE_PATH)
    # Tambahkan argumen untuk model kustom baru
    parser.add_argument("--abstractive_custom_path", type=str, default=DEFAULT_ABSTRACTIVE_CUSTOM_PATH)
    parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument("--use_sample", action="store_true", help="Gunakan 10% sampel test set.")

    args = parser.parse_args()
    main(args)