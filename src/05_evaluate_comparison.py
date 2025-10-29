# src/05_evaluate_abstractive.py
import torch
import nltk
import evaluate
import argparse
from datasets import load_dataset
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import warnings

# Menonaktifkan peringatan tokenisasi yang tidak perlu
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.tokenization_utils_base')

# Impor model Extractive dan utilitas
from model_architecture import BertSumClassifier
from transformers import BertTokenizer
from utils import clean_article_text, generate_baseline_summary

# --- Konfigurasi ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path ke model-model terbaik Anda
EXTRACTIVE_MODEL_PATH = './bert-summarizer-best-model'
ABSTRACTIVE_MODEL_PATH = './bert-abstractive-best-model'

# Tokenizer untuk masing-masing model
EXTRACTIVE_TOKENIZER_NAME = 'indobenchmark/indobert-base-p1'
ABSTRACTIVE_TOKENIZER_NAME = 'panggi/t5-base-indonesian-summarization-cased'

# Strategi (dari EDA)
EXTRACTIVE_K = 2 
BASELINE_K = 2

# --- Fungsi Prediksi ---
def predict_extractive_summary(article_text: str, model, tokenizer, k: int) -> str:
    if not isinstance(article_text, str) or not article_text: return ""
    cleaned_article = clean_article_text(article_text)
    try:
        sentences = nltk.sent_tokenize(cleaned_article)
    except LookupError:
        # --- PERBAIKAN 1 ---
        nltk.download('punkt_tab') 
        sentences = nltk.sent_tokenize(cleaned_article)
    if not sentences: return ""
    cls_id, sep_id, max_len = tokenizer.cls_token_id, tokenizer.sep_token_id, 512
    input_ids, token_type_ids, sent_pos_ids = [], [], []
    segment_id = 0
    for i, sent in enumerate(sentences):
        tokenized_sent = tokenizer.encode(sent, add_special_tokens=False)
        tokens_to_add = [cls_id] + tokenized_sent + [sep_id]
        types_to_add = [segment_id] * len(tokens_to_add)
        pos_to_add = [i] * len(tokens_to_add)
        if (len(input_ids) + len(tokens_to_add)) > max_len: break
        input_ids.extend(tokens_to_add); token_type_ids.extend(types_to_add); sent_pos_ids.extend(pos_to_add)
        segment_id = 1 - segment_id
    if not input_ids: return ""
    input_ids = torch.tensor([input_ids]).to(DEVICE)
    token_type_ids = torch.tensor([token_type_ids]).to(DEVICE)
    sent_pos_ids = torch.tensor([sent_pos_ids]).to(DEVICE)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, sentence_pos_ids=sent_pos_ids)
        logits = outputs.logits
    cls_indices = (input_ids[0] == cls_id).nonzero(as_tuple=True)[0]
    probs = torch.softmax(logits[0, cls_indices], dim=1)[:, 1]
    top_k_indices = torch.topk(probs, k=min(k, len(probs))).indices
    sorted_indices = sorted(top_k_indices.tolist())
    return " ".join([sentences[i] for i in sorted_indices])

def predict_abstractive_summary(article_text: str, model, tokenizer) -> str:
    if not isinstance(article_text, str) or not article_text:
        return ""
    
    # --- PERBAIKAN 2 ---
    # Model T5 dilatih dengan prefix. 
    prefix = "ringkas: "
    inputs_text = prefix + article_text
    
    # Tokenisasi artikel
    inputs = tokenizer(
        inputs_text,  # Gunakan inputs_text yang sudah diberi prefix
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    inputs = inputs.to(DEVICE)
    
    # Generate ringkasan
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=4,
            max_length=64,
            early_stopping=True
        )
    
    # Decode teks yang di-generate
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main(args):
    print(f"Menggunakan device: {DEVICE}")

    # --- 1. Muat Model-Model ---
    print("Memuat model Extractive (IndoBERT)...")
    model_extractive = BertSumClassifier.from_pretrained(args.extractive_path).to(DEVICE)
    tokenizer_extractive = BertTokenizer.from_pretrained(EXTRACTIVE_TOKENIZER_NAME)
    model_extractive.eval()
    
    print("Memuat model Abstractive (T5)...") 
    model_abstractive = AutoModelForSeq2SeqLM.from_pretrained(args.abstractive_path).to(DEVICE)
    tokenizer_abstractive = T5TokenizerFast.from_pretrained(ABSTRACTIVE_TOKENIZER_NAME)
    model_abstractive.eval()

    print(f"Memuat test set dari {args.test_file}...")
    # --- PERBAIKAN 4 (Logika Sampling) ---
    full_test_dataset = load_dataset('csv', data_files={'test': args.test_file})['test']
    
    # --- LOGIKA SAMPLING 10% BARU ---
    if args.use_sample:
        print(f"Menggunakan 10% sampel dari test set ({len(full_test_dataset)} entri)...")
        sample_split = full_test_dataset.train_test_split(train_size=0.1, seed=42, shuffle=True)
        test_dataset = sample_split['train'] # Ini adalah 10% sampel kita
        print(f"Jumlah data evaluasi: {len(test_dataset)}")
    else:
        test_dataset = full_test_dataset # Gunakan dataset penuh
        print(f"Jumlah data evaluasi: {len(test_dataset)}")
    
    print("Memuat metrik ROUGE...")
    rouge_metric = evaluate.load('rouge')

    preds_extractive, preds_abstractive, preds_baseline, references = [], [], [], []

    print(f"Menjalankan prediksi komparatif pada {len(test_dataset)} artikel...")
    for example in tqdm(test_dataset):
        article = example['clean_article']
        reference = example['clean_summary']
        if not isinstance(article, str) or not isinstance(reference, str): continue
            
        preds_extractive.append(predict_extractive_summary(article, model_extractive, tokenizer_extractive, k=EXTRACTIVE_K))
        preds_abstractive.append(predict_abstractive_summary(article, model_abstractive, tokenizer_abstractive))
        preds_baseline.append(generate_baseline_summary(article, k=BASELINE_K))
        references.append(reference)

    print("Menghitung skor ROUGE akhir...")
    rouge_extractive = rouge_metric.compute(predictions=preds_extractive, references=references)
    rouge_abstractive = rouge_metric.compute(predictions=preds_abstractive, references=references)
    rouge_baseline = rouge_metric.compute(predictions=preds_baseline, references=references)

    print("\n" + "="*50)
    print("--- SKOR AKHIR KOMPARATIF PADA TEST SET ---")
    if args.use_sample: 
        print("--- (DIEVALUASI PADA 10% SAMPEL TEST SET) ---")
    print("="*50)
    
    print(f"\n--- Baseline (Lead-{BASELINE_K}) ---")
    print(f"ROUGE-1: {rouge_baseline['rouge1'] * 100:.2f} | ROUGE-2: {rouge_baseline['rouge2'] * 100:.2f} | ROUGE-L: {rouge_baseline['rougeL'] * 100:.2f}")

    print(f"\n--- Model Extractive (k={EXTRACTIVE_K}) (BERT Kustom) ---")
    print(f"ROUGE-1: {rouge_extractive['rouge1'] * 100:.2f} | ROUGE-2: {rouge_extractive['rouge2'] * 100:.2f} | ROUGE-L: {rouge_extractive['rougeL'] * 100:.2f}")

    print(f"\n--- Model Abstractive (T5) ---") # --- PERBAIKAN 5 (Kosmetik) ---
    print(f"ROUGE-1: {rouge_abstractive['rouge1'] * 100:.2f} | ROUGE-2: {rouge_abstractive['rouge2'] * 100:.2f} | ROUGE-L: {rouge_abstractive['rougeL'] * 100:.2f}")
    
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi komparatif model summarization.")
    parser.add_argument("--extractive_path", type=str, default=EXTRACTIVE_MODEL_PATH)
    parser.add_argument("--abstractive_path", type=str, default=ABSTRACTIVE_MODEL_PATH)
    parser.add_argument("--test_file", type=str, default="liputan6_dataset_test.csv")
    
    # --- PERBAIKAN 6: Tambahkan argumen yang hilang ---
    parser.add_argument(
        "--use_sample",
        action="store_true",
        help="Gunakan 10% sampel dari test set untuk evaluasi cepat."
    )
    
    args = parser.parse_args()
    main(args)