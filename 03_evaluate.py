# 03_evaluate.py
import torch
import nltk
import evaluate
import argparse
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm.auto import tqdm

# Impor model dan utilitas
from model_architecture import BertSumClassifier
from utils import clean_article_text, generate_baseline_summary

# --- Konfigurasi ---
TOKENIZER_NAME = 'indobenchmark/indobert-base-p1'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# (EDA 5) Kita putuskan k=2 adalah yang terbaik
BERT_K_SENTENCES = 2 
BEST_MODEL_DIR = './bert-summarizer-best-model'

def predict_bert_summary(article_text: str, model, tokenizer, k: int) -> str:
    """
    Menghasilkan ringkasan menggunakan model BERT yang sudah dilatih.
    """
    if not isinstance(article_text, str) or not article_text:
        return ""

    cleaned_article = clean_article_text(article_text)
    try:
        sentences = nltk.sent_tokenize(cleaned_article)
    except LookupError:
        nltk.download('punkt_tab')
        sentences = nltk.sent_tokenize(cleaned_article)
        
    if not sentences:
        return ""

    # Tokenisasi format BertSum
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    max_len = 512

    input_ids = []
    token_type_ids = []
    sent_pos_ids = []

    segment_id = 0
    for i, sent in enumerate(sentences):
        tokenized_sent = tokenizer.encode(sent, add_special_tokens=False)
        tokens_to_add = [cls_id] + tokenized_sent + [sep_id]
        types_to_add = [segment_id] * len(tokens_to_add)
        pos_to_add = [i] * len(tokens_to_add)

        if (len(input_ids) + len(tokens_to_add)) > max_len:
            break
        
        input_ids.extend(tokens_to_add)
        token_type_ids.extend(types_to_add)
        sent_pos_ids.extend(pos_to_add)
        segment_id = 1 - segment_id
    
    # Konversi ke Tensor
    input_ids = torch.tensor([input_ids]).to(DEVICE)
    token_type_ids = torch.tensor([token_type_ids]).to(DEVICE)
    sent_pos_ids = torch.tensor([sent_pos_ids]).to(DEVICE)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Prediksi
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_pos_ids=sent_pos_ids
        )
        logits = outputs.logits
    
    cls_indices = (input_ids[0] == cls_id).nonzero(as_tuple=True)[0]
    cls_logits = logits[0, cls_indices]
    
    # Ambil probabilitas untuk kelas '1'
    probs = torch.softmax(cls_logits, dim=1)[:, 1]
    
    # (EDA 5) Ambil 'k' kalimat
    top_k_indices = torch.topk(probs, k=min(k, len(probs))).indices
    sorted_indices = sorted(top_k_indices.tolist())
    
    predicted_summary = " ".join([sentences[i] for i in sorted_indices])
    return predicted_summary

def main(args):
    print(f"Menggunakan device: {DEVICE}")

    print("Memuat model terbaik dan tokenizer...")
    try:
        model = BertSumClassifier.from_pretrained(args.model_path).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error memuat model dari {args.model_path}: {e}")
        print("Pastikan Anda sudah menjalankan 02_train.py")
        return

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    
    print(f"Memuat test set dari {args.test_file}...")
    try:
        test_dataset = load_dataset('csv', data_files={'test': args.test_file})['test']
    except FileNotFoundError:
        print(f"ERROR: File {args.test_file} tidak ditemukan.")
        return

    print("Memuat metrik ROUGE...")
    rouge_metric = evaluate.load('rouge')

    test_predictions_bert = []
    test_predictions_baseline = []
    test_references = []

    print(f"Menjalankan prediksi pada {len(test_dataset)} artikel test set...")
    for example in tqdm(test_dataset):
        article = example['clean_article']
        reference = example['clean_summary']
        
        if not isinstance(article, str) or not isinstance(reference, str):
            continue
            
        # 1. Prediksi Model BERT (k=2)
        pred_bert = predict_bert_summary(article, model, tokenizer, k=BERT_K_SENTENCES)
        
        # 2. Prediksi Baseline (Lead-2)
        pred_baseline = generate_baseline_summary(article, k=2)
        
        test_predictions_bert.append(pred_bert)
        test_predictions_baseline.append(pred_baseline)
        test_references.append(reference)

    print("Menghitung skor ROUGE akhir...")
    final_rouge_bert = rouge_metric.compute(predictions=test_predictions_bert, references=test_references)
    final_rouge_baseline = rouge_metric.compute(predictions=test_predictions_baseline, references=test_references)

    print("\n" + "="*50)
    print("--- SKOR AKHIR PADA TEST SET ---")
    print("="*50)
    
    print(f"\n--- Model BERT (k={BERT_K_SENTENCES}) ---")
    print(f"ROUGE-1: {final_rouge_bert['rouge1'] * 100:.2f}")
    print(f"ROUGE-2: {final_rouge_bert['rouge2'] * 100:.2f}")
    print(f"ROUGE-L: {final_rouge_bert['rougeL'] * 100:.2f}")

    print("\n--- Baseline (Lead-2) ---")
    print(f"ROUGE-1: {final_rouge_baseline['rouge1'] * 100:.2f}")
    print(f"ROUGE-2: {final_rouge_baseline['rouge2'] * 100:.2f}")
    print(f"ROUGE-L: {final_rouge_baseline['rougeL'] * 100:.2f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi model BertSum pada test set.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=BEST_MODEL_DIR,
        help="Path ke folder model terbaik yang sudah dilatih."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="liputan6_dataset_test.csv",
        help="Path ke file .csv test set mentah."
    )
    args = parser.parse_args()
    main(args)