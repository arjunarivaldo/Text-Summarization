# utils.py
import re
import nltk
from nltk.tokenize import sent_tokenize

def clean_article_text(text: str) -> str:
    """
    Membersihkan boilerplate 'Liputan6 . com...' dari awal artikel.
    (Berdasarkan EDA Langkah 4)
    """
    if not isinstance(text, str):
        return ""
        
    # Pola regex yang kita temukan di EDA
    pattern = r"^\s*Liputan6\s*\.\s*com\s*,\s*[^:]+:\s*"
    cleaned_text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
    return cleaned_text.strip()

def generate_baseline_summary(article_text: str, k: int = 3) -> str:
    """
    Menghasilkan ringkasan 'Lead-k' (hanya ambil k kalimat pertama).
    (Berdasarkan EDA Langkah 6)
    """
    if not isinstance(article_text, str) or not article_text:
        return ""
        
    # 1. Cleaning
    cleaned_article = clean_article_text(article_text)
    
    # 2. Segmentasi
    try:
        sentences = sent_tokenize(cleaned_article)
    except LookupError:
        nltk.download('punkt')
        sentences = sent_tokenize(cleaned_article)
        
    if not sentences:
        return ""
    
    # 3. Ambil 'k' kalimat pertama
    baseline_sentences = sentences[:k]
    
    return " ".join(baseline_sentences)