from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from config import HF_TOKEN  # Load from .env

def load_sanskrit_model():
    # Using IndicTrans for Sanskrit (or another suitable model)
    model_name = "ai4bharat/indictrans2-en-indic-ssft"  # Check for Sanskrit support
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, oken=HF_TOKEN )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, oken=HF_TOKEN )
    
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )