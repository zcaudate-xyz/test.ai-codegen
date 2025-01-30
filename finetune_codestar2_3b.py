from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "bigcode/starcoder2-3b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("models/llm")
tokenizer.save_pretrained("models/llm")