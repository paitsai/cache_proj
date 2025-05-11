import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# 加载模型和tokenizer
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()

# 加载数据集
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
texts = [text for text in dataset["text"] if text.strip()]  # 过滤空文本

# 计算PPL的函数
def calculate_ppl(texts, stride=512, max_length=2048):
    total_logprob = 0
    total_tokens = 0
    
    for text in tqdm(texts[:100]):  # 测试前100条加快速度
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        total_logprob += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)
    
    ppl = np.exp(total_logprob / total_tokens)
    return ppl

# 执行计算
ppl = calculate_ppl(texts)
print(f"Perplexity: {ppl:.2f}")
