import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Dict, List, Tuple

choices = ["A", "B", "C", "D"]

def format_subject(subject: str) -> str:
    """将下划线分隔的主题转换为自然语言格式"""
    return " ".join(subject.split("_"))

def format_example(df: pd.DataFrame, idx: int, include_answer: bool = True) -> str:
    """格式化单个问题示例"""
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df: pd.DataFrame, subject: str, k: int = -1) -> str:
    """生成包含k个示例的提示模板"""
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.inference_mode()
def eval(args: argparse.Namespace, 
         subject: str, 
         model: torch.nn.Module, 
         tokenizer: AutoTokenizer,
         dev_df: pd.DataFrame, 
         test_df: pd.DataFrame) -> Tuple[np.ndarray, float, np.ndarray]:
    """评估模型在指定学科上的表现"""
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    
    # 预编码选项token
    option_ids = [tokenizer(choice).input_ids[-1] for choice in choices]  # 取最后一个token避免前缀问题
    
    for i in range(test_df.shape[0]):
        # 动态调整prompt长度
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        
        # 智能截断处理
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_position_embeddings - 50  # 保留缓冲
        )
        input_ids = inputs.input_ids.to(model.device)
        
        # 模型推理
        outputs = model(input_ids=input_ids)
        last_token_logits = outputs.logits[:, -1, :]  # 取最后一个token的logits
        
        # 计算选项概率
        probs = torch.nn.functional.softmax(
            last_token_logits[:, option_ids], 
            dim=-1
        ).cpu().numpy().flatten()
        
        # 预测结果
        pred = choices[np.argmax(probs)]
        label = test_df.iloc[i, test_df.shape[1] - 1]
        cors.append(pred == label)
        all_probs.append(probs)

    acc = np.mean(cors)
    print(f"Average accuracy {acc:.3f} - {subject}")
    return np.array(cors), acc, np.array(all_probs)

def load_model_and_tokenizer(model_name: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """加载模型和tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"  # 因果LM需要左填充
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def main(args: argparse.Namespace):
    """主执行流程"""
    # 初始化模型
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # 准备目录结构
    os.makedirs(args.save_dir, exist_ok=True)
    results_dir = os.path.join(args.save_dir, f"results_{args.model}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取测试学科
    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(args.data_dir, "test"))
        if "_test.csv" in f
    ])
    
    # 初始化结果容器
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() 
        for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    
    # 分学科评估
    for subject in subjects:
        try:
            dev_df = pd.read_csv(
                os.path.join(args.data_dir, "dev", f"{subject}_dev.csv"), 
                header=None
            )[:args.ntrain]
            test_df = pd.read_csv(
                os.path.join(args.data_dir, "test", f"{subject}_test.csv"),
                header=None
            )
            
            cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
            
            # 记录分类结果
            for subcat in subcategories.get(subject, []):
                subcat_cors[subcat].append(cors)
                for cat, subcat_list in categories.items():
                    if subcat in subcat_list:
                        cat_cors[cat].append(cors)
            all_cors.append(cors)
            
            # 保存结果
            test_df[f"{args.model}_correct"] = cors
            for j, choice in enumerate(choices):
                test_df[f"{args.model}_choice{choice}_probs"] = probs[:, j]
            test_df.to_csv(
                os.path.join(results_dir, f"{subject}.csv"),
                index=None
            )
            
        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")
            continue
    
    # 打印汇总结果
    for subcat in subcat_cors:
        if subcat_cors[subcat]:  # 跳过空列表
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            print(f"Average accuracy {subcat_acc:.3f} - {subcat}")
    
    for cat in cat_cors:
        if cat_cors[cat]:
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            print(f"Average accuracy {cat_acc:.3f} - {cat}")
    
    if all_cors:
        weighted_acc = np.mean(np.concatenate(all_cors))
        print(f"Overall average accuracy: {weighted_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5,
                      help="Number of few-shot training examples")
    parser.add_argument("--data_dir", "-d", type=str, default="data",
                      help="Directory containing test/dev data")
    parser.add_argument("--save_dir", "-s", type=str, default="results",
                      help="Directory to save results")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-8B",
                      help="HuggingFace model identifier")
    args = parser.parse_args()
    
    # 启动主流程
    start_time = time.time()
    main(args)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
