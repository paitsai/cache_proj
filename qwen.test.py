from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers import GenerationConfig

from cache.myatt import MyCache, calculate_cache_size
import time, torch

def txt_to_string_advanced(file_path: str, 
                          encoding: str = 'utf-8',
                          max_size: int = None) -> str:
    """
    参数:
        file_path: 文件路径
        encoding: 指定编码（默认utf-8）
        max_size: 最大读取字节数（None表示无限制）
    """
    with open(file_path, 'r', encoding=encoding) as file:
        if max_size:
            return file.read(max_size)
        print("txt读取成功")
        return file.read()[:130000]

    


def kv_cache_monitor(past_key_values, **kwargs):
    current_step = kwargs.get("current_step", 0)
    if current_step % 16 == 0:  # Adjust the frequency as needed
        kv_cache_size = calculate_cache_size(past_key_values)
        print(f"Step {current_step}: KV Cache = {kv_cache_size:.2f} MB")


def print_memory_usage(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{prefix} Memory: Alloc={allocated:.2f}MB | Reserved={reserved:.2f}MB | Peak={max_allocated:.2f}MB")


def strategy_test(type: str,model,model_inputs):
    if type=="MyCache":
        past_cache_matrix=MyCache()
    else:
        past_cache_matrix=DynamicCache()
    start = time.time()
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024*1,
        past_key_values=past_cache_matrix,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    total_token_num=len(output_ids)
    end=time.time()
    # parse thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    print("thinking content:", thinking_content)
    print("content:", content)
    print(f"KV Cache使用：{calculate_cache_size(past_cache_matrix)}MB")
    print(f"{type} cache策略耗时: {end - start:.4f}秒，共生成{total_token_num}个tokens，推理速度：{total_token_num/(end-start):.4f}token per second")  # 输出示例: 耗时: 0.0342秒
    
    print_memory_usage("after generation:")


print_memory_usage("Before loading model")

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prepare the model input
prompt = txt_to_string_advanced("text.txt") + "**Please give me a summary of this textbook!**"
messages = [
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
strategy_test("MyCache",model,model_inputs)
strategy_test("default",model,model_inputs)