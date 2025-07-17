import json
import sys
from transformers import AutoTokenizer
from datasets import Dataset


def is_prime(n):
    """判断一个数是否是质数"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_magic_prime(data_size, ctx_len):
    """找到小于 data_size / ctx_len - 1 的最大形如 3n+2 的质数"""
    max_candidate = int(data_size // ctx_len) - 1
    for i in range(max_candidate, 0, -1):
        if i % 3 == 2 and is_prime(i):
            return i
    return None


# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/69043a6d-b152-4bd1-be10-e1130af6487f/miniRWKV/MiniMind2")


def count_tokens_in_jsonl(file_path, batch_size=1000):
    # 读取所有数据到内存并构建 Dataset
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 提取 content 并拼接成文本
    texts = []
    for entry in data:
        conversations = entry.get("conversations", [])
        text = "\n".join([msg.get("content", "") for msg in conversations])
        texts.append(text)

    # 构建 Dataset
    dataset = Dataset.from_dict({"text": texts})

    # 使用 tokenizer 批量 tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=batch_size)
    
    # 统计总 token 数（通过 attention_mask 长度）
    total_tokens = 0
    for idx in range(len(tokenized_datasets)):
        input_ids = tokenized_datasets[idx]["input_ids"]
        total_tokens += len(input_ids)

    return total_tokens


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python token_count.py <input_file.jsonl>")
        sys.exit(1)

    input_file = sys.argv[1]
    token_count = count_tokens_in_jsonl(input_file)
    print(f"Total tokens in {input_file}: {token_count}")

    ctx_len = 512
    magic_prime = find_magic_prime(token_count, ctx_len)
    if magic_prime:
        print(f"\nmagic_prime = {magic_prime} (for ctxlen {ctx_len})")
        print(f'--my_exit_tokens {token_count} --magic_prime {magic_prime} --ctx_len {ctx_len}')
    else:
        print("No suitable magic_prime found.")