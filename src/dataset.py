import math
import json
import torch
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset

from .binidx import MMapIndexedDataset


def is_prime(n):
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


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(
            self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## train stage {args.train_stage} ##########")
        dataset_slot = self.data_size // args.ctx_len

        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1


    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len


        dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y
    
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    # def _generate_loss_mask(self, input_ids):
    #     loss_mask = [0] * len(input_ids)
    #     i = 0
    #     while i < len(input_ids):
    #         if input_ids[i:i + len(self.bos_id)] == self.bos_id:
    #             start = i + len(self.bos_id)
    #             end = start
    #             while end < len(input_ids):
    #                 if input_ids[end:end + len(self.eos_id)] == self.eos_id:
    #                     break
    #                 end += 1
    #             for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
    #                 loss_mask[j] = 1
    #             i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
    #         else:
    #             i += 1
    #     return loss_mask

    # def __getitem__(self, index):
    #     sample = self.samples[index]
    #     # 构建对话提示
    #     prompt = self._create_chat_prompt(sample['conversations'])
    #     input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
    #     input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

    #     # 生成动态损失掩码
    #     loss_mask = self._generate_loss_mask(input_ids)

    #     # 构建训练数据
    #     X = torch.tensor(input_ids[:-1], dtype=torch.long)
    #     Y = torch.tensor(input_ids[1:], dtype=torch.long)
    #     loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

    #     return X, Y, loss_mask
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)

        return X, Y