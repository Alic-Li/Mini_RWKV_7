# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.binidx import MMapIndexedDataset  # noqa: E402


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


# Modify these to your DATA_NAME and CTX_LEN, to compute the correct --my_exit_tokens and --magic_prime ####
# Use only the path with the DataName, without .bin or .idx extensions. ####
# Usage: python compute_magic_prime.py ####
DATA_NAME = '/home/rwkv/RWKV-LM-V7/data/demo'
CTX_LEN = 4096

print(f"### Loading {DATA_NAME}")
data = MMapIndexedDataset(DATA_NAME)
data_len = len(data)
data_size = len(data._bin_buffer) // data._index._dtype_size

print(f"\n### {DATA_NAME}.bin/idx has {data_size} tokens, {data_len} items. Dtype {data._index.dtype}")

n_chunk = int(data_size // CTX_LEN) - 1
for i in range(n_chunk, 0, -1):
    if i % 3 == 2:
        if is_prime(i):
            print(f"\n### magic_prime = {i} (for ctxlen {CTX_LEN})")
            print(
                f'\n--my_exit_tokens {data_size} --magic_prime {i} --ctx_len {CTX_LEN}\n')
            exit(0)
