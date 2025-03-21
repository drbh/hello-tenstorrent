import pytest
from PIL import Image
import ast
import torch
import ttnn

from llama_models.llama3.api.tokenizer import Tokenizer
from tt_transformers.tt.common import (
    copy_host_to_device,
    get_padded_prefill_len,
    num_blocks_in_seq,
    get_block_size,
    get_max_prefill_chunk_size,
)


print(Tokenizer)
print(get_max_prefill_chunk_size)

out = get_max_prefill_chunk_size(
    max_prefill_seq_len=2048 * 2,
    seq_len=2048 * 1,
)

print(out)