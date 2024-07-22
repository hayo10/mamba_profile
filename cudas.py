import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from collections import OrderedDict
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import copy
import time
from transformers import PreTrainedModel, MambaConfig
from transformers import AutoTokenizer
from datasets import load_dataset

from packaging import version
from typing import Any, ContextManager, Iterable, List, Tuple
from functools import partial

#mamba import
# from copied_selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

##import beam search generation

from hugMamba import MambaForCausalLM

##data load
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")


torch.cuda.nvtx.range_push("model load")   
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to('cuda')
torch.cuda.nvtx.range_pop()
model.eval()


##preprocess
max_length = 50
prompt = "My cat wrote all this CUDA code for a new language model and"
model_input = tokenizer(prompt, max_length=max_length, truncation=True, return_tensors="pt").to('cuda')

def beam_search(model, tokenizer, input_ids,beam_size=2, max_length=50):
    finished_beams = []
    running_beam = [(0, input_ids)]

    while len(finished_beams) < beam_size and running_beam:
        beam_score, new_input_ids = running_beam.pop(0)
        with torch.no_grad():
            outputs = model(new_input_ids)
            logits = outputs.logits[:, -1, :]
            top_k_values, top_k_indices = torch.topk(logits, beam_size, dim=-1)
            
 
        input_ids_per_beam = [new_input_ids] * beam_size
        for i in range(beam_size):
            score = top_k_values[:,i]
            token = top_k_indices[:,i]
            
            # Add the new token and update attention_mask
            new_input_per_beam = torch.cat((input_ids_per_beam[i], token.unsqueeze(1)), dim=1)

            if token == tokenizer.eos_token_id or new_input_ids.shape[1] == max_length+14:
                finished_beams.append((beam_score + score, new_input_per_beam))
            else:
                running_beam.append((beam_score + score, new_input_per_beam))

        # Sort the running beams by score
        running_beam.sort(key=lambda x: x[0], reverse=True)
    torch.cuda.nvtx.range_pop()
    # Return the highest scoring finished beam
    return max(finished_beams, key=lambda x: x[0])[1]

start = time.time()
pred = beam_search(model, tokenizer, model_input['input_ids'])
end = time.time()
ref = tokenizer.decode(pred[0], skip_special_tokens=True)
print(ref)
print(end-start)

