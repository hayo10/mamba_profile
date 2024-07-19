import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from collections import OrderedDict
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, MambaConfig
from transformers import AutoTokenizer, MambaForCausalLM
from datasets import load_dataset

from packaging import version
from typing import Any, ContextManager, Iterable, List, Tuple
from functools import partial

#mamba import
# from copied_selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


##data load
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
dataset = load_dataset("theprint/alpaca_cthulhu_small")
sm = dataset['train'].select(range(10))


torch.cuda.nvtx.range_push("model load")   
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").to('cuda')
torch.cuda.nvtx.range_pop()

##preprocess
max_length = 50280
prefix = "summarize this: "
suffix = "Here's the summary: "

def preprocess_function(examples):
    inputs = []

    for i in range(len(examples['input'])):
        if examples['input'][i] != '':
            inputs.append(prefix + examples['instruction'][i] + suffix + examples['input'][i])
        else:
            inputs.append(prefix + examples['instruction'][i])

    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    targets = tokenizer(examples['output'], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    model_inputs["labels"] = targets["input_ids"]
    
    return model_inputs


print(sm)
column_names = ['output', 'input', 'instruction']
train_dataset = sm.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names
        )

device = 'cuda'
from torch.utils.data import DataLoader
import torch.optim as optim

train_dataset.set_format(type="torch")
dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)

# total_loss = 0.0
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        torch.cuda.nvtx.range_push("input loading")
        input_data = batch['input_ids'].clone().detach().to(device)
        torch.cuda.nvtx.range_pop()
#        target = batch['labels'].clone().detach().to(device)

        output = model(input_data)
        # loss = criterion(output.logits, target)
        
        # total_loss += loss.item()
        # torch.cuda.nvtx.range_pop()
            
    # return total_loss / len(dataloader)
print('finished')


