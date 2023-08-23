#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM
)
from peft import LoraConfig, LoraXConfig, get_peft_model, TaskType

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers import Trainer, AutoModelForCausalLM

from data.data_utils import LANGUAGE_VOCAB, smart_tokenizer_and_embedding_resize
from trainer.peft_trainer import PeftTrainer, SavePeftModelCallback




IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
GIST_TOKEN = "<GIST>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

    model_type: Optional[str] = field(
        default="vanilla"
    )
    align_alpha: Optional[float] = field(
        default=0.
    )

    # Peft Args
    peft_type: Optional[str] = field(
        default="lora"
    )
    use_peft: bool = field(
        default=False
    )

    # LoRA Args
    lora_target_type: Optional[str] = field(
        default="q_proj;v_proj"
    )
    lora_r: Optional[int] = field(
        default=4
    )
    lora_alpha: Optional[int] = field(
        default=32
    )
    lora_dropout: Optional[float] = field(
        default=0.1
    )
    use_shared_lora: Optional[bool] = field(
        default=False
    )
    lora_composition_type: Optional[str] = field(
        default="multiply"
    )

@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    valid_file: str = field(default=None, metadata={"help": "Path to the training data."})

    # Peft Args
    peft_per: Optional[str] = field(
        default="language"
    )
    use_language_id: Optional[bool] = field(
        default=False
    )
    blocking_ngram : Optional[str]= field(
        default=None
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def make_peft_config(model_args,data_args):
    if model_args.peft_type == "lora":
        return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, 
        r = model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout
        )
    elif model_args.peft_type == "lorax":
        return LoraXConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, 
        r = model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        x = len(LANGUAGE_VOCAB),
        shared_lora=model_args.use_shared_lora,
        composition_type=model_args.lora_composition_type
        )
    else:
        raise ValueError("Unknown Peft Type")


def load_model(model_module,model_args,data_args,training_args):
    ## construct peft model
    model = model_module.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.float16,  
        )
    if model_args.model_type == "peft":
        peft_config = make_peft_config(model_args, data_args)
        model = get_peft_model(model,peft_config)
        model = model.half()
        model.print_trainable_parameters()
    return model

def load_model_no_ds(model_module,model_args,data_args,training_args):
    ## construct peft model
    model = model_module.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    if model_args.use_peft:
        peft_config = make_peft_config(model_args, data_args)
        model = get_peft_model(model,peft_config)
        model.print_trainable_parameters()
    return model

def load_model_and_data_module(tokenizer,data_args,model_args,training_args):
    model_module = AutoModelForCausalLM
    from data.lm_dataset import make_lm_data_module
    data_module = make_lm_data_module(tokenizer,data_args)
    if training_args.deepspeed:
        model = load_model(model_module,model_args,data_args,training_args)
    else:
        model = load_model_no_ds(model_module,model_args,data_args,training_args)
    return model, data_module


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",

    )

    model, data_module = load_model_and_data_module(tokenizer,data_args,model_args,training_args)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    if model_args.model_type == "peft":
        trainer = PeftTrainer(model=model, tokenizer=tokenizer, args=training_args,callbacks=[SavePeftModelCallback], **data_module)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()