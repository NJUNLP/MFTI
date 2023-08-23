from torch.utils.data import Dataset
import copy
import logging
from typing import  Dict, Sequence
from dataclasses import dataclass, field


import torch
import transformers
from torch.utils.data import Dataset

from data.utils import jload

from data.data_utils import LANGUAGE_VOCAB, pair2index

def preprocess(
    inputs: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    inputs_tokenized = _tokenize_fn(inputs, tokenizer) 
    input_ids = inputs_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    return dict(input_ids=input_ids, labels=labels)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

class LMWithLanguageIDDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer, peft_per):
        super(LMWithLanguageIDDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        inputs = [
            example.get("input")
            for example in list_data_dict
        ]

        if peft_per == "language":
            language_ids = [torch.tensor([LANGUAGE_VOCAB[example.get("srclang")], LANGUAGE_VOCAB[example.get("tgtlang")]]) for example in list_data_dict]
        else:
            language_ids = [torch.tensor([pair2index[(example.get("srclang"),example.get("tgtlang"))]]) for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(inputs, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.language_ids = language_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ret = dict(input_ids=self.input_ids[i], labels=self.labels[i], language_ids=self.language_ids[i])
        return ret
    

@dataclass
class DataCollatorForLMWithLangIdDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    use_language_id: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, language_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "language_ids","labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        language_ids = torch.nn.utils.rnn.pad_sequence(
            language_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return dict(
            input_ids=input_ids,
            language_ids=language_ids if self.use_language_id else None,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def make_lm_with_langid_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, use_language_id=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LMWithLanguageIDDataset(tokenizer=tokenizer, data_path=data_args.data_path, peft_per=data_args.peft_per)
    data_collator = DataCollatorForLMWithLangIdDataset(tokenizer=tokenizer,use_language_id=use_language_id)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    
