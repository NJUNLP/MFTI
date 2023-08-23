import logging
from typing import  Dict
from datasets import load_dataset


import transformers


from data.data_collator_for_partial_language_modeling import DataCollatorForPartialLanguageModeling


def make_lm_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, use_gist=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    def tokenize_function(examples):
        output = tokenizer(
            examples[text_column_name],
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False)
        if output['input_ids'][0][-1] != tokenizer.eos_token_id:
            for i in output['input_ids']:
                i.append(tokenizer.eos_token_id)
            for i in output['attention_mask']:
                i.append(1)
        return output
    dataset_args = {}
    data_files = {'train':data_args.train_file} 
    extension = "text"
    dataset_args["keep_linebreaks"] = False
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        **dataset_args,
    )
    logging.warning("Tokenizing inputs... This may take some time...")
    text_column_name = list(raw_datasets["train"].features)[0]
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=["text"],
        desc="Running tokenizer on dataset",
    )
    # Data collator
    data_collator = DataCollatorForPartialLanguageModeling(
        tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,
    )
    if data_args.blocking_ngram is not None:
        data_collator.blocking_ngram = " ".join(data_args.blocking_ngram.split("_"))
    return dict(train_dataset=tokenized_datasets['train'], data_collator=data_collator)
    
