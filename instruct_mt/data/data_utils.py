LANGUAGE_VOCAB = {
    'English': 0,
    'Russian': 1,
    'Chinese': 2,
    'German': 3,
    'Spanish': 4,
    'French': 5,
    'Japanese': 6,
    'Italian': 7,
    'Portuguese': 8,
    'Greek': 9,
    'Korean': 10,
    'Finnish': 11,
    'Indonesian': 12,
    'Turkish': 13,
    'Arabic': 14,
    'Vietnamese': 15,
    'Thai': 16,
    'Bulgarian': 17,
    'Catalan': 18,
    'Hindi': 19,
    'Estonian': 20,
    'Bengali': 21,
    'Tamil': 22,
    'Urdu': 23,
    'Swahili': 24,
    'Telugu': 25,
    'Basque': 26,
    'Burmese': 27,
    'Haitian': 28,
    'Quechua': 29
}
def make_language_pair_vocab():
    pair2index = {}
    for lang_i in LANGUAGE_VOCAB:
        for lang_j in LANGUAGE_VOCAB:
            if lang_i != lang_j:
                pair2index[(lang_i,lang_j)] = len(pair2index)
    return pair2index

pair2index = make_language_pair_vocab()

def batch_decode(
        token_ids,
        tokenizer
):
        outputs = []
        for _token_ids in token_ids:
            begin = False
            output = []
            for t in _token_ids:
                if  t == tokenizer.eos_token_id:
                    begin = True
                    continue
                elif not begin:
                    continue
                elif t == tokenizer.pad_token_id:
                    outputs.append(tokenizer.convert_tokens_to_string(output))
                    output = []
                    break
                elif t == tokenizer.unk_token_id:
                    continue
                else:
                    output.append(tokenizer._convert_id_to_token(t))
            if len(output) > 0:
                outputs.append(tokenizer.convert_tokens_to_string(output))
        if len(outputs) == 0:
            import pdb; pdb.set_trace()
        return outputs
        



def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

def group_to_batches(examples,sizes,max_tokens_per_batch):
    sorted_examples = [e for _,e in sorted(zip(sizes,examples),key=lambda pair: pair[0])]
    sorted_sizes = list(sorted(sizes))
    batches = []
    cur_batch_size = sorted_sizes[0]
    batch = [sorted_examples[0]]
    for idx in range(1,len(examples)):
        if cur_batch_size + sorted_sizes[idx] < max_tokens_per_batch:
            batch.append(sorted_examples[idx])
            cur_batch_size += sorted_sizes[idx]
        else:
            batches.append(batch)
            batch = [sorted_examples[idx]]
            cur_batch_size = sorted_sizes[idx]

    if len(batch) > 0:
        batches.append(batch)
    return batches

class LMPrefixDataLoader:
    def __init__(self,lm_prefixes,tokenizer,max_tokens):
        datas = []
        sizes = []
        for i,lm_prefix in enumerate(lm_prefixes):
            input_ids = tokenizer(lm_prefix,return_tensors='pt')['input_ids'].squeeze(0)
            datas.append((input_ids,i))        
            sizes.append(input_ids.size(0))
        self.batches = group_to_batches(datas,sizes,max_tokens_per_batch=max_tokens)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg