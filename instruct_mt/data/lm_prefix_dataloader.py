def group_to_batches(examples,sizes,max_tokens_per_batch):
    sorted_examples = [e for _,e in sorted(zip(sizes,examples),key=lambda pair: pair[0])]
    sorted_sizes = list(sorted(sizes))
    batches = []
    cur_batch_size = 0
    batch = []
    for idx in range(len(examples)):
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


class LMPrefixDataLoader:
    def __init__(self,lm_prefixes,tokenizer,max_tokens):
        datas = []
        sizes = []
        for i,lm_prefix in enumerate(lm_prefixes):
            input_ids = tokenizer(lm_prefix,return_tensors='pt')['input_ids'].squeeze(0)
            datas.append((input_ids,i))        
            sizes.append(input_ids.size(0))
        self.batches = group_to_batches(datas,sizes,max_tokens_per_batch=max_tokens)
        self.encodings, self.ids = [], []
        for batch in self.batches:
            input = collate_tokens([b[0] for b in batch],pad_idx=tokenizer.pad_token_id,left_pad=True).cuda()
            attention_mask = input.ne(tokenizer.pad_token_id)
            self.ids.extend([b[1] for b in batch])
            self.encodings.append({'input_ids':input, 'attention_mask': attention_mask})

    def __iter__(self):
        return iter(self.encodings)

    def __len__(self):
        return len(self.encodings)