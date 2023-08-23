from tqdm import tqdm
import torch
import sacrebleu
import argparse
from transformers import AutoModelForCausalLM,AutoTokenizer
from instruct_mt.templates.mt_template import PromptTemplate
import langcodes
import os
import random
import torch.multiprocessing
import time

from instruct_mt.peft import (
    PeftConfig,
    PeftModel
)

from instruct_mt.models.modeling_xglm import XGLMForCausalLM
from instruct_mt.data.data_utils import LMPrefixDataLoader, collate_tokens, batch_decode, LANGUAGE_VOCAB



def load_model_and_tokenizer(model_name_or_path):
    if args.peft_type is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.padding_side = 'left'
        # tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch.float16)
    else:
        adapter_config = PeftConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_config.base_model_name_or_path)
        tokenizer.padding_side = 'left'
        if args.peft_type in ['lorax','prefix_x']:
            model = XGLMForCausalLM.from_pretrained(adapter_config.base_model_name_or_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(adapter_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model,args.model_name_or_path)
        model = model.half()
    model = model.eval()
    return model, tokenizer

def load_test_set(filename,templator,srclang,tgtlang,subset_size=None,few_shot_pool=None,few_shot_k=None):

    lm_prefixes = []
    references = []
    srclang_name, tgtlang_name = langcodes.Language.get(srclang).display_name(), langcodes.Language.get(tgtlang).display_name()
    with open(filename+'.'+srclang) as fsrc, open(filename+'.'+tgtlang) as ftgt:
        for i,(srcline, tgtline) in enumerate(zip(fsrc,ftgt)):
            if few_shot_pool is not None and few_shot_k is not None and few_shot_k > 0:
                with open(few_shot_pool+"."+srclang) as fsrc, open(few_shot_pool+"."+tgtlang) as ftgt:
                    srclines = [line.strip() for line in fsrc.readlines()]
                    tgtlines = [line.strip() for line in ftgt.readlines()]
                    random_idxs = [random.randint(0,len(srclines)-1) for _ in range(few_shot_k)] 
                    xs = [srclines[i] for i in random_idxs]
                    ys = [tgtlines[i] for i in random_idxs]
            else:
                xs = ys = None
            lm_prefixes.append(
                templator.construct_prefix(srcline.strip(),
                                           srclang_name,
                                           tgtlang_name,
                                           xs=xs,
                                           ys=ys
                                           )
            )
            references.append(tgtline.strip())
            if subset_size is not None and i > subset_size:
                break
    language_ids = [LANGUAGE_VOCAB[srclang_name],LANGUAGE_VOCAB[tgtlang_name]]
    return {
        'lm_prefixes': lm_prefixes,
        "language_ids": language_ids,
        'references':references,
        "tgtlang_name": tgtlang_name,
        "srclang_name": srclang_name
    }


def evaluate(model,tokenizer,templator,test_dataset,max_tokens=2048,savefile=None,tokenize='flores101',device=None):
    lm_prefixes, references = test_dataset['lm_prefixes'], test_dataset['references']
    dataloader = LMPrefixDataLoader(lm_prefixes,tokenizer,max_tokens)
    translations = []
    ids = []
    for batch in tqdm(dataloader):
        input = collate_tokens([b[0] for b in batch],pad_idx=tokenizer.pad_token_id,left_pad=True).to(device)
        attention_mask = input.ne(tokenizer.pad_token_id)
        b = input.size(0)
        encoding = {'input_ids':input, 'attention_mask': attention_mask}
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                max_new_tokens=150,
                num_beams=4,
                early_stopping=True,
                )
            lm_generated = batch_decode(generated_ids,tokenizer)
            _lm_prefixes = batch_decode(input,tokenizer)
            translations.extend(
                [templator.extract_translation(lp,lg) for lp,lg in zip(
                    _lm_prefixes,
                    lm_generated,
                )]
            )
        ids.extend([b[1] for b in batch])
    translations = [e for _,e in sorted(zip(ids,translations),key=lambda pair: pair[0])]
    
    if savefile is not None:
        bleu = sacrebleu.corpus_bleu(
            translations,[references], tokenize=tokenize
        )
        os.makedirs(os.path.dirname(savefile),exist_ok=True)
        with open(savefile,'w') as f:
            for lp, r, t in zip(lm_prefixes,references,translations):
                f.write('\n'.join([lp,r,t]) + '\n\n')
            f.write(str(bleu) + '\n')
    return translations

def load_validation_dataset(filename,templator,subset_size):
    lm_prefixes = []
    references = []
    langs = ["en","fr","de","ru","ko","bg","fi","zh","ar","ko","hi","ta","ca"]
    for srclang in langs:
        for tgtlang in langs:
            srclang_name, tgtlang_name = langcodes.Language.get(srclang).display_name(), langcodes.Language.get(tgtlang).display_name()
            with open(filename+'.'+srclang) as fsrc, open(filename+'.'+tgtlang) as ftgt:
                for i,(srcline, tgtline) in enumerate(zip(fsrc,ftgt)):
                    xs = ys = None
                    lm_prefixes.append(
                        templator.construct_prefix(srcline.strip(),
                                                srclang_name,
                                                tgtlang_name,
                                                xs=xs,
                                                ys=ys
                                                )
                    )
                    references.append(tgtline.strip())
                    if subset_size is not None and i > subset_size:
                        break
    return {
        'lm_prefixes': lm_prefixes,
        'references':references,
    }

def validate_checkpoint(args,step,device,score_dict):
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path + "/checkpoint-{}".format(step))
    model = model.to(device)

    validation_dataset = load_validation_dataset(args.validation_file,templator,20)
    generated = evaluate(
                model,
                tokenizer,
                templator,
                validation_dataset,
                args.max_tokens,
                savefile=None,
                tokenize=args.tokenize,
                device=device
                )

    references = validation_dataset['references']
    score = sacrebleu.corpus_bleu(
            generated,[references], tokenize=args.tokenize
    ).score

    score_dict[step] = round(score,4)


def evaluate_srclangs(args,step,srclangs,tgtlangs,device,score_dict):
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path + "/checkpoint-{}".format(step))
    model = model.to(device)
    for srclang in srclangs:
        srclang_scores = []
        for tgtlang in tgtlangs:
            if srclang == tgtlang:
                continue
            test_dataset = load_test_set(args.test_file,templator,srclang,tgtlang,args.subset_size,args.few_shot_pool,args.few_shot_k)
            savefile = args.savedir  + "/{}-{}.txt".format(srclang,tgtlang)
            generated = evaluate(
                model,
                tokenizer,
                templator,
                test_dataset,
                args.max_tokens,
                savefile,
                tokenize=args.tokenize,
                device=device
                )
            
            references = test_dataset['references']
            score = sacrebleu.corpus_bleu(
                generated,[references], tokenize=args.tokenize
            ).score
            print("{}-{}: {}".format(srclang,tgtlang,score))
            srclang_scores.append(round(score,2))
        score_dict[srclang] = srclang_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path')
    parser.add_argument('--template-str',default="Translation: [<srclang>]: <input> [<tgtlang>]:")
    parser.add_argument('--test-file')
    parser.add_argument("--validation-file")
    parser.add_argument('--srclang',nargs="+",default=["en","de","fr","ca","fi","ru","bg","zh","ko","ar","sw","hi","ta"])
    parser.add_argument('--tgtlang',nargs="+",default=["en","de","fr","ca","fi","ru","bg","zh","ko","ar","sw","hi","ta"])
    parser.add_argument('--max-tokens',type=int,default=256)
    parser.add_argument('--savedir')
    parser.add_argument("--few-shot-pool")
    parser.add_argument("--few-shot-k",type=int)
    parser.add_argument("--peft-type",default=None)
    parser.add_argument("--subset-size",type=int)
    parser.add_argument("--tokenize")
    parser.add_argument("--save-steps",type=int)

    args = parser.parse_args()
    if args.template_str == "1":
        templator = PromptTemplate("Translation: [<srclang>]: <input> [<tgtlang>]:")
    elif args.template_str == "2":
        templator = PromptTemplate("Translation: <input> [<tgtlang>]:")
    elif args.template_str == "3":
        templator = PromptTemplate("<input> Translate from [<srclang>] to [<tgtlang>]:")
    elif args.template_str == "4":
        templator = PromptTemplate("<input> Translate to [<tgtlang>]:")
    elif args.template_str == "5":
        templator = PromptTemplate("The <tgtlang> translation of <srclang> sentence '<input>' is:")
    elif args.template_str == "6":
        templator = PromptTemplate("The <tgtlang> translation of the sentence '<input>' is:")
    else:
        templator = PromptTemplate(args.template_str)

    start_time = time.time()
    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(1,9):
        device = "cuda:{}".format(i-1)
        p = torch.multiprocessing.Process(target=validate_checkpoint, args=(args, i*args.save_steps, device,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict)

    best_step = max(return_dict, key=return_dict.get)
    print("Best Step is {}".format(best_step))

    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    srclangs = [
        ["en","de"],
        ["fr","ca"],
        ["fi","ru"],
        ["bg","zh"],
        ["ko","ar"],
        ["sw"],
        ["hi"],
        ["ta"]
    ]
    #evaluate_srclangs(args,best_step,srclangs[0],args.tgtlang,{})
    for i, _srclangs in enumerate(srclangs):
        device = "cuda:{}".format(i)
        p = torch.multiprocessing.Process(target=evaluate_srclangs, args=(args, best_step, _srclangs,args.tgtlang,device,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    
    total_scores = 0
    total_langs = 0
    for srclang, srclang_scores in return_dict.items():
        print(srclang,srclang_scores)
        total_scores += sum(srclang_scores)
        total_langs += len(srclang_scores)
    print("Average {}".format(total_scores/total_langs))

    end_time = time.time()
    print("Total Time: {}".format(end_time - start_time))

