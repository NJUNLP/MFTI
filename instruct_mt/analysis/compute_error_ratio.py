from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import argparse
import sacrebleu
import fasttext
import re
import os

def read_file(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()][:-2]

    references = lines[1::4]
    hypothesis = lines[2::4]
    return references, hypothesis

def is_too_short(r,h,tgtlang):
    if tgtlang == "zh":
        return len(h) < 0.5*len(r)
    else:
        return len(h.split()) < 0.5 * len(r.split())

def is_too_long(r,h,tgtlang):
    if tgtlang == "zh":
        return len(h) > 2*len(r)
    else:
        return len(h.split()) > 2 * len(r.split())

REPEATER = re.compile(r"(.+?)\1+$")

def principal_period(s):
    i = (s+s).find(s, 1, -1)
    return None if i == -1 else s[:i]

def contain_repeat_ngram(r,h,tgtlang):
    words = h.split() if tgtlang != 'zh' else h
    len_w = len(words)
    for i in range(len_w):
        for j in range(1,len_w):
            if words[i:i+j] == words[min(i+j,len_w):min(len_w,i+2*j)] and words[min(i+2*j,len_w):min(i+3*j,len_w)]:
               # print(words[i:i+j])
                return True
    return False

def repeat_template_pattern(r,h,tgtlang):
    return len(re.split(r'\[(.*?)\]',h)) > 2

def low_comet_score(comet_score,threshold):
    return comet_score < threshold

def is_copy_source(s,r,h):
    return sacrebleu.sentence_bleu(h, [s]).score > 50

def not_srclang_tgtlang(s,r,h,srclang,tgtlang,model):
    # if "டைனோசர்கள் பறவைகள் அல்ல, எனினும் இவற்றின் இறகுகள் பல்வேறு பறவை" in h:
    #     import pdb; pdb.set_trace()
    # if not halfjudge(h, srclang, model):
    #     if not halfjudge(h, tgtlang, model):
    #         return True
    # return False
    #return (not halfjudge(h, srclang, model)) and (not halfjudge(h, tgtlang, model))

    return (not halfjudge(h, srclang, model)) and (not halfjudge(h, tgtlang, model))

def compute_comet_scores(sources, references, hypothesis):
    from comet import download_model, load_from_checkpoint
    model_path = download_model("wmt21-comet-da")
    model = load_from_checkpoint(model_path)
    data = [{"src":src_sent,"mt":hyp_sent,"ref":ref_sent} for src_sent,ref_sent,hyp_sent in zip(sources,references,hypothesis)]
    model_output = model.predict(data, batch_size=80,gpus=1)
    mean_score,scores = model_output['system_score'],model_output['scores']
    return scores

def halfjudge(sent,label_lang,model):
    filt_sent = re.sub(r'[^\w\s]', '', sent)
    word_lst = filt_sent.split()
    first_sent = ' '.join(word_lst[:int(len(word_lst)/2)])
    last_sent = ' '.join(word_lst[int(len(word_lst)/2):])

    first_test_lang = model.predict(first_sent)[0][0][-2:] 
    last_test_lang =  model.predict(last_sent)[0][0][-2:] 

    if first_test_lang == label_lang and last_test_lang == label_lang :
        return True
    return False

def fullsentjudge(sent,label_lang,model):
    
    first_test_lang = model.predict(sent)[0][0][-2:] 
    if first_test_lang == label_lang :
        return True
    return False



def main(args):
    references, hypothesis = read_file(args.infile)
    srclang, tgtlang = tuple(args.langs.split("_"))
    print(srclang,tgtlang)
    srcfile = "../data/raw_data/flores101_dataset/test/test." + srclang
    with open(srcfile) as f:
        sources = [line.strip() for line in f.readlines()][:len(hypothesis)]

    lang_model = fasttext.load_model('../data/lid.176.bin')
    #comet_scores = compute_comet_scores(sources, references, hypothesis)

    #too_short, too_long_repeat_ngram, too_long_repeat_pattern, too_long_low_comet, copy_source, wrong_lang_id, ordinary_repeat_ngram, ordinary_repeat_pattern, ordinary_low_comet = [],[],[],[],[],[],[],[],[]
    #too_long_others, ordinary_others = [], []
    repeat_ngram, repeat_pattern, copy_source, wrong_language_id, too_short,too_long, low_comet, others = [],[],[],[],[],[],[],[]
    for s,r,h in zip(sources,references,hypothesis):
        if contain_repeat_ngram(r, h, tgtlang):
            repeat_ngram.append((s,r,h))
        elif repeat_template_pattern(r,h,tgtlang):
            repeat_pattern.append((s,r,h))
        elif is_copy_source(s,r,h):
            copy_source.append((s,r,h))
        elif not_srclang_tgtlang(s, r, h, srclang, tgtlang, lang_model):
            wrong_language_id.append((s,r,h))
        elif is_too_short(r, h, tgtlang):
            too_short.append((s,r,h))
        elif is_too_long(r,h,tgtlang):
            too_long.append((s,r,h))
        else:
            others.append((s,r,h))


        # if is_too_short(r, h, tgtlang):
        #     too_short.append((s,r,h))
        # elif is_too_long(r, h, tgtlang):
        #     if contain_repeat_ngram(r, h, tgtlang):
        #         too_long_repeat_ngram.append((s,r,h))
        #     elif repeat_template_pattern(r, h, tgtlang):
        #         too_long_repeat_pattern.append((s,r,h))
        #     elif low_comet_score(score, args.threshold):
        #         too_long_low_comet.append((s,r,h))
        #     else:
        #         too_long_others.append((s,r,h))
        # else:
        #     if contain_repeat_ngram(r, h, tgtlang):
        #         ordinary_repeat_ngram.append((s,r,h))
        #     elif repeat_template_pattern(r,h,tgtlang):
        #         ordinary_repeat_pattern.append((s,r,h))
        #     elif is_copy_source(s, r, h):
        #         copy_source.append((s,r,h))
        #     elif not_srclang_tgtlang(s, r, h, args.srclang, args.tgtlang, lang_model):
        #         wrong_lang_id.append((s,r,h))
        #     elif low_comet_score(score, args.threshold):
        #         ordinary_low_comet.append((s,r,h))
        #     else:
        #         ordinary_others.append((s,r,h))

    print("Repeat Ngram", len(repeat_ngram))
    print("Repeat pattern",len(repeat_pattern))
    print("Copy Source", len(copy_source))
    print("Wrong Language Id",len(wrong_language_id))
    print("Too short",len(too_short))
    print("Too long",len(too_long))
    #print("Low comet",len(low_comet))
    print("Others",len(others))
    if args.savedir is not None:
        os.makedirs(args.savedir,exist_ok=True)
        with open(args.savedir + "/repeat_ngram","w") as f:
            for s,r,h in repeat_ngram:
                f.write(s + "\n" + r + "\n" + h + "\n\n")

        with open(args.savedir + "/repeat_pattern","w") as f:
            for s,r,h in repeat_pattern:
                f.write(s + "\n" + r + "\n" + h + "\n\n")

        with open(args.savedir + "/copy_source","w") as f:
            for s,r,h in copy_source:
                f.write(s + "\n" + r + "\n" + h + "\n\n")

        with open(args.savedir + "/wrong_language_id","w") as f:
            for s,r,h in wrong_language_id:
                f.write(s + "\n" + r + "\n" + h + "\n\n")

        with open(args.savedir + "/too_short","w") as f:
            for s,r,h in too_short:
                f.write(s + "\n" + r + "\n" + h + "\n\n")

        with open(args.savedir + "/too_long","w") as f:
            for s,r,h in too_long:
                f.write(s + "\n" + r + "\n" + h + "\n\n")

        with open(args.savedir + "/low_comet","w") as f:
            for s,r,h in low_comet:
                f.write(s + "\n" + r + "\n" + h + "\n\n")

        with open(args.savedir + "/others","w") as f:
            for s,r,h in others:
                f.write(s + "\n" + r + "\n" + h + "\n\n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile")
    parser.add_argument("--langs")
    parser.add_argument("--srcfile")
    parser.add_argument("--savedir")
    parser.add_argument("--threshold",type=float)

    #print(contain_repeat_ngram("r", "the"," tgtlang"))
    #print(contain_repeat_ngram("", "ஆனால், பின்னர், விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்கிரவாண்டி, விக்க", ""))
    #print(sacrebleu.sentence_bleu("La desena tempesta amb nom de la temporada d'huracans a l'Atlàntic es diu Jerry, ja que es tracta d'una tempesta subtropical que s'ha format avui a l'Oceà Atlàntic.", ["La desena tempesta amb nom de la temporada d'huracans a l'Atlàntic es diu Jerry, i es tracta d'una tempesta subtropical que s'ha format avui a l'Oceà Atlàntic."]).score)
    args = parser.parse_args()
    main(args)

    
