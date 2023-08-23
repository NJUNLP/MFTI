from copy import deepcopy
import argparse
languages = ["en","de","fr","ca","fi","ru","bg","zh","ko","ar","sw","hi","ta"]
language_splits = {
    "mix": ["en","ko","sw"],
    "src": ["zh","fi"],
    "tgt": ["de","hi"],
    "unseen": ["ru","fr","bg","ar","ca","ta"]
}

def load_few_shot_scores(filename):
    scores_dict = {}
    with open(filename) as f:
        for line in f:
            srclang = line[:2]
            scores = line[3:].split("\t")
            for tgtlang, s in zip(languages,scores):
                if srclang == tgtlang:
                    continue
                else:
                    scores_dict[(srclang,tgtlang)] = float(s)
    return scores_dict


def load_scores(filename):
    scores_dict = {}
    with open(filename) as f:
        for line in f:
            srclang = line[:2]
            scores = line[4:-2].split(",")
            copyed_languages = deepcopy(languages)
            copyed_languages.remove(srclang)
            for tgtlang, s in zip(copyed_languages,scores):
                scores_dict[(srclang,tgtlang)] = float(s)
    return scores_dict


def main(args):
    if args.few_shot:
        scores_dict = load_few_shot_scores(args.filename)
    else:
        scores_dict = load_scores(args.filename)
    direction_to_partition = {
        ("src","src"): "one_side_reverse",
        ("src","tgt"): "seen_direction",
        ("src","mix"): "seen_direction",
        ("src","unseen"): "tgt_unseen",
        ("tgt","src"): "reverse_direction",
        ("tgt","tgt"): "one_side_reverse",
        ("tgt","mix"): "reverse_direction",
        ("tgt","unseen"): "tgt_unseen",
        ("mix","src"): "reverse_direction",
        ("mix","tgt"): "seen_direction",
        ("mix","mix"): "seen_direction",
        ("mix","unseen"): "tgt_unseen",
        ("unseen","src"): "src_unseen",
        ("unseen","tgt"): "src_unseen",
        ("unseen","mix"): "src_unseen",
        ("unseen","unseen"): "both_unseen"

    }
    partition_dict = {
        "seen_direction": [],
        "reversed_direction": [],
        "one_side_reverse": [],
        "src_unseen": [],
        "tgt_unseen": [],
        "both_unseen": []
    }

    for source_split in language_splits:
        for target_split in language_splits:
            partition_scores = []
            for srclang in language_splits[source_split]:
                for tgtlang in language_splits[target_split]:
                    if srclang == tgtlang:
                        continue
                    else:
                        partition_scores.append(
                            scores_dict[(srclang,tgtlang)]
                        )
            average_score = sum(partition_scores) / len(partition_scores)
            if direction_to_partition[(source_split,target_split)] == "seen_direction":
                partition_dict["seen_direction"].extend(partition_scores)
            elif direction_to_partition[(source_split,target_split)] == "reverse_direction":
                partition_dict["reversed_direction"].extend(partition_scores)
            elif direction_to_partition[(source_split,target_split)] == "one_side_reverse":
                partition_dict["one_side_reverse"].extend(partition_scores)
            elif direction_to_partition[(source_split,target_split)] == "src_unseen":
                partition_dict["src_unseen"].extend(partition_scores)
            elif direction_to_partition[(source_split,target_split)] == "tgt_unseen":
                partition_dict["tgt_unseen"].extend(partition_scores)
            elif direction_to_partition[(source_split,target_split)] == "both_unseen":
                partition_dict["both_unseen"].extend(partition_scores)
            else:
                raise ValueError("unknown {}".format((source_split,target_split)))

            print("{}-{}: {}".format(source_split,target_split,round(average_score,1)))

    for k,v in partition_dict.items():
        average = sum(v) / len(v)
        print("{}: {}".format(k,round(average,1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--few-shot",action="store_true",default=False)
    args = parser.parse_args()

    main(args)







