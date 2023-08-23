import argparse
from instruct_mt.template import PromptTemplate
from instruct_mt.utils.langdict import LANGDICT
import pdb
import langcodes

def main(args):

    templater = PromptTemplate(args.template_str,)
    with open(args.infile+'.'+args.srclang) as fsrc, open(args.infile+'.'+args.tgtlang) as ftgt, open(args.outfile,'w') as fout:
        for srcline, tgtline in zip(fsrc,ftgt):
            src,tgt = srcline.strip(),tgtline.strip()
            fout.write(templater.construct_full(src,tgt,langcodes.Language.get(args.srclang).display_name(),
                            langcodes.Language.get(args.tgtlang).display_name()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile')
    parser.add_argument('--outfile')
    parser.add_argument('--template-str',default="[<srclang>]: <input> [<tgtlang>]:")
    parser.add_argument('--srclang')
    parser.add_argument('--tgtlang')

    args = parser.parse_args()
    main(args)
