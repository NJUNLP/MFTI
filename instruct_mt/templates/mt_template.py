import re

class PromptTemplate:
    def __init__(self,template_str):
        self.template_str = template_str

    def _construct_prefix(self,x,srclang,tgtlang,xs=None,ys=None):
        return self.template_str.replace('<srclang>',srclang).replace('<tgtlang>',tgtlang).replace('<input>',x)
    
    def construct_full(self,x,y,srclang,tgtlang):
        return self._construct_prefix(x,srclang,tgtlang) + ' ' + y

    def extract_translation(self,lm_prefix,lm_generated):
        return lm_generated.replace(lm_prefix,'')
    
    def construct_prefix(self,x,srclang,tgtlang,xs=None,ys=None):
        prefix = self._construct_prefix(x,srclang,tgtlang)
        few_shot_examples = []
        if xs is not None and ys is not None:
            for _x,_y in zip(xs,ys):
                few_shot_examples.append(self.construct_full(_x,_y,srclang,tgtlang))
        if len(few_shot_examples) == 0:
            return prefix
        else:
            return ' '.join(few_shot_examples) + " " + prefix
    

        
    



