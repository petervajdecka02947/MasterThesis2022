from io import StringIO
from html.parser import HTMLParser
import re

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = True
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def preprocess_explanation(text):
    #result_type = ["true","mostly-true","half-true","barely-true","false","pants-fire",
    #               "mostly true", "mostly false", "half true","half false","barely true", "barely false",
    #               "pants on fire", "pants fire","misleading","truth","evidence","wrong","incorrect","correct","fake","absurd",
    #"ridiculous","ridiculously","mistake","inaccurate","exaggerates","claim"]
    with open("forbidden_words.txt") as f:
            result_type = [line.strip() for line in f]
     
    text = strip_tags(text)
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", " ", text)
    text = re.sub(r"\n|\t", " ", text)
    for pattern in result_type:
        reg = "[^.?!]*(?<=[.?\s!]){}(?=[\s.?!])[^.?!]*[.?!]".format(pattern)
        text = re.sub(reg, "", text, flags = re.I)
    text = re.sub(r" +", " ", text)
    return text.strip()
    
    

def preprocess(text):
    
    text = strip_tags(text)
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", " ", text)
    text = re.sub(r"\n|\t", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def Get_tokens_length(txt):
    return len(re.findall(r'\w+', txt))

def RemoveLastSentence(txt, tokenizer):
    text_no_last_sentence = tokenizer.tokenize(txt)[:-1]
    return ' '.join(text_no_last_sentence)    

def Truncate_to_max_length(text, max_tokens, tokenizer):
    length = Get_tokens_length(text)
    while length > max_tokens:
        text = RemoveLastSentence(text,tokenizer)
        length = Get_tokens_length(text)
    return text



