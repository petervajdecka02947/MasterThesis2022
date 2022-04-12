#import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sentence_splitter import SentenceSplitter
from sklearn.neighbors import LocalOutlierFactor
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
tqdm.pandas()

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils.visualization import *



def add_multi_index(lst):
    return list(range(lst))

def split_data(dat, sour_col, targ_col):  
    
    dat["label"] = list(range(len(dat)))
    dat['source_text'] = dat[sour_col]
    dat['target_text'] = dat[targ_col]

    splitter = SentenceSplitter(language='en')
    dat['source_text_sentences'] = dat['source_text'].progress_apply(lambda x : splitter.split(text = x))
    print("Data splitted into list")
    dat['source_text_sentences_len'] = dat['source_text_sentences'].str.len() 
    dat['source_text_sentences_index'] = dat['source_text_sentences_len'].progress_apply(lambda x : add_multi_index(x))
    dat = dat.explode(['source_text_sentences',"source_text_sentences_index"]).reset_index()
    print("Data splitted into rows")
    
    dat['sentence_token_len'] = dat['source_text_sentences'].str.split().str.len()
    
    dat = dat[dat.sentence_token_len > min_sentence_tokens]

    return dat

def plot_column_distribution(dat, col_name):
    """
    plot distribution of given columns 
    """
    return sns.distplot(dat[col_name], hist=True, kde=False, 
             bins=int(len(dat)/10), color = 'lightblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 5}) 

def get_tf_idf_lof_score(dat):   
    try:
        lof = ""
        tf_idf = TfidfVectorizer(min_df = 1, ngram_range=(1, 1), stop_words = stop_words)
        train_df = tf_idf.fit_transform(dat).toarray()
        lof = LocalOutlierFactor(n_neighbors = neighbours,metric='cosine')

        embeds = list(train_df.astype(float)) 
        lof.fit_predict(embeds).tolist()

        return 1 - (1/(1-(lof.negative_outlier_factor_)))
    
    except:
        
        return [0.50]*len(dat)
    
def get_doc2vec_lof_score(dat):  
    try:
        card2vec = []
        mod = Doc2Vec(vector_size=64, window = 5 ,min_count=1, epochs = 5)
        lof = LocalOutlierFactor(n_neighbors = neighbours, metric='cosine')
        dat_tokened = [doc.split(" ") for doc in dat]
        card_docs = [TaggedDocument(doc, [i]) 
                     for i, doc in enumerate(dat_tokened)]
        #print(card_docs)
        mod.build_vocab(card_docs)
        #print(model)
        mod.train(card_docs, total_examples=mod.corpus_count, epochs=mod.epochs)

        card2vec = [mod.infer_vector(dat_tokened[i]) for i in range(0,len(dat))]

        embeds = list(np.array(card2vec))
        lof.fit_predict(embeds).tolist()
        return 1 - (1/(1-(lof.negative_outlier_factor_)))
        
    except:
        return [0.50]*len(dat)    