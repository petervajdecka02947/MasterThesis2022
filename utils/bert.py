import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from tqdm.auto import tqdm
import pickle
import itertools
from sklearn import metrics
import operator
import warnings
from sentence_transformers import models, losses
import time
from datetime import datetime
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from sentence_splitter import SentenceSplitter
import time
import torch, gc
import math
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
tqdm.pandas()

def add_multi_index(lst):
    return list(range(lst))

def get_triple_from_df(df, idx_col ,
                       label_col, text_col,
                       claim_col):
    """
    Function to to create triples from main text column by target label
    Method:
            triplet = [anchor, positive, negative] = [
                                                      claim, 
                                                      sentence_related_to_anchor_claim, 
                                                      sentence_related_to_not_anchor_claim
                                                      ]
    """

   
    triplets = []
    negatives_lst = []
    labels = df[label_col].unique()
    
    
    #idxs = df[idx_col].to_list()
    #sentences = df[text_col].to_list()
    #claims = df[claim_col].to_list()
    
    for label in tqdm(labels):
        data_in = df[(df[label_col] == label)][[text_col,claim_col]]
        data_out = df[(df[label_col] != label)]
        data_len = len(data_in)
        data_out_indexes = data_out.index
        numbers = np.random.choice(data_out_indexes, data_len)
        negatives = data_out.loc[numbers,text_col].to_list()
        data_in["negatives"] = negatives
        triplets.append(data_in)
    res = pd.concat(triplets)
    res = [InputExample(texts = [str(rows[claim_col]), str(rows[text_col]), str(rows["negatives"])]) for index, rows in tqdm(res.iterrows())]
    return res

def get_triple_from_df_A(df, idx_col ,
                       label_col, text_col):
    """
    Function to to create triples from main text column by target label
    
    Method:
            triplet = [anchor, positive, negative] = [
                                                      sentence, 
                                                      another_sentence_with_label, 
                                                      another_sentence_with__different_label
                                                      ]
    
    
    """
    triplets = []
    negatives_lst = []
    labels = df[label_col].unique()
    df = df[[idx_col ,label_col, text_col]]
     
    for label in tqdm(labels):
        data_in = df[(df[label_col] == label)]#[[text_col]]
        data_out = df[(df[label_col] != label)]
        
        data_len = len(data_in) # get number of records
        
        data_out_indexes = list(data_out.index)
        data_in_indexes = list(data_in.index)
        
        #print(data_len)
        #print(data_in_indexes)
        try:
            if data_len > 1:

                numbers_out = np.random.choice(data_out_indexes, data_len, replace=False)
                numbers_in = [np.random.choice(data_in_indexes[0:i] + data_in_indexes[(i+1):data_len], 
                                               replace=False) for i,v in enumerate(data_in_indexes)]
            
                negatives = data_out.loc[numbers_out,text_col].to_list()
                positives = data_in.loc[numbers_in,text_col].to_list()

                data_in["positives"] = positives
                data_in["negatives"] = negatives

                triplets.append(data_in)
           
            else:
                 print("Too few sentences !")
                
        
        
        except:
            print(data_len)
            print(data_in_indexes)
            print(numbers_in)
            print(data_in.index)
        
        
    
    res = pd.concat(triplets)
    res = res[res[text_col] != res.positives]
    res = res[res.source_text_sentences != res.negatives]
    df = res
    res = [InputExample(texts = [str(rows[text_col]), str(rows["positives"]), str(rows["negatives"])]) for index, rows in tqdm(res.iterrows())]
    return res, df

def get_triple_from_df_B(df,df2,
                       label_col, text_col):
    """
    Function to to create triples from main text column by target label
    
    Method:
            triplet = [anchor, positive, negative] = [
                                                      sentence, 
                                                      random_sentence_from_oracles, 
                                                      random_sentence_from_opposite_oracles
                                                      ]
    
    """
    triplets = []
    negatives_lst = []
    labels = df[label_col].unique()
    df = df[[label_col, text_col]]
     
    for label in tqdm(labels):
        data_in = df[(df[label_col] == label)]#[[text_col]]
        data_out = df2[(df2[label_col] == label)]
        
        data_len = len(data_in) # get number of records
        
        data_out_indexes = list(data_out.index)
        data_in_indexes = list(data_in.index)
        
        #print(data_len)
        #print(data_in_indexes)
        try:
            if data_len > 1:

                numbers_out = np.random.choice(data_out_indexes, data_len, replace=False)
                numbers_in = [np.random.choice(data_in_indexes[0:i] + data_in_indexes[(i+1):data_len], 
                                               replace=False) for i,v in enumerate(data_in_indexes)]
            
                negatives = data_out.loc[numbers_out,text_col].to_list()
                positives = data_in.loc[numbers_in,text_col].to_list()

                data_in["positives"] = positives
                data_in["negatives"] = negatives

                triplets.append(data_in)
           
            else:
                 print("Too few sentences !")
                
        
        
        except:
            print(data_len)
            print(data_in_indexes)
            print(numbers_in)
            print(data_in.index)
        
        
    
    res = pd.concat(triplets)
    res = res[res[text_col] != res.positives]
    res = res[res.source_text_sentences != res.negatives]
    df = res
    res = [InputExample(texts = [str(rows[text_col]), str(rows["positives"]), str(rows["negatives"])]) for index, rows in tqdm(res.iterrows())]
    return res, df


def embeddings_sentence_bert(text, IsBase, Bert_name):
    
        start = time.time()
        if IsBase==True:            
            model = SentenceTransformer(Bert_name, device = 'cuda:0')  # model  bert-base-uncased           
        else:     
                     
            word_embedding_model = models.Transformer(Bert_name)
            
            # Apply mean pooling to get one fixed sized sentence vector
            
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
            
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device = 'cuda:0')

        
        #Sentences are encoded by calling model.encode()
        sentence_vectors = model.encode(text, show_progress_bar=True, batch_size = 1000)            

        end = time.time()

        print("Time for creating "+ str(len(sentence_vectors))+" embedding vectors " + str((end - start)/60))
        print('Model used :'+ Bert_name )

        return sentence_vectors

def split_data(dat, sour_col):  
    
    dat["label"] = list(range(len(dat)))
    dat['source_text'] = dat[sour_col]

    splitter = SentenceSplitter(language='en')
    dat['source_text_sentences'] = dat['source_text'].progress_apply(lambda x : splitter.split(text = x))
    dat['source_text_sentences_len'] = dat['source_text_sentences'].str.len() 
    dat['source_text_sentences_index'] = dat['source_text_sentences_len'].progress_apply(lambda x : add_multi_index(x))
    dat = dat.explode(['source_text_sentences',"source_text_sentences_index"]).reset_index()
    dat['sentence_len'] = dat['source_text_sentences'].str.split().str.len()
    
    dat = dat[dat.sentence_len > 5]

    return dat

def split_data_A(dat, sour_col):  
    
    dat["label"] = list(range(len(dat)))
    dat['source_text'] = dat[sour_col]

    splitter = SentenceSplitter(language='en')
    dat['source_text_sentences'] = dat['source_text'].progress_apply(lambda x : splitter.split(text = x))
    print("Data splitted into list")
    dat['source_text_sentences_count'] = dat['source_text_sentences'].str.len() 
    dat['source_text_sentences_index'] = dat['source_text_sentences_count'].progress_apply(lambda x : add_multi_index(x))
    dat = dat.explode(['source_text_sentences',"source_text_sentences_index"]).reset_index()
    print("Data splitted into rows")
    sentences_lst = dat["source_text_sentences"].tolist()
    embeddings_1 = embeddings_sentence_bert(sentences_lst, True, base_model_name)
    
    print("Embeddings done")
    
    sentences_lst = dat["shortExplanation_prep"].tolist()
    embeddings_2 = embeddings_sentence_bert(sentences_lst, True, base_model_name)
    
    print("Embeddings done")
    
    dat['source_text_sentences_embed'] = list(embeddings_1)
    dat['shortExplanation_prep_embed'] = list(embeddings_2)
    
    print("Columns added")
    
    dat['sentence_token_len'] = dat['source_text_sentences'].str.split().str.len()
    
    dat = dat[dat.sentence_token_len > min_sentence_tokens]

    return dat

def split_data_B(dat, sour_col):  
    
    dat["label"] = list(range(len(dat)))
    dat['source_text'] = dat[sour_col]

    splitter = SentenceSplitter(language='en')
    dat['source_text_sentences'] = dat['source_text'].progress_apply(lambda x : splitter.split(text = x))
    print("Data splitted into list")
    dat['source_text_sentences_count'] = dat['source_text_sentences'].str.len() 
    dat['source_text_sentences_index'] = dat['source_text_sentences_count'].progress_apply(lambda x : add_multi_index(x))
    dat = dat.explode(['source_text_sentences',"source_text_sentences_index"]).reset_index()
    print("Data splitted into rows")
    sentences_lst = dat["source_text_sentences"].tolist()
    embeddings_1 = embeddings_sentence_bert(sentences_lst, True, base_model_name)
    
    print("Embeddings done for columns source_text_sentences")
    
    sentences_lst = dat["shortExplanation_prep"].tolist()
    embeddings_2 = embeddings_sentence_bert(sentences_lst, True, base_model_name)
    
    print("Embeddings done")
    
    dat['source_text_sentences_embed'] = list(embeddings_1)
    dat['shortExplanation_prep_embed'] = list(embeddings_2)
    
    print("Columns added")
    
    dat['sentence_token_len'] = dat['source_text_sentences'].str.split().str.len()
    
    dat = dat[dat.sentence_token_len > min_sentence_tokens]

    return dat




#################################################################################################################################################### A

def cos_sim(x):
   # A_sparse = sparse.csr_matrix(np.array(x,x))
    similarities = spatial.distance.cosine(x[0], x[1])
    return (1/(1+similarities)) 

def plot_column_distribution(dat, col_name):
    """
    plot distribution of given columns 
    """
    return sns.distplot(dat[col_name], hist=True, kde=False, 
             bins=int(len(dat)/10), color = 'lightblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 5}) 

def get_oracles_A(data, isHead, constant):
    """
    
    Get top n similar sentences to fro rullink comments to short summaries
    
    constant: ilustrates top n selected sentences for summary
   
   """
    if isHead == True:
        output = data.sort_values(['id','cos_similarity'],ascending = False).groupby('id').head(constant)
    else:
        output = data[data.cos_similarity > constant]
    print("Selected {} % of all sentences".format((len(output)/len(data))*100))
    return output 

def get_oracles_B(data, isHead, constant):
    """
    
    Get top n similar sentences to ruling comments to short summaries
    
    constant: ilustrates top n selected sentences for summary from head and tail
   
   """
    output = data.sort_values(['id','cos_similarity'],ascending = False)
    top = output.groupby('id').head(constant)[["id" , "source_text_sentences"]]
    down = output.groupby('id').tail(constant)[["id" , "source_text_sentences"]]
    return top, down 