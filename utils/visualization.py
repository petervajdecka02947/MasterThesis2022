#install important packages
#!pip install pickle-mixin
#!pip install corpy
#!pip install nltk 
#!pip install wordcloud 

#import of packages intended for work with dataset
import os
import pickle
import nltk                                                   
from nltk.corpus import stopwords                             #Stopwords corpus
from sklearn.feature_extraction.text import TfidfVectorizer   #For TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import itertools
from corpy.morphodita import Tagger
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import joblib
from tqdm.auto import tqdm
tqdm.pandas()

class Model:

    def __init__(self,
                 model_dir,
                 tfidf_dir,
                 tagger_dir,
                 stopwords_path):

                         self.model = joblib.load(model_dir)
                         self.tfidf = joblib.load(tfidf_dir)
                         self.tagger = Tagger(tagger_dir)
                         self.words = list(pd.read_csv(stopwords_path,
                                                       header = None,
                                                       index_col = None)[0])
                         self.stop_words = set([re.sub(r"\'", r'', word).lower() for word in self.words])

    def get_prediction(self, text: str):
                    sample = preprocess([text],self.stop_words,self.tagger)
                    sample_tranfered = self.tfidf.transform(sample)
                    sample_tranfered = pd.DataFrame(sample_tranfered.toarray()).astype(float)
                    sample_tranfered.columns = self.tfidf.get_feature_names()
                    probs = self.model.predict_proba(sample_tranfered)
                    self.preds = probs[:,1]
                    self.labels = self.model.predict(sample_tranfered)

                    return str({'text': text,
                            'probability':self.preds[0],
                            'positive-1/negative-0':self.labels[0]
                            })

def preprocess(text,stopwords,tagger_ins): # Peter Vajdecka 5.11.2020
    """
     Function preprocess text such as substract unwanted symbols, tokenize, lemmatize, 
    """
    temp =[]
    sent = []
    
    for sentence in tqdm(text):
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence)           #Removing HTML tags
        sentence = re.sub(r'\'|"',r' ',sentence)
        sentence = re.sub(r'[?|!|"|#]',r'',sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)  #Removing Punctuations
        sentence = re.sub(r'\d+', '', sentence)            #Remove numbers
        sentence = re.sub(r' +', ' ', sentence) 

   # Lemmatization
        tokens = list(tagger_ins.tag(sentence,sents=False,guesser=True,convert="pdt_to_conll2009"))
        words=[el[1] for el in tokens if el[1].lower() not in stopwords]
        #words = [snow.stem(word) for word in sentence.split() if word not in stop]   # Stemming and removing stopwords
        temp.append(words)

    for row in temp:
        sequ = ''
        for word in row:
            sequ = sequ + ' ' + word
        sent.append(sequ.strip())
    return sent  

def transporttoTfIdf(train_df,tf_idf_ins, opt):
    """
       Function transform text matrix to tf-idf matrix
    """
    # 1.Independent variables
    tf_id = tf_idf_ins
    train_df = tf_idf_ins.fit_transform(train_df)
    if opt == True:
        train_df = pd.DataFrame(train_df.toarray())
        train_df.columns = tf_id.get_feature_names()
        return train_df
    else:
        return train_df.toarray()

def wordCloud(input_df,stop_words,tagger,tf_idf,name_str,IsLemma):
    """
       Function to nicely visulize most frequent words
    """
    if IsLemma==True:
        df_trans=preprocess(input_df,stop_words,tagger)
    else:
        df_trans = input_df  
    df_trans= transporttoTfIdf(df_trans, tf_idf, True)
    print("Number of unique words in report:{}".format(len(df_trans.columns)))
    
    cl= WordCloud(width=1600, height=800 , background_color='white', stopwords=stop_words).generate_from_frequencies(df_trans.T.sum(axis=1))
    plt.figure(figsize=(20,10))
    plt.imshow(cl)
    #plt.imshow(cl, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(name_str)
    return plt.show()

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.RdPu):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmapS)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f' # if normalize else 'd'
    fns=11
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],fmt),
                 horizontalalignment="center",
                 fontsize=fns,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   # plt.tight_layout()
    plt.ylim(1.5, -0.5)  # top to bottom solution
    
    return 

def get_lof_score(dat):
    ad = []
    try:
        lof = LocalOutlierFactor(n_neighbors = neighbours,metric='cosine')
        embeds = dat.to_list()
        lof.fit_predict(embeds).tolist()
        return 1 - (1/(1-(lof.negative_outlier_factor_)))
    except:
        return [0.51]*len(dat)

def embeddings_sentence_bert(text, IsBase, Bert_name):
    
        start = time.time()
        if IsBase==True:            
            model = SentenceTransformer(Bert_name, device = 'cpu')  # model  bert-base-uncased           
        else:     
                     
            word_embedding_model = models.Transformer(Bert_name)
            
            # Apply mean pooling to get one fixed sized sentence vector
            
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
            
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device = 'cuda:0')

        
        #Sentences are encoded by calling model.encode()
        sentence_vectors = model.encode(text, show_progress_bar=True, batch_size = 10)            

        end = time.time()

        print("Time for creating "+ str(len(sentence_vectors))+" embedding vectors " + str((end - start)/60))
        print('Model used :'+ Bert_name )

        return sentence_vectors
