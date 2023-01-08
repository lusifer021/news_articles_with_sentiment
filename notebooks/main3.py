from newsapi import NewsApiClient
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
import spacy
import os
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time 
from nltk.tokenize import sent_tokenize

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from spacy import displacy
from collections import Counter

import warnings
warnings.filterwarnings('ignore')
# config Completer.use_jedi = False # if autocompletion doesnot work in kaggle notebook  | hit tab

# !pip install text_hammer
import text_hammer as th
from tqdm import *

def text_preprocessing(df,col_name):
  column = col_name
  df[column] = df[column].progress_apply(lambda x:str(x).lower())
  df[column] = df[column].progress_apply(lambda x:th.cont_exp(x))
  df[column] = df[column].progress_apply(lambda x:th.remove_emails(x))
  df[column] = df[column].progress_apply(lambda x:th.remove_html_tags(x))
  df[column] = df[column].progress_apply(lambda x:th.remove_special_chars(x))
  df[column] = df[column].progress_apply(lambda x:th.remove_accented_chars(x))

  return (df)

from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')
tokenizer.save_pretrained('bert-tokenizer')
bert.save_pretrained('bert-model')
# for saving model locally and we can load it later on

import shutil
shutil.make_archive('bert-tokenizer', 'zip', 'bert-tokenizer')
shutil.make_archive('bert-model','zip','bert-model')

### we can use distilbert its lighter cheaper and similar performance 

from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')

max_len = 60
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
# embeddings = dbert_model(input_ids,attention_mask = input_mask)[0]


embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 means pooler_output)
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
out = Dense(16,activation = 'relu')(out)
y = Dense(3,activation = 'softmax')(out)

    
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True

import tensorflow as tf
optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
### Changed Categorical to Binary
loss =tf.keras.losses.CategoricalCrossentropy(from_logits = True)
metric = tf.keras.metrics.CategoricalAccuracy('balanced_accuracy'),
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

model.load_weights("sentiment_weights.h5")

def pred(texts):
    x_val = tokenizer(text=texts,add_special_tokens=True,max_length=60,truncation=True,padding='max_length',return_tensors='tf',return_token_type_ids=True,return_attention_mask=True,verbose=True)
    validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    pred_val = np.argmax(validation,axis = 1)
    return pred_val



from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest





from flask import Flask, Response, request
app = Flask(__name__)
import pymongo
import json
from bson.objectid import ObjectId
try:
    mongo = pymongo.MongoClient(host = "mongodb://mongo:xx3vSKOQPPY1X72jvSi6@containers-us-west-100.railway.app:6105",
                 
                serverSelectionTimeoutMS = 2000)
    db = mongo.news
    mongo.server_info()
except:
    print("ERROR - Cannot connect to db")

newsapi = NewsApiClient(api_key='01b1cf475c13462b89997b60b3aa8ee7')

def get_data(keyword):
    news = []
    all_articles = newsapi.get_everything(
        q=keyword,
        language='en',   
    )
    articles = all_articles["articles"]
    if all_articles["totalResults"] ==0:
        return 0
    if len(articles) > 5:
        articles = articles[:5]
    for i in articles:

        article = {'title':i["title"],'link':i['url'],'published':i['publishedAt']}
        news.append(article)
    return news

# Function to predict the sentiment of the title
def sentiment(text):
  blob = TextBlob(text)
  return blob.sentiment.polarity


CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

def func(ent,name):
    for x in ent.split(" "):
        if x in name.split(" "):
            return True
    
    return False

@app.route("/users", methods = ["POST"])
def create_user_article():

    try:
        name = request.form["name"]
        # print(name)

        if db.users.count_documents({"name": name},limit =1)>0:
            db.users.delete_one({"name" : name})
        
        data = get_data(name)
        if data == 0:
            return Response(
                response = json.dumps({"message":"no articles found with this name"}),
                status = 200,
                mimetype = "application/json"
            )
        df = pd.DataFrame(data)

        a = []
        # news = ""
        # news_list_ner = ""
        news_all = ""
        relevant_passages = []
        for i in range(df.shape[0]):
                
            news = ""
            news_list_ner = ""
            url = df["link"][i]
            r1 = requests.get(url)
            

            # We'll save in coverpage the cover page content
            coverpage = r1.content

            soup1 = BeautifulSoup(coverpage, 'html5lib')
            # print(soup1)
            # News identification using ner
            coverpage_news = soup1.find_all('p')
            passage = ""
            for c in coverpage_news:
                passage = passage + " " + c.text
            lst = sent_tokenize(passage)
            # for c in coverpage_news:
            #     article = nlp (c.text)
            #     for ent in article.ents:
            #         if (ent.label_=="PERSON") & ((ent.text == name) | (ent.text in name.split(' '))):
            #             news_list_ner = news_list_ner + " " + c.text
            #     # print(c.text)
            #     news = news + " " + c.text

            min = []
            flag=0
            for i in lst:
                article = nlp(i)
                if(flag==1):
                    for word in article:
                     if (word.pos_ == "PRON"):
                        min.append(i)
                        flag=0
                for ent in article.ents:
                    if (ent.label_ == "PERSON" and (ent.text == name or func(name,ent.text))):  #and ent.text == "Anthony Albanese"
                        min.append(i)
                        flag=1
                    else:
                        flag=0

            min = set(min)
            min = list(min)
            relevant_passages.append(min)

            final_list=[]
            for i in min:
                for j in i:
                    final_list.append(j)
            
            final_str = ''
            for i in final_list:
                final_str+=i

            # news_all = news_all + " " + news_list_ner
            # print(df['published'])
            if(len(final_str)> 1):
                result = pred(final_str)
                print(result)
                a.append({"test" : news, "date" : df["published"][i], "sentiment" : result[0]})

            
            
        # main_result = sentiment(news_all)


        new = { "name" : name,  "articles": a}
        print(name)
        dbResponse = db.users.insert_one(new)

        print(dbResponse.inserted_id)
        # print(new)
        return Response(
                response = json.dumps(db.users.find_one({"name": name},{"_id" : False})),
                status = 200,
                mimetype = "application/json"
            )
    except Exception as ex :
        print(ex)
        return Response(
            response = json.dumps({"message":"cannot read users"}),
            status = 500,
            mimetype = "application/json")

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))

