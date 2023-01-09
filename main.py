from newsapi import NewsApiClient
import pandas as pd
from textblob import TextBlob
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
# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import spacy
# from spacy import displacy
# from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time 
# from nltk.tokenize import sent_tokenize

import spacy
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


newsapi = NewsApiClient(api_key='977653ce268a4d6987ff737a4d238684')

def get_data(keyword):
    news = []
    all_articles = newsapi.get_everything(
        q=keyword,
        language='en',   
    )
    articles = all_articles["articles"]
    if all_articles["totalResults"] ==0:
        return 0
    if len(articles) > 20:
        articles = articles[:20]
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

def summarize(final_str, per=0.6):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(final_str)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary

@app.route("/users", methods = ["POST"])
def create_user_article():

    try:
        name = request.form["name"]
        print(name)

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

            for c in coverpage_news:
                article = nlp (c.text)
                for ent in article.ents:
                    if (ent.label_=="PERSON") & ((ent.text == name) | (ent.text in name.split(' '))):
                        news_list_ner = news_list_ner + " " + c.text
                # print(c.text)
                news = news + " " + c.text


            news_all = news_all + " " + news_list_ner
            # print(df['published'])
            if(len(news_list_ner)> 1):
                result = sentiment(news_list_ner)
                a.append({"test" : news_list_ner, "date" : df["published"][i], "sentiment" : result})
            
        main_result = sentiment(news_all)

        new = { "name" : name,  "articles": a , "sentiment_analysis" : main_result, "summary" : summarize(news_all)}
        # print(name)
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


@app.route("/getusers", methods = ["POST"])
def get_some_users():

    try: 
        data = db.users.find_one({"name": request.form["name"]}, {"_id" : False})

        return Response(
            response = json.dumps(data),
            status = 200,
            mimetype = "application/json")
    
    except Exception as ex:
        print(ex)
        return Response(
            response = json.dumps({"message":"cannot read users"}),
            status = 500,
            mimetype = "application/json")





if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))






            
