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
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from spacy import displacy
from collections import Counter





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
                    if (ent.label_ == "PERSON" and (ent.text == "Shri Narendra Modi" or func("Shri Narendra Modi",ent.text))):  #and ent.text == "Anthony Albanese"
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
                result = sentiment(news_list_ner)
                a.append({"test" : news, "date" : df["published"][i], "sentiment" : result})
            
        main_result = sentiment(news_all)

        new = { "name" : name,  "articles": a , "sentiment_analysis" : main_result}
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

