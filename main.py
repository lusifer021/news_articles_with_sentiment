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

from flask import Flask, Response, request, jsonify
app = Flask(__name__)
import pymongo
import json
from bson.objectid import ObjectId
try:
    mongo = pymongo.MongoClient(host = "mongodb+srv://lusifer021:ilovemango@cluster0.ibnfukw.mongodb.net/?retryWrites=true&w=majority",
                 
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


@app.route("/users", methods = ["POST"])
def create_user_article():

    try:
        name = request.form["name"]
        print(name)

        if db.users.count_documents({"name": name},limit =1)>0:
            db.users.delete_one({"name" : name})
        
        data = get_data(name)
        if data == 0:
            return  jsonify({"message":"no articles found with this name"})
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
                a.append({"test" : news, "date" : df["published"][i], "sentiment" : result})
            
        main_result = sentiment(news_all)

        new = { "name" : name,  "articles": a , "sentiment_analysis" : main_result}
        # print(name)
        dbResponse = db.users.insert_one(new)

        print(dbResponse.inserted_id)
        # print(new)
        return jsonify(db.users.find_one({"name": name},{"_id" : False}))
    except Exception as ex :
        print(ex)
        return jsonify({"message":"cannot read users"})


@app.route("/getusers", methods = ["POST"])
def get_some_users():

    try: 
        data = db.users.find_one({"name": request.form["name"]}, {"_id" : False})

        return jsonify(data)
    
    except Exception as ex:
        print(ex)
        return jsonify({"message":"cannot read users"})





if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))






            
