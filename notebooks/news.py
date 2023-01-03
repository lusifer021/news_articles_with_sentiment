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
nlp = spacy.load("en_core_web_sm")

from flask import Flask, Response, request
app = Flask(__name__)
import pymongo
import json
from bson.objectid import ObjectId
try:
    mongo = pymongo.MongoClient(host = "mongodb://localhost",
                 port = 27017,
                serverSelectionTimeoutMS = 1000)
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
    articles = articles[:20]
    for i in articles:
        # if () :          #if date in 
        #     pass
        # else:
        #     pass

        article = {'title':i["title"],'link':i['url'],'published':i['publishedAt']}
        news.append(article)
        # break
    # print(news)
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
        # user = {"name" : request.form["name"] , "lastName": request.form["lastName"]}

        # cur = db.news.find({"name": request.form["name"]})
        # results = list(cur)
        # print(len(results))
        # print(cur["name"])
        # print(db.users.count_documents({"name": request.form["name"]}))
        if db.users.count_documents({"name": request.form["name"]},limit =1)==0:
            
            name = request.form["name"]
             # function to create a dictionary oftitle and sentiment
            

            data = get_data(name)
            df = pd.DataFrame(data)

            a = []
            news = ""
            news_list_ner = ""
            #Genralising what we have done above
            for i in range(df.shape[0]):
                
                
                url = df["link"][i]
                r1 = requests.get(url)
                

                # We'll save in coverpage the cover page content
                coverpage = r1.content

                # Soup creation
                soup1 = BeautifulSoup(coverpage, 'html5lib')
                # print(soup1)
                # News identification using ner
                coverpage_news = soup1.find_all('p')

                #Using Named Entity Recognition
                for c in coverpage_news:
                    article = nlp (c.text)
                    for ent in article.ents:
                        if (ent.label_=="PERSON") & ((ent.text == name) | (ent.text in name.split(' '))):
                            news_list_ner = " ".join(news_list ,c.text)
                    # print(c.text)
                    news = " ".join((news,str(c.text)))

                #converting bs4.element.ResultSet to string so that it can be processeable
                # coverpage_news = str(news_list[0])
                
                #cleaning the html tags 
                # coverpage = cleanhtml(coverpage_news)
                # result = sentiment(coverpage)
                
                # a.append(result)
            result = sentiment(news_list_ner)
            
            new = {"name" : request.form["name"], "date" : list(df["published"]), "articles" : [{"text": news, "sentiment": result}]}

            dbResponse = db.users.insert_one(new)
            print(dbResponse.inserted_id)
            # for attr in dir(dbResponse):
            #     print(attr)
            return Response(
                response = json.dumps({"message":"user created", "id": f"{dbResponse.inserted_id}"}),
                status = 200,
                mimetype = "application/json"
            )
        else:
            name = request.form["name"]
             # function to create a dictionary oftitle and sentiment
            dates =  db.users.find_one({"name": name})["date"]
            # print(dates)
            news = []
            all_articles = newsapi.get_everything(
                q=name,
                language='en',   
            )
            articles = all_articles["articles"]
            for i in articles:
                
                article = {'title':i["title"],'link':i['url'],'published':i['publishedAt']}
                news.append(article)
                
            
            data = news
            df = pd.DataFrame(data)
            a = []
            #Genralising what we have done above
            news_list = ["Modi is good"]
            news = ""
            for i in range(df.shape[0]):
                # news_list = ["Modi is good"]
                # news = ""
                url = df["link"][i]
                r1 = requests.get(url)
                

                # We'll save in coverpage the cover page content
                coverpage = r1.content

                # Soup creation
                soup1 = BeautifulSoup(coverpage, 'html5lib')

                # News identification using ner
                coverpage_news = soup1.find_all('p')

                #Using Named Entity Recognition
                for c in coverpage_news:
                    article = nlp (c.text)
                    for ent in article.ents:
                        if (ent.label_=="PERSON") & ((ent.text == name) | (ent.text in name.split(' '))):
                            news_list.append(c.text)
                    news = " ".join((news,c.text))
                
                print(news)

                #converting bs4.element.ResultSet to string so that it can be processeable
                # print(news_list)
                coverpage_news = str(news_list[0])
                
                #cleaning the html tags 
                coverpage = cleanhtml(coverpage_news)
                result = sentiment(coverpage)
                
                a.append(result)

            dbResponse1 = db.users.update_one({'name': name}, {'$push': {'date': df["published"][0]}})
            newvalue = {"$push": {'articles': {"text": news, "sentiment": a[0]}}}
            dbResponse2 = db.users.update_one({'name':name}, newvalue)
            print(dbResponse1.modified_count)
            if dbResponse1.modified_count + dbResponse2.modified_count ==  2:
                return Response(
                    response = json.dumps({"message":"user updated"}),
                    status = 200,
                    mimetype = "application/json"
                )
            else:
                return Response(
                    response = json.dumps({"message":"nothing to update"}),
                    status = 200,
                    mimetype = "application/json"
                )
    except Exception as ex:
        print(ex)
        return Response(
            response = json.dumps({"message":"sorry cannot update user"}),
            status = 500,
            mimetype = "application/json"
        )



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







if __name__ == "__main__":
    app.run(port = 8081, debug =True)