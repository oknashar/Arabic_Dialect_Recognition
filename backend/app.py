from cgitb import text
from os import name
from unittest import result
from flask import Flask, request, abort, jsonify , render_template
import numpy as np
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

def create_app(test_config=None):
        # create and configure the app
    model_ml = load('SVMClassifier.joblib')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model_dl = tf.keras.models.load_model('Lstm_model.hdf5')    
    app = Flask(__name__)
    @app.route("/")
    def Home():
        return render_template("index.html",result="arabic")
    
    @app.route('/',methods = ['POST', 'GET'])
    def Predict_machine_learning():
        if request.method == 'POST':
            countries=np.array(['United Arab Emirates','Bahrain','Algeria','Egypt','Iraq','Jordan','Kuwait','Lebanon','Libya','Morocco','Oman','Palestine','Qatar','Saudi Arabia','Sudan','Syrian','Tunisia','Yemen'])
            Arab_countries=np.array(['الامارات','البحرين','الجزائر','مصر','العراق','الاردن','الكويت','لبنان','ليبيا','’المغرب','عمان','فلسطين الحره','قطر','السعوديه','السودان','سوريا','تونس','اليمن'])
            title=request.form['search']
            title=clean_text(title)
            res1,res2 = predict_ml(model_ml,title,countries,Arab_countries)
            return render_template("index.html",title=title,result=res1,ar_coun=res2)
        
    @app.route('/deep',methods = ['POST', 'GET'])
    def Predict_deep_learning():
        if request.method == 'POST':
            countries=np.array(['United Arab Emirates','Bahrain','Algeria','Egypt','Iraq','Jordan','Kuwait','Lebanon','Libya','Morocco','Oman','Palestine','Qatar','Saudi Arabia','Sudan','Syrian','Tunisia','Yemen'])
            Arab_countries=np.array(['الامارات','البحرين','الجزائر','مصر','العراق','الاردن','الكويت','لبنان','ليبيا','’المغرب','عمان','فلسطين الحره','قطر','السعوديه','السودان','سوريا','تونس','اليمن'])

            title=request.form['search2']
            title=clean_text(title)
            res1,res2 = predict_dl(model_dl,title,countries,Arab_countries)
            return render_template("index.html",title=title,result=res1,ar_coun=res2)
        
        
        

        
    def remove_hashtags(text):
        text = re.sub(r'@\w+', '', text)
        return text

    def remove_emojis(text):
        text = [x for x in text.split(' ') if x.isalpha()]
        text = ' '.join(text)
        return text
    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    def remove_urls(text):
        text = re.sub(r'http\S+', '', text)
        return text
    def preprocess(text):
        text = remove_hashtags(text)
        text = remove_emojis(text)
        text = remove_urls(text)
        return text


    def clean_text(text):  

        search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى",
                "\\",'\n', '\t','&quot;','?','؟','!']
        replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا",
                "","","","ي","",' ', ' ',' ',' ? ',' ؟ ', ' ! ']
        
        special_chars=['\n','\t','&quot;','?','؟','!','.','،',',','؛']
        #remove tashkeel
        tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(tashkeel,"", text)
    
        text =remove_emojis(text)
        text=remove_hashtags(text)
        text = remove_urls(text)
        
        text = re.sub(r"[^\w\s]", '', text)
        #remove english words
        text = re.sub(r"[a-zA-Z]", '', text)
        #remove spaces
        text = re.sub(r"\d+", ' ', text)
        text = re.sub(r"\n+", ' ', text)
        text = re.sub(r"\t+", ' ', text)
        text = re.sub(r"\r+", ' ', text)
        text = re.sub(r"\s+", ' ', text)
        #remove repetetions
        text = text.replace('وو', 'و')
        text = text.replace('يي', 'ي')
        text = text.replace('اا', 'ا')
        
            
        for i in range(0, len(search)):
            text = text.replace(search[i], replace[i])
        
            
            
        text = text.strip()
        
        return text
        
    def predict_ml(model,text,country,ara_country):
        text =pd.Series(text)
        p=model.predict(text)
        print(p)
        res1=country[p][0]
        res2=ara_country[p][0]
        return res1,res2
    
    def predict_dl(model,text,countries,ara_countries):
        text =pd.Series(text)
        text1=tokenizer.texts_to_sequences(text)
        text1=pad_sequences(text1, padding='post', maxlen=20)
        p=model.predict(text1)
        res1=countries[np.argmax(p,axis=1)][0]
        res2=ara_countries[np.argmax(p,axis=1)][0]
        return res1,res2
    
    
    
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": 404,
            "message": "resource not found"
        }), 404

    @app.errorhandler(422)
    def unprocessable(error):
        return jsonify({
            "success": False,
            "error": 422,
            "message": "unprocessable"
        }), 422

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "success": False,
            "error": 400,
            "message": "bad request"
        }), 400

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "success": False,
            "error": 405,
            "message": "method not allowed"
        }), 405

    return app
