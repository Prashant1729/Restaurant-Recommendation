from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################
preds_df_NMF = pd.read_csv("C:/Users/Admin/Desktop/Diploma project/preds_df_NMF.csv")
preds_df_NMF = preds_df_NMF.set_index('customer_id')
users_10_features = pd.read_csv("C:/Users/Admin/Desktop/Diploma project/users_10_features.csv")
users_10_features = users_10_features.set_index('customer_id')
result_customers_clusters = pd.read_csv("C:/Users/Admin/Desktop/Diploma project/result_customers.csv")
result_customers_clusters = result_customers_clusters.set_index('customer_id')

def location(loc):
    if loc == "Work":
        return 0
    elif loc =="Home":
        return 1
    else:
        return 2

def favorite(f):
    if f == 'YES':
        return 0
    elif f =='NO':
        return 1
    else:
        return 2

def gender(f):
    if f == 'Male':
        return 0
    elif f =='Female':
        return 1
    else:
        return 2

def vendor_category(f):
    if f == 'Restaurants':
        return 0
    else:
        return 1


def convert(to_predict_list):
    features = []
    global location,fav_vendor,gender,vendor_category,morning,afternoon,evening
    customer_id = to_predict_list['customer_id']
    vendor_id = to_predict_list['vendor_id']
    location = location(to_predict_list['Location'])
    fav_vendor = favorite(to_predict_list['fav_vendor'])
    gender = gender(to_predict_list['gender'])
    vendor_category = vendor_category(to_predict_list['vendor_category'])
    vendor_rating = to_predict_list['vendor_rating']
    driver_rating = to_predict_list['driver_rating']
    user_features = users_10_features.loc[customer_id]
    vendor_rating1 = preds_df_NMF.loc[customer_id][vendor_id]
    cluster_id = result_customers_clusters.loc[customer_id]['cluster_ID']

    if 'morning' in to_predict_list:
        morning = 1
    else:
        morning = 0

    if 'afternoon' in to_predict_list:
        afternoon = 1
    else:
        afternoon = 0

    if 'evening' in to_predict_list:
        evening = 1
    else:
        evening = 0


    features = [gender,vendor_category,location,vendor_rating1,vendor_id,vendor_rating,fav_vendor]
    features.extend(user_features)
    features.extend([cluster_id,morning,afternoon,evening,driver_rating])
    features = np.array(features).reshape(1,-1)
    return features

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('finalized_RF_model.pkl')
    to_predict_list = request.form.to_dict()
    datapoint = convert(to_predict_list)
    #print(to_predict_list)
    #return datapoint[0]
    #preds_df_NMF.loc[customer_id]
    #review_text =
    pred = clf.predict(datapoint)
    print(pred)
    #return pred[0]
    if pred[0]:
        prediction = "Recommend"
    else:
        prediction = "Do not Recommend"

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
