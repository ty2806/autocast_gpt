import json
import re
from datetime import datetime

from IPython.lib.pretty import pprint

subscription_key = "key"
assert subscription_key

search_url = "https://api.bing.microsoft.com/v7.0/search"


import requests

headers = {"Ocp-Apim-Subscription-Key": subscription_key}
dic={}

test_questions = json.load(open('autocast_competition_test_set.json',encoding='utf-8') )


count=0.0
total = len(test_questions)
for question in test_questions:
    print(count/total*100, "%")
    count+=1

    q=question['question']
    start=str(datetime.strptime(question['publish_time'].split(" ")[0], '%Y-%m-%d').date())
    end = str(datetime.strptime(question['close_time'].split(" ")[0], '%Y-%m-%d').date())
    tag = ", ".join(question['tags'])
    q_id = question['id']
    if q_id in dic.keys():
        continue


    freshness=start+".."+end
    search_term = q+" "+tag
    params = {"q": search_term,  "textFormat": "HTML", "responseFilter":"webpages", "freshness":freshness}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    lis = []
    try:
        for i in search_results['webPages']['value']:
            s=i['name']+" "+i['snippet']
            result = re.sub('<[^>]+>', '', s)
            lis.append(result)
        dic[q_id]=lis
    except KeyError:
        try:
            search_term = q
            params = {"q": search_term,  "textFormat": "HTML", "responseFilter":"webpages", "freshness":freshness}
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            lis = []
            for i in search_results['webPages']['value']:
                s=i['name']+" "+i['snippet']
                result = re.sub('<[^>]+>', '', s)
                lis.append(result)
            dic[q_id]=lis
        except:
            print(search_results)

with open("my_dict1.txt", "w") as file:
    file.write(json.dumps(dic))