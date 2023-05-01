import csv
import datetime
import openai
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import backoff
from datasets import Dataset

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

# load cc news
dataset_path = "/media/ty2806/data/cc_news"
dataset = Dataset.load_from_disk(dataset_path)
print("load dataset from disk")

with open('autocast_competition_test_set.json') as f:
    test_questions = json.load(f)

similar_news = {}
with open('final_result_dict.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 0:
            similar_news[row[0]] = [int(index) for index in row[1:]]

# Define function to generate text from the API
def generate_chatgpt_response(model, message):
    response = openai.ChatCompletion.create(
        model=model,
        messages=message,
        temperature=1,
        n=5
    )
    reply = []
    for i in range(len(response['choices'])):
        reply_content = response['choices'][i]['message']['content']
        reply.append(reply_content)
    return reply

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_gpt3_response(model, prompt):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1024,
        n=5,
        stop=None,
        temperature=1,
    )
    reply = []
    for i in range(len(response['choices'])):
        reply_content = response['choices'][i].text.strip()
        reply.append(reply_content)
    return reply


def create_message(user_input):
    message = [
        {"role": "system",
         "content": "You are a financial analysis assistant AI. Your responsibility is to read news and forecast all kinds of events. For all answers you will only reply with a number. If you find the question unclear or unfathomable, you will still output a number. When you give your answer, put the number in a pair of <output>."},
        {"role": "user",
         "content": "I will give you some news. Then give you a question to answer. You will try your best to give your answer. Even if you do not know the answer, you must still guess an answer. Do you understand?"},
        {"role": "assistant",
         "content": "<output>1<output>"},
        {"role": "user",
         "content": user_input}
    ]
    return message


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_result(reply, qtype, test_question):
    for i in reply:
        try:
            answer = i.split("<output>")[1].split(",")
            answer = [float(i) for i in answer]
            if qtype == 't/f':
                ans = np.array(answer) / sum(answer)
                if len(answer) != 2:
                    continue

            elif qtype == 'mc':
                ans = np.array(answer) / len(answer)
                if len(answer) != len(test_question["choices"]):
                    continue

            elif qtype == 'num':
                ans = float(answer)
                if ans > 1:
                    print("handled", reply, i, test_question)
                    ans = ans/(test_question["choices"]["max"]+test_question["choices"]["min"])
            return ans
        except:
            continue
    print("parse reply failed")
    print(test_question["id"] + test_question["question"])
    print(reply)
    if qtype == 't/f' or qtype == 'mc':
        ans = np.ones(len(test_question['choices']))
        ans = ans / len(test_question['choices'])

    elif qtype == 'num':
        ans = 0.5
    return ans




chatgpt_model = "gpt-3.5-turbo"
gpt3_model = "text-davinci-003"

def process_question(test_question):
    question = test_question['background'] + " " + test_question['question']
    news_aggregate = ""
    for news_index in similar_news[test_question['id'][:5]]:
        news = dataset[news_index]
        news_aggregate += news['title']+" "+news['description']+"\n"
    user_input = news_aggregate + question + "\n"
    if test_question['qtype'] == 't/f':
        user_input += "You should choose between 1.yes or 2.no. Output 2 float numbers range from 0 to 1 as your confidence score of No and Yes respectively. Remember only output the number and nothing else."

    elif test_question['qtype'] == 'mc':
        user_input += "Your Answer should be one of :\n"
        for i, choice in enumerate(test_question["choices"]):
            user_input += str(i)+"."+choice+"\n"
        user_input += "You should choose between options. Give "+str(len(test_question["choices"]))+" float numbers ranging from 0 to 1 as your confidence score of each option respectively. Remember only output the confidence scores and nothing else. If you find the question unclear or unfathomed, you will output a lower confidence score."

    elif test_question['qtype'] == 'num':
        user_input += "Your Answer should be a float number from 0 to 1. It represents a number between "\
                      +str(test_question["choices"]["min"])+" and "\
                      +str(test_question["choices"]["max"]) + ". Remember to give a float number from 0 to 1."
    user_input += "\nWhen you give your answer, put it in a pair of <output>. For example <output>0.7, 0.3<output> "

    if datetime.datetime.strptime(test_question['close_time'].split(" ")[0], '%Y-%m-%d').date() < datetime.date(2021,10,1):
        reply = generate_gpt3_response(gpt3_model, user_input)
    else:
        reply = generate_chatgpt_response(chatgpt_model, create_message(user_input))

    ans = parse_result(reply, test_question['qtype'], test_question)
    return ans

executor = ThreadPoolExecutor()
preds = []
for pred in tqdm(executor.map(partial(process_question), test_questions), total=len(test_questions)):
    preds.append(pred)
print(preds)
if not os.path.exists('submission'):
    os.makedirs('submission')

with open(os.path.join('submission', 'predictions.pkl'), 'wb') as f:
    pickle.dump(preds, f, protocol=2)
