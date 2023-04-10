import openai
import os
import json
from datasets import Dataset
from datetime import datetime, timedelta
from tqdm import tqdm

# Set up OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the dataset path
dataset_path = "/media/ty2806/data/cc_news"

# Define the dataset object
dataset = Dataset.load_from_disk(dataset_path)

with open('autocast_competition_test_set.json') as f:
    test_questions = json.load(f)

# Define function to generate text from the API
def generate_text(model, message):
    response = openai.ChatCompletion.create(
        model=model,
        messages=message
    )
    reply_content = response['choices'][0]['message']['content']
    total_token = response['usage']['total_tokens']
    return reply_content, total_token

# Example usage
model = "gpt-3.5-turbo"


def create_message(user_input):
    message = [
        {"role": "system",
         "content": "You are a news reader. Read the following news descriptions and reply the index that relelvent to the question"},
        {"role": "user",
         "content": user_input}
    ]
    return message

# Extracting date only from publish_time and close_time
publish_date = datetime.strptime(test_questions[0]['publish_time'].split(" ")[0], '%Y-%m-%d').date()
close_date = datetime.strptime(test_questions[0]['close_time'].split(" ")[0], '%Y-%m-%d').date()

current_date = publish_date

news_date = dataset['date']
dates = [d for d in news_date]
dates_dict = {}
for i, value in enumerate(dates):
    if value not in dates_dict:
        dates_dict[value] = [i]
    else:
        dates_dict[value].append(i)

while current_date <= close_date:
    news_current_day = []
    count = 0
    for index in dates_dict[str(current_date)]:
        if len(dataset[index]['description']) > 0:
            count += 1
            news = str(count) + " " + dataset[index]['description']
            news_current_day.append(news)
    news_string = "\n".join(news_current_day)
    news_string = news_string + "\n" + "Question:" + test_questions[0]['background']+test_questions[0]['question']
    print(news_string)
    message = create_message(news_string)
    reply_content, total_token = generate_text(model, message)
    print(reply_content, total_token)
    break
    current_date += timedelta(days=1)

