import json
import datasets
from datasets import Dataset
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR('hkunlp/instructor-xl')
model.max_seq_length = 512
print("finish loading instructor model")
# Define the dataset path
dataset_path = "/media/ty2806/data/cc_news"

# Define the dataset object

dataset = Dataset.load_from_disk(dataset_path)
print("load dataset from disk")

with open('autocast_competition_test_set.json') as f:
    test_questions = json.load(f)
print("load test questions")

def get_retrieval_embedding(text):
    instruction = "Represent the news article for retrieval: "
    embeddings = model.encode([[instruction, text]])
    return embeddings


def get_query_embedding(text):
    instruction = "Represent the news question for retrieving supporting documents: "
    embeddings = model.encode([[instruction, text]])
    return embeddings

test_id = "G1851"
target_question = None

for test_question in test_questions:
    # Extracting date only from publish_time and close_time
    if test_question["id"] == test_id:
        target_question = test_question

if target_question is None:
    print("Error: no test question has id:" + test_id)
    exit()

publish_date = datetime.strptime(target_question['publish_time'].split(" ")[0], '%Y-%m-%d').date()
close_date = datetime.strptime(target_question['close_time'].split(" ")[0], '%Y-%m-%d').date()

current_date = publish_date

while current_date <= close_date:
    news_current_day = []
    news_indices = []
    print(current_date)
    for index in tqdm(range(41944825, 41944825+2000)):
        data = dataset[index]
        if data['date'] == str(current_date):
            news = data['title'] + " " + data['text'] + data['description']
            news_embedding = get_retrieval_embedding(news)
            news_current_day.append(news_embedding.squeeze(0))
            news_indices.append(index)

    print("Number of news in this day:" + str(len(news_indices)))
    target_text = target_question['background'] + target_question['question']
    target_embedding = get_query_embedding(target_text)
    news_current_day = np.array(news_current_day)
    np.save("temp.npy", news_current_day)
    distances = cosine_distances(target_embedding, news_current_day)
    k = 10
    indices = np.argsort(distances[0])[:k]
    smallest = distances[0][indices]
    smallest_indices = np.array(news_indices)[indices]

    print("Question: " + target_text)
    for i in range(k):
        print(smallest[i])
        print(smallest_indices[i], dataset[int(smallest_indices[i])]['title'] + " " + dataset[int(smallest_indices[i])]['description'])
    break
    current_date += timedelta(days=1)