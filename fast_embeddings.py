import json
from datasets import Dataset
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from sentence_transformers import SentenceTransformer
import csv


model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
print("finish loading glove model")
# Define the dataset path
dataset_path = "/media/ty2806/data/cc_news"

# Define the dataset object
dataset = Dataset.load_from_disk(dataset_path)
print("load dataset from disk")

with open('autocast_competition_test_set.json') as f:
    test_questions = json.load(f)
print("load test questions")

def get_embedding(text):
    embeddings = model.encode([text])
    return embeddings


def get_embedding_in_batch(text):
    embeddings = model.encode(text)
    return embeddings

question_ids = set()
question_dates = {}
question_embeddings = []

for test_question in tqdm(test_questions):
    # Extracting date only from publish_time and close_time
    if test_question["id"] in question_ids:
        continue
    question_ids.add(test_question["id"])
    test_embedding = get_embedding(test_question["background"]+" "+test_question["question"])
    question_embeddings.append(test_embedding[0])
    publish_date = datetime.strptime(test_question['publish_time'].split(" ")[0], '%Y-%m-%d').date()
    close_date = datetime.strptime(test_question['close_time'].split(" ")[0], '%Y-%m-%d').date()

    current_date = publish_date
    while current_date < close_date:
        if str(current_date) in question_dates:
            question_dates[str(current_date)].append(len(test_embedding)-1)
        else:
            question_dates[str(current_date)] = [len(test_embedding)-1]
        current_date += timedelta(days=1)
question_embeddings = np.array(question_embeddings)
print('build test embeddings')

news_date = dataset['date']
dates = [d for d in news_date]
dates_dict = {}
for i, value in tqdm(enumerate(dates)):
    if value in question_dates:
        if value not in dates_dict:
            dates_dict[value] = [i]
        else:
            dates_dict[value].append(i)
print('build cc_news date dict')


filtered_dates_dict = {}
length_of_dates = len(dates_dict)
for index, date in tqdm(enumerate(dates_dict)):
    news_current_day = []
    batch_size = 1000
    for news_index in range(0, len(dates_dict[date]), batch_size):
        news_batch = dates_dict[date][news_index:news_index+batch_size]
        news_batch = [dataset[i]['title']+" "+dataset[i]['description'] for i in news_batch]
        news_embedding = get_embedding_in_batch(news_batch)
        news_current_day.append(news_embedding)
    news_current_day = np.vstack(news_current_day)
    test_embeddings = question_embeddings[question_dates[date], :]
    # calculate the cosine distances between test embeddings and news embeddings
    distances = cosine_distances(test_embeddings, news_current_day)
    print("distance:", distances.shape)
    # find the union of smallest k distances for each row of test embeddings
    k = 5  # set the value of k
    smallest_indices = np.argsort(distances, axis=1)[:, :k]
    print("smallest_indices:", smallest_indices.shape)
    union_indices = np.unique(smallest_indices)
    filtered_dates_dict[date] = np.array(dates_dict[date])[union_indices]
    if index > 10:
        print(filtered_dates_dict)
        break

# open a new CSV file in write mode
with open('my_dict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # write the header row with the dictionary keys
    for key in filtered_dates_dict:
        writer.writerow([key] + list(filtered_dates_dict[key]))

