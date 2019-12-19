from random import randint
import numpy as np
import torch
from models import InferSent
model_version = 2
MODEL_PATH = "infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

use_cuda = True
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else '../../model/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)

import csv


def get_top_business_reviews(business_reviews, review_weight):
    desc_business_rev = {}
    desc_rev_weight = sorted(review_weight, key=review_weight.get, reverse=True)
    for bid in review_weight:
        indexes = sorted(range(len(review_weight[bid])), key=lambda i: review_weight[bid][i], reverse=True)[:20]
        desc_business_rev[bid] = [business_reviews[bid][indexes[0]]]
    return desc_business_rev

def get_agg_business_reviews():
    print("==Reading business dataset==")
    bid_to_num = {}
    business_reviews = {}
    review_weight = {}
    reader = csv.DictReader(open("../../data/2018-review-LasVegas-Restaurants.csv",encoding='utf-8'))
    for row in reader:
        bid = row['business_id']
        if bid not in bid_to_num:
            bid_to_num[bid] = count
            count += 1

        if bid not in business_reviews:
            business_reviews[bid] = [row['text']]
            review_weight[bid] = [int(row['useful']) + int(row['cool']) + 1]
        else:
            business_reviews[bid].append(row['text'])
            review_weight[bid].append(int(row['useful']) + int(row['cool']) + 1)

    print("==Reading business dataset finished==")

    business_reviews = get_top_business_reviews(business_reviews, review_weight)
    return business_reviews, review_weight, bid_to_num


def get_business_embedding():
    print("==Preparing business embedding==")
    business_reviews, review_weight, bid_to_num = get_agg_business_reviews()

    business_embedding = []
    count = 0
    print("Total businesses", len(business_reviews))
    with open("../../output/review_embedding.txt","w") as fw:
        for bid, reviews in business_reviews.items():
            print(count)
            embeddings = model.encode(reviews, bsize=128, tokenize=False, verbose=True)
            normalized_weighted_embdding = get_weighted_embedding(embeddings,review_weight[bid])
            fw.write(str(bid_to_num[bid])+"\t"+ str(normalized_weighted_embdding)+"\n")
            # business_embedding.append((bid_to_num[bid],normalized_weighted_embdding))
            count += 1
    print("Total restaurant", count)
    print("==Reading business embedding finished==")
    print()
    return None


def get_weighted_embedding(review_embedding, review_weights):
    weighted_emb = np.zeros(4096, )
    total = 0
    for weight, rev_emb in zip(review_weights, review_embedding):
        weighted_emb += weight * np.array(rev_emb)
        total += int(weight)
    weighted_emb = weighted_emb / total
    return  weighted_emb.tolist()


if __name__=="__main__":
    get_business_embedding()