import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from annoy import AnnoyIndex

embed = hub.Module('../model/content-512-dim')
dim = 512
an = AnnoyIndex(dim, 'angular')

def add_annoy(ids,embedding):
    for k,v in zip(ids,embedding):
        an.add_item(k,v)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    def get_embedding(messages):
        message_embeddings = session.run(embed(messages))
        return message_embeddings

    reviews_emb, ids, ids_to_num, count = [], [], {}, 0
    with open("../data/review.json", "r") as fr:
        for idx, line in enumerate(fr):
            data = json.loads(line)
            if count > 1000:
                print(idx/1000)
                embedding = get_embedding(reviews_emb)
                add_annoy(ids,embedding)
                reviews_emb, ids = [], []
                count = 0
            reviews_emb.append(data['text'])
            ids.append(idx)
            ids_to_num[idx] = data['review_id']
            count+=1
    
    an.build(10)
    an.save('../model/review.ann')
    with open('../output/num_to_review','w') as fw:
        json.dump(ids_to_num,fw)