import json
from annoy import AnnoyIndex
from models import InferSent
import torch

model_version = 2
MODEL_PATH = "infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else '../../model/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)

def add_annoy_index(path):
    dim = 4096
    an = AnnoyIndex(dim, 'angular')
    print("STARTED ANNOY INDEX.")
    with open(path,'r',encoding='utf-8') as fr:
        for i in fr:
            data = i.split("\t")
            index = data[0]
            embedding = eval(data[1].strip())
            an.add_item(int(index), embedding)

    an.build(10)
    an.save('../../model/business.ann')
    print("ANNOY INDEXING FINISHED.")

def get_business_data():
    business = {}
    with open('../../data/business.json') as f:
        for i in f:
            data = json.loads(i)
            business[data["business_id"]] = data["name"]
    return business

def get_num_to_bid(bid_to_json):
    n_to_b = {}
    for i,j in bid_to_json.items():
        n_to_b[j]=i
    return n_to_b

def get_recommendation(reviews):
    an = AnnoyIndex(4096, 'angular')
    an.load('../../model/business.ann')
    bid_to_json = json.load(open('../../data/bid.json'))
    num_to_bid = get_num_to_bid(bid_to_json)
    business = get_business_data()
    embeddings = model.encode([reviews], bsize=1, tokenize=False, verbose=True)
    ids = an.get_nns_by_vector(embeddings[0], 10, search_k=-1, include_distances=False)
    result = []
    for i in ids:
        result.append(business[num_to_bid[i]])
    print(result)

if __name__=='__main__':
    path = '../../output/review_embedding.txt'
    add_annoy_index(path)
    get_recommendation("This was the best pizza place ever")
    get_recommendation("Worst ambience ever. Never go to this place")
