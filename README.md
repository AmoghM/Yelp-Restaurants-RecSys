### Steps to follow for recommendation on Yelp dataset
a) Pre-run step:
* Clone the repository
* Install packages and dependencies: `pip install -r requirements.txt`
* Download Yelp dataset from here: https://www.yelp.com/dataset/challenge
* Place the extracted folder into the `data/` of the repository
* Run `Data_Pre_Processing.ipynb` from `src` folder. This will create the relevant datasets necessary to run the models we have defined 

b) For Bias Baseline and ALS model:
* Run `src/ALS_Baseline.ipynb`

c) For Factorization Machine model:
* Run `src/CMF_FM.ipynb`

d) For Wide and Deep model:
* Run `src/Wide and Deep.ipynb`

e) For Content-based recommendation:
* Download glove, infersent model mentioned here: https://github.com/facebookresearch/InferSent
* Move the `infersent2.pkl` to `src/Content-Recommendation`
* Run `python src/json_to_csv.py` to convert json to csv consisting of Las Vegas restaurant dataset for 2018.
* Run `python src/Content-Recommendation/get_review_embedding.py` to generate weighted review2vec and export it to a file.
* Run `python src/Content-Recommendation/content_recommendation.py` to create annoy index from review embeddings and provide top 10 recommendations for the input string.

### Models Used:
* [Wide and Deep Recommendation System](https://arxiv.org/pdf/1606.07792.pdf)
![Wide and Deep Recommendation model](https://github.com/AmoghM/Yelp-Restaurants-RecSys/blob/master/images/wide-deep-collage.png)

* [Infersent](https://arxiv.org/pdf/1705.02364.pdf)
![Sentence Embedding](https://github.com/AmoghM/Yelp-Restaurants-RecSys/blob/master/images/infersent.JPG)

* [Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf)
![USE](https://github.com/AmoghM/Yelp-Restaurants-RecSys/blob/master/images/use.JPG)

* [KGAT: Knowledge Graph Attention Network for
Recommendation](https://arxiv.org/pdf/1905.07854.pdf)
![KGAT](https://github.com/AmoghM/Yelp-Restaurants-RecSys/blob/master/images/KGAT.JPG)

* [Graph Embedding Based Hybrid Social
Recommendation System](https://arxiv.org/pdf/1908.09454.pdf)
![GraphEmb](https://github.com/AmoghM/Yelp-Restaurants-RecSys/blob/master/images/Graph-embedding.JPG)

* [Alternating Least Square](https://dl.acm.org/citation.cfm?id=1608614)
![ALS](https://github.com/AmoghM/Yelp-Restaurants-RecSys/blob/master/images/ALS.png)
* Factor Machines 
![FM](https://github.com/AmoghM/Yelp-Restaurants-RecSys/blob/master/images/Factorization-Machine.png)


### Team
L-R: Benjamin, Siddhant, Swarna and Amogh
![team](https://github.com/AmoghM/Yelp-Restaurants-RecSys/blob/master/images/team.jpeg)
