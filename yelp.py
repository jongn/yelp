import numpy as np
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import json
import csv
import unicodecsv
import codecs
from collections import defaultdict
import sys
import lda
reload(sys)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)


"""
trouble@stat.berkeley.edu

pylda
"""

valid_ids = defaultdict(lambda: 'does not exist')
reviews = []
review_text = []
review_star = []

def business_filter():
    
    with codecs.open('yelp_academic_dataset_business.json', 'rU', 'utf-8') as f:
        for line in f:
            business_data = json.loads(line)
            categories = business_data['categories']
            if categories != None:
                if 'Restaurants' in categories:
                    valid_ids[business_data['business_id']] = business_data['name']
    print(len(valid_ids))
    f = unicodecsv.writer(open('restaurants.csv', 'wb+'), encoding='utf-8')
    f.writerow(['business_id', 'name'])
    for key, value in valid_ids.items():
        f.writerow([key, value])



def review_filter():
    with codecs.open('yelp_academic_dataset_review.json', 'rU', 'utf-8') as f:
        for line in f:
            review_data = json.loads(line)
            if valid_ids[review_data['business_id']] != 'does not exist':
                reviews.append(review_data)
    print(len(reviews))
    f = unicodecsv.writer(open('reviews.csv', 'wb+',), encoding='utf-8')
    f.writerow(['review_id', 'business_id', 'stars', 'date', 'text', 'useful', 'funny', 'cool', 'type'])
    for review in reviews:
        f.writerow([review['review_id'], review['business_id'], review['stars'], review['date'], review['text'], review['useful'], review['funny'], review['cool'], review['type']])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

def load():
    with open('reviews.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        # TODO: make this random 
        head = [next(reader) for x in xrange(10000)]
        for line in head:
            review_text.append(line[4])
            review_star.append(line[2])
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(review_text)
    model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    model.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    #print_top_words(lda, tf_feature_names, 20)
    doc_topic = model.doc_topic_
    for i in range(0, 10):
        print ("{} (top topic: {})".format(review_text[i], doc_topic[i].argmax()))
        print (doc_topic[i].argsort()[::-1][:3])
    for i, topic_dist in enumerate(model.topic_word_):
        topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-50:-1]
        print ('Topic {}'.format(i))
        print (' '.join(topic_words))
    """
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    """


def main():
    #business_filter()
    #review_filter()
    load()

if __name__ == "__main__":
    main()