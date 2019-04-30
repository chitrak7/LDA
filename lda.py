import re
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model,svm
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split                   # regular expression
from nltk.corpus import stopwords
from collections import defaultdict
import gensim
import numpy as np
from os import listdir
import math
import scipy
import matplotlib.pyplot as plt


log_gamma = scipy.special.gammaln


def transformText(text):
    stops = set(stopwords.words("english"))
    # Convert text to lower
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]',r' ',text)     # Removing non ASCII chars
    text = re.sub(r'(<br /><br />)',r' ',text)
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 1] # filter out short tokens
    filtered_words = [word for word in tokens if word not in stops]
    filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=3)
    text = " ".join(filtered_words)
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    text = gensim.parsing.preprocessing.strip_numeric(text)
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    frequency = defaultdict(int)
    text = text.split()
    for token in text:
            frequency[token] += 1

    text = [token for token in text if frequency[token] > 1]
    s = " "
    return s.join(text)

def reduce_vocab(docs):
    frequency = defaultdict(int)
    for text in docs:
        for token in text:
            frequency[token] += 1

    docs = [[token for token in text if (frequency[token] > 0 and frequency[token] < 4000)] for text in docs]
    return docs

def helper_theta(gamma, lambdan, v, K):
    t = []
    for k in range(K):
        a = log_gamma(gamma[k]) - log_gamma(np.sum(gamma))
        if(gamma[k]==0):
            a = -np.inf
        b = log_gamma(lambdan[k][v]) - log_gamma(np.sum(lambdan[k]))
        if(lambdan[k][v]==0):
            b = -np.inf
        if(a+b==np.inf):
            t.append(-np.inf)
        else:
            t.append(a+b)
    t = np.array(t)
    mx = np.amax(t)
    #print(mx)
    t  = t - mx
    t  = np.exp(t)
    t  = t/np.sum(t)
    return t

def learning_rate(i):
    return np.power(1+i, -0.55)

def remove_dup(text):
    frequency = defaultdict(int)
    r = []
    for token in text:
        if(frequency[token] == 1):
            continue
        frequency[token] = 1
        r.append(token)
    return r


File = open('nytimes_news_articles.txt', encoding='utf-8').read()
docs = File.split("URL")[:2000]
docs = [transformText(s.split("\n",1)[1]) for s in docs if (len(s.split("\n",1))>1)]
'''
docs = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
'''
tokens = reduce_vocab([remove_dup(s.split()) for s in docs])
dictionary = gensim.corpora.Dictionary(tokens)
doc_idx = [np.array(dictionary.doc2idx(i)) for i in tokens]
V = len(dictionary.token2id)
D = len(docs)
K = 40
print(V,D,K)

eta = 0.01
eta_arr = eta*np.array([1 for i in range(V)])
alpha = 1/K
alpha_arr = alpha*np.array([1 for i in range(K)])
batch_size = 50
epochs     = 1001
lambda_n = np.random.dirichlet(eta_arr, K)
x_plot = []
nlp_plot = []
for epoch in range(epochs):
    print("Epoch : ", epoch)
    lambda_g = np.zeros(lambda_n.shape)
    for batch in range(batch_size):
        gamma  = np.array([1 for i in range(K)])
        d      = np.random.randint(D)
        gammap = 0
        N      = doc_idx[d].shape[0]
        thetap = np.zeros((doc_idx[d].shape[0],K))
        theta = np.random.uniform(size=(doc_idx[d].shape[0],K))
        rN    = range(N)
        while(np.linalg.norm(gamma-gammap)>=0.1 and np.linalg.norm(thetap-theta) >=1):
            thetap = theta
            gammap = gamma
            theta = np.array([helper_theta(gamma, lambda_n,doc_idx[d][n], K) for n in rN])
            gamma = alpha_arr + np.sum(theta, axis=0)
        for k in range(K):
            for n in range(N):
                lambda_g[k][doc_idx[d][n]] += theta[n][k]

    lambda_g = lambda_g*(D/batch_size)
    for k in range(K):
        lambda_g[k] += eta_arr
    rho      = learning_rate(epoch)
    lambda_n = lambda_n*(1-rho) + lambda_g*rho



ct = 0
for i in lambda_n:
    for k in i:
        if(k!=0.01):
            ct+= 1
print(ct)
for k in range(10):
    idx = np.argsort(lambda_n[k])[:10]
    print("k=",k)
    for i in idx:
        print(dictionary[i])
print(lambda_n)
