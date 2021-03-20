# CS445
# Nick Murphy
# This code uses partial code from
# https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
# for the purposes of tokenizing data and utilizing gensim
# as well as partial code from
# https://medium.datadriveninvestor.com/simple-text-summarizer-using-nlp-d8aaf5828e68
# for the purposes of utilizing beautiful soup to parse HTML data
# The LSA Summarization selection algorithm is the exact same as detailed in
# Steinberger and Ježek

import numpy as np
import wikipedia
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from rouge import Rouge
from gensim import corpora
from gensim.models import LsiModel
from gensim.matutils import corpus2dense
import matplotlib.pyplot as plt

# clean and create a list of tokenized sentences
def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    p_stemmer = PorterStemmer()
    en_stop = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    texts = []
    for sentence in sentences:
        words = tokenizer.tokenize(sentence)
        stopped_words = [word for word in words if not word in en_stop]
        stemmed_words = [p_stemmer.stem(word) for word in stopped_words]
        texts.append(stemmed_words)
    return texts

# find the length of sentence vector v_i
# as specified in paper
def find_length(S, V):
    def find_dim(S):
        maximum = S[0]
        i = 1
        while (not(i == len(S)) and S[i] > maximum/2 ):
            i += 1
        return i
    dims = find_dim(S)
    lengths = []
    for i in range(0, len(V)): # find length of all sentences
        length = 0
        for j in range(0, dims): # use only the length of important topics
            length += np.power(V[i][j], 2) * np.power(S[j], 2)
        np.sqrt(length)
        lengths.append(length)
    return lengths

# For the purposes of finding only text in these sections
def find_text(nlp):
    sections = ["Methods: Rules, statistics, neural networks", "Statistical methods", "Neural networks"]
    text = ""
    for title in sections:
        text += nlp.section(title)
    return text

# Creates a list that holds sentence lengths and their indices
# then sorts based on sentence length to find the longest (in singular vector space)
# x sentences and returns their indices, where x is num_sent
def find_indices(length, num_sent):
    indices = []
    lengths = []
    for i in range(0, len(length)):
        lengths.append((length[i], i))
    lengths.sort()
    for i in range(0, num_sent):
        indices.append(lengths[i][1])
    return indices

if __name__ == '__main__':
    # collect corpus
    nlp = wikipedia.page("Natural Language Processing")
    num_sentences = 3

    text = find_text(nlp)
    sentences = sent_tokenize(text)
    text = text.lower()
    print(text)

    # tokenize text into sentences
    tokens = tokenize(text)

    # create dictionary of tokens
    dictionary = corpora.Dictionary(tokens)
    sent_term_matrix = [dictionary.doc2bow(sentence) for sentence in tokens]
    # Gensim's LsiModel performs Truncated SVD such that dimension of S = numtopics
    # if numtopics is not specified, performs SVD
    lsamodel = LsiModel(sent_term_matrix, id2word = dictionary)
    print(lsamodel.print_topic(0)) # good check; what terms does the model associate with the most important topic?
    print(lsamodel.print_topic(1))

    # Grab matrix V from SVD, A = USV^t
    V = corpus2dense(lsamodel[sent_term_matrix], len(lsamodel.projection.s)).T / lsamodel.projection.s
    
    # Output sentence with the longest vector lengths, no repeats
    lengths = find_length(lsamodel.projection.s, V)
    indices = find_indices(lengths, 3) # number of sentences printed = 5
    indices.sort() # a summary makes more sense in-order

    hypothesis = "" # the collection of chosen sentences
    for i in range(0, len(indices)):
        hypothesis += sentences[indices[i]]

    print(hypothesis)
    reference = nlp.summary # provided wikipedia summary
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    print(scores)
    

    
        
    

    

