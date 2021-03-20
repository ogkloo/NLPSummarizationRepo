# CS445
# Nick Murphy
# This code uses partial code from
# https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
# for the purposes of tokenizing data and utilizing gensim
# This code uses partial code from
# https://medium.datadriveninvestor.com/simple-text-summarizer-using-nlp-d8aaf5828e68
# for the purposes of utilizing beautiful soup to parse HTML
# The LSA Summarization selection algorithm is the exact same as detailed in
# Steinberger and Ježek
# This code generates mutliple ROUGE scores from random wikipedia pages
# and plots their averages

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
    text_length = 0
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
        while (not(i == len(S)) and S[i] > maximum/2):
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

# Used to sort scores
def find_text_length(sentences):
    text_length = 0
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            text_length += 1
    return text_length
    
    
if __name__ == '__main__':
    num_pages = 50
    # contains 50 lists, which are lists of rouge scores per page for a summary sentence of 1-5 sentences
    total_scores = [] 
    text_lengths = [] # text lengths

    pages = 0
    # for i in range(0, num_pages):
    while (pages < num_pages):
        pages += 1
        print(pages)
        # collect corpus
        try:
            page_title = wikipedia.random(pages=1)
            page_title = wikipedia.search(page_title, results=1)
            page = wikipedia.page(title=page_title)
        # wikipedia sometimes throws Disambiguation errors and warnings when being given a random title
        # in these cases, it was easier to simply try again with a new title rather than attempt to fix the issue
        except (wikipedia.exceptions.WikipediaException, Warning): 
            pages -= 1 
            continue
        
        soup = BeautifulSoup(page.html(), 'html.parser')
        text_p = soup.find_all('p')
        text = ""
        for i in range(0, len(text_p)):
            text += text_p[i].text
        text = text.lower()

        # tokenize text into sentences
        sentences = sent_tokenize(text)
        text_length = find_text_length(sentences)
        tokens = tokenize(text)

        # create dictionary of tokens
        dictionary = corpora.Dictionary(tokens)
        sent_term_matrix = [dictionary.doc2bow(sentence) for sentence in tokens]
        # Gensim's LsiModel performs Truncated SVD such that dimension of S = numtopics
        # if numtopics is not specified, performs SVD
        lsamodel = LsiModel(sent_term_matrix, id2word = dictionary)

        # Grab matrix V from SVD, A = USV^t
        V = corpus2dense(lsamodel[sent_term_matrix], len(lsamodel.projection.s)).T / lsamodel.projection.s

        # Output sentence with the longest vector lengths, no repeats
        lengths = find_length(lsamodel.projection.s, V)
        if (len(sentences) < 5):
            num_sentences = len(sentences)
        else:
            num_sentences = 5
        indices = find_indices(lengths, num_sentences) # number of sentences printed = 5
        
        scores = [] # rouge scores
        for j in range(1, num_sentences+1): # number of sentences printed out
            temp_indices = []
            for i in range(0, j):
                temp_indices.append(indices[i])
            temp_indices.sort()
            hypothesis = "" # our list of chosen sentences
            for i in range(0, j):
                hypothesis += sentences[temp_indices[i]]
            # print(hypothesis)
            
            reference = page.summary # provided wikipedia summary
            ref_sent = sent_tokenize(reference)
            reference = ""
            for i in range(0, len(ref_sent)):
                reference += ref_sent[i]
            rouge = Rouge()
            score_text = rouge.get_scores(hypothesis, reference)
            f = score_text[0]['rouge-l']['f']
            p = score_text[0]['rouge-l']['p']
            r = score_text[0]['rouge-l']['r']
            score = (f, p, r)
            scores.append(score)
        if len(scores) < 5:
            for i in range(len(scores), 5):
                scores.append((0, 0, 0))
        total_scores.append(scores)
        text_lengths.append(text_length)
        
    # sorting 
    bin200 = []
    bin500 = []
    bin1000 = []
    bin1001 = [] # bin >1000
    for i in range(0, num_pages):
        if text_lengths[i] < 200:
            bin200.append(total_scores[i])
        elif text_lengths[i] < 500:
            bin500.append(total_scores[i])
        elif text_lengths[i] < 1000:
            bin1000.append(total_scores[i])
        else:
            bin1001.append(total_scores[i])

    bins = [bin200, bin500, bin1000, bin1001]
    # plot every bin
    for bin in bins:
        f_means = []
        p_means = []
        r_means = []
        for i in range(0, 5):
            f = 0
            p = 0
            r = 0
            for j in range(0, len(bin)):
                f += bin[j][i][0]
                p += bin[j][i][1]
                r += bin[j][i][2]
            if not(len(bin) == 0):
                f_avg = f/len(bin)
                p_avg = p/len(bin)
                r_avg = r/len(bin)
                f_means.append(f_avg)
                p_means.append(p_avg)
                r_means.append(r_avg)
            else:
                f_means.append(0)
                p_means.append(0)
                r_means.append(0)
        
        labels = ['1', '2', '3', '4', '5']
        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, f_means, width, label='f')
        rects2 = ax.bar(x, p_means, width, label='p')
        rects3 = ax.bar(x + width, r_means, width, label='r')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by metric and # of summary sentences')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.show()
    
    

    
        
    

    

