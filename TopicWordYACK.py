# Catriona McKay
# Topic Words with YAKE model
import wikipedia as wiki
import re
from pke.unsupervised import YAKE
from nltk.corpus import stopwords
import nltk.data
from nltk.tokenize import RegexpTokenizer
import numpy as np
import urllib.request
from bs4 import BeautifulSoup
from rouge import Rouge
import pandas as pd
import matplotlib.pyplot as plot


# function to run text summarization on article
def make_summarization(article, page, rouge, sum_length):
    # preprocessing of the text
    article = re.sub(r'[[0-9]*]', '', article)
    article = re.sub(r'[[citation needed]]', '', article)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(article)

    # getting the keywords using Yake
    extractor = YAKE()
    extractor.load_document(input=article,
                            language='en',
                            normalization=None)
    stoplist = stopwords.words('english')
    extractor.candidate_selection(n=2, stoplist=stoplist)
    extractor.candidate_weighting(window=2,
                                  stoplist=stoplist,
                                  use_stems=False)
    key_phrases = extractor.get_n_best(n=10, threshold=0.8)

    # subtract the key phrases from 1 so that the important words have a higher score
    # this should make it easier to rank the sentences
    key_words = []
    rankings = []
    for key in key_phrases:
        key_words.append(key[0])
        rankings.append(1 - key[1])

    # rank the sentences
    # this is done by adding the score of the key words in then diving by the number of words in the sentence
    # get the ranks for each sentence
    sentence_ranks = key_word_density(sentences, key_words, rankings)
    top_ranks = np.argsort(sentence_ranks)[-sum_length:]
    top_ranks = np.flip(top_ranks)
    # add the top ranked sentences to each other for the summary
    full_summary = ''
    for j in range(len(top_ranks)):
        full_summary += sentences[top_ranks[j]] + ' '
    full_summary = re.sub(r'\n', '', full_summary)
    # getting the word count so that the summary can be properly categorized.
    tokenizer = RegexpTokenizer(r'\w+')
    article_words = tokenizer.tokenize(article)
    article_length = len(article_words)
    scores = rouge.get_scores(full_summary, page.summary)
    return scores, article_length, full_summary


# function that gets the density rank of a sentence for keywords
def key_word_density(text, keys, ranks):
    sentence_rank = np.zeros(len(text))
    # go through each sentence and get the rank
    for k in range(len(text)):
        temp_sent = text[k].lower()
        # check for each keyword and add to the score if the sentence contains the keyword
        for j in range(len(keys)):
            temp_word = keys[j].lower()
            count = temp_sent.count(temp_word)
            sentence_rank[k] += count * ranks[j]
        sent_words = temp_sent.split(' ')
        sentence_rank[k] = sentence_rank[k]/len(sent_words)
    return sentence_rank


# function to put rouge scores in correct dataframe spot
def assign_length_graph(score, length, score_total, count):
    if length < 200:
        score_total[0][0] += score[0]['rouge-l']['f']
        score_total[0][1] += score[0]['rouge-l']['p']
        score_total[0][2] += score[0]['rouge-l']['r']
        count[0][0] += 1
    elif length < 500:
        score_total[1][0] += score[0]['rouge-l']['f']
        score_total[1][1] += score[0]['rouge-l']['p']
        score_total[1][2] += score[0]['rouge-l']['r']
        count[0][1] += 1
    elif length < 1000:
        score_total[2][0] += score[0]['rouge-l']['f']
        score_total[2][1] += score[0]['rouge-l']['p']
        score_total[2][2] += score[0]['rouge-l']['r']
        count[0][2] += 1
    elif length >= 1000:
        score_total[3][0] += score[0]['rouge-l']['f']
        score_total[3][1] += score[0]['rouge-l']['p']
        score_total[3][2] += score[0]['rouge-l']['r']
        count[0][3] += 1

    return score_total, count


# function that divides the sum of rouge scores by the number of articles for that section
def find_average_length(score_total, count):
    # go through each rouge score for each article length and divide by the number of articles for that range.
    for j in range(len(count)):
        if count[0][j] != 0:
            for k in range(score_total.shape[0]):
                score_total[k] = score_total[k] / count[0][j]
    return score_total


# get the summary and rouge score for the Natural language processing wikipedia page
wiki_page = wiki.page('Natural language processing', auto_suggest=False)
rouge = Rouge()
# for the comparison part of the we looked at the methodologies section of NLP
NLPtext = ''
NLPtext += wiki_page.section("Methods: Rules, statistics, neural networks")
NLPtext += wiki_page.section("Statistical methods")
NLPtext += wiki_page.section("Neural networks")
rouge_scores, length, summary = make_summarization(NLPtext, wiki_page, rouge, 3)
print('Natural language processing summary:\n', summary)
print('ROUGE scores: ', rouge_scores)

# make the variables needed to get the average rouge score for each range.
article_length_data = np.zeros(shape=(4, 3))
article_length_data_index = ['range 1', 'range 2', 'range 3', 'range 4']
article_count = np.zeros(shape=(1, 4))

# assign the value for how many random pages will be used.
page_count = 50
# get the random pages
wikipedia_pages = wiki.random(page_count)
# for each page run the summary creation algorithm and add collect the rouge scores.
for i in range(len(wikipedia_pages)):
    page_title = wikipedia_pages[i]
    # there is an exception if there are multiple articles that can be applied to the random article title provided
    # by the wiki API
    try:
        # get the page content
        wiki_page = wiki.page(page_title, auto_suggest=False)
        # preprocessing of the text from the wiki API
        request = urllib.request.urlopen(wiki_page.url).read().decode('utf8', 'ignore')
        # read the data from the url
        soup = BeautifulSoup(request, features='html.parser')
        # find all text that has p tag
        text_p = soup.find_all('p')
        full_text = ''
        for i in range(0, len(text_p)):
            full_text += text_p[i].text
        # run the summary creation algortihm
        rouge_scores, length, summary = make_summarization(full_text, wiki_page, rouge, 5)
        # collect the data
        article_length_data, article_count = assign_length_graph(rouge_scores, length, article_length_data,
                                                                 article_count)
    except wiki.exceptions.DisambiguationError as e:
        # skip the article if there were multiple options.
        print('Could not get wiki page for ', e.title)

# calculate the average for each rouge in each range
article_length_data = find_average_length(article_length_data, article_count)

# create and display the graph.
article_length_graph = pd.DataFrame(data=article_length_data, index=article_length_data_index)
article_length_graph.columns = ['f', 'p', 'r']
article_length_graph.plot.bar()
plot.title('Article Length vs ROUGE Scores')
plot.xlabel('Article Length')
plot.ylabel('ROUGE score')
plot.show()

# print the article count for each range and the averge rouge scores.
print(article_count)
print(article_length_data)