# CS 445 Final Group Project

# generating text summarization through the selection of key sentences using word probabilities

import wikipedia
import string
import nltk
from collections import Counter
from rouge_score import rouge_scorer


# fucntion to get R-L and R-1 values for a wikipedia page
# returns those values and the length of the article in characters
def get_rouge(page):
    article = page.content.rsplit('Further reading', 1)[0]
    article = article.rsplit('History', 1)[0]
    article = article.rsplit('See also', 1)[0]

    # adopted from: https://stackabuse.com/text-summarization-with-nltk-in-python/
    word_frequencies = {}
    for word in nltk.word_tokenize(article):
        if word not in string.punctuation:
            if word.lower() not in word_frequencies.keys():
                word_frequencies[word.lower()] = 1
            else:
                word_frequencies[word.lower()] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_list = nltk.sent_tokenize(article)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    k = Counter(sentence_scores)

    high = k.most_common(5)

    t = ""
    for i in high:
        t += i[0] + "  "

    s = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return s.score(page.summary, t), len(article)


# adopted from this code: https://stackabuse.com/text-summarization-with-nltk-in-python/

# dictionary holding scores for the
scores = {}
# run it through 100 articles
# for i in range(100):
#     print(i)
#     while True:
#         try:
#             p = wikipedia.page(wikipedia.random())
#             break
#         except wikipedia.exceptions.DisambiguationError:
#             print('something wrong')
#         except wikipedia.exceptions.PageError:
#             print('something wrong')
#     scores[p.title] = get_rouge(p)

# # this was how I printed it to process it
# a = []
# for x in scores:
#     a += [str(x) + ":" + str(scores[x]) + "#"]
#
# print(a)




# function to get 3 highest priority sentences from text
# duplicate code from rouge function
def get_sentences(text):
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in string.punctuation:
            if word.lower() not in word_frequencies.keys():
                word_frequencies[word.lower()] = 1
            else:
                word_frequencies[word.lower()] += 1
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
    sentence_list = nltk.sent_tokenize(text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    k = Counter(sentence_scores)
    high = k.most_common(3)
    t = ""
    for i in high:
        t += i[0] + "  "
    return t


# now getting sentences for single comparison
p = wikipedia.page(wikipedia.page('Natural language processing'))
s = p.section('Methods: Rules, statistics, neural networks')
print(get_sentences)



