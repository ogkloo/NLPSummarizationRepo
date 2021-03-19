# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:46:06 2021

@author: David Froman
"""
import wikipedia
from rouge import Rouge
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
from heapq import nlargest
nltk.download('stopwords')
nltk.download('punkt')
from collections import defaultdict

#stuff for NLP
nlp = wikipedia.page("Natural language processing")

#stuff for rest
numArticles = 50
maxSummaryLength = 10
rest = []
numDocs = {}
textDocs = []
length = np.zeros(numArticles)
rougeScores = np.zeros(numArticles*10).reshape(numArticles,10)
freqList = []

#create the summaries holder 
summariesNLP = []
for i in range(numArticles):
    summariesNLP.append([[],[],[],[],[],[],[],[],[],[]])
summariesWiki = []
for i in range(numArticles):
    summariesWiki.append([[],[],[],[],[],[],[],[],[],[]])

#get the articles
pagesSucc = 0
while pagesSucc < numArticles:
    try:
        #get the random wikipedia pages
        p = wikipedia.random(1)
        rest.append(wikipedia.page(p))
        for sentances in range(maxSummaryLength):
            summariesWiki[pagesSucc][sentances] = wikipedia.summary(p,sentances)
        pagesSucc += 1
        print(pagesSucc)
    except:
        print("This is ridiculous")

#begin loop
for i in range(numArticles):
    text = ""
        
    # read the data from the url
    soup= BeautifulSoup(rest[i].html(), 'html.parser')
    # find all text that has p tag
    text_p = soup.find_all('p')

    #lowercase all text
    for j in range(0,len(text_p)):
        text += text_p[j].text
    text = text.lower()
    
    # tokenize the text, clean tokens
    tokens =[t for t in text.split()]    
    clean_token =tokens[:]
    
    #define irrelevant words that include stop words , punctuations and numbers
    stopword = set(stopwords.words('english') + list(punctuation) + list("0123456789") )
    for token in tokens:
        if token in stopword:
            clean_token.remove(token)
    
    #cpture the length of the page
    print(i)
    length[i] = len(clean_token)
    
    #append the word freq list to the end of freqList
    freqList.append(nltk.FreqDist(clean_token))
    textDocs.append(text)
    
#find the number of documents each word is in
for doc in range(numArticles):
    for word in freqList[doc]:
        if word in numDocs:
            numDocs[ word ] += 1
        else:
            numDocs[ word ] = 1
    

        
#loop over each document to get variable length summaries
for doc in range(numArticles):
    
    #do upkeep stuff
    top_words=[]
    top_words=freqList[doc].most_common(100)
    sentences = sent_tokenize(textDocs[doc])
    
    #loop over each summary length option
    for summLen in range(maxSummaryLength):
        #get and store the rouge-l score for each summary length option
        ranking = defaultdict(int)
        for i, sent in enumerate(sentences):
            wordCount = 1
            for word in word_tokenize(sent.lower()):
                if word in freqList[doc]:
                    ranking[i]+= (freqList[doc][word] * np.log(numArticles/numDocs[word]))
                    wordCount += 1
            ranking[i] = ranking[i]/wordCount
            top_sentences = nlargest(min(summLen+1, len(sentences)), ranking, ranking.get)
            sorted_sentences = [sentences[j] for j in sorted(top_sentences)]
            summariesNLP[doc][summLen] = sorted_sentences
            


rougeScores = np.zeros(numArticles*10).reshape(numArticles,10)
for i in range(numArticles):
    for j in range(maxSummaryLength):
        rouge = Rouge()
        temp1 = ""
        temp2 = ""
        for k in range(len(summariesNLP[i][j])):
            temp1 += summariesNLP[i][j][k]
        for k in range(len(summariesWiki[i][j])):
            temp2 += summariesWiki[i][j][k]
        rougeScores[i][j] = rouge.get_scores(temp1, temp2, avg=True)['rouge-l']['f']
        

#split articles into 4 groups, make list of indicies of what article is in what group
avgHolder = np.zeros(40).reshape(4,10)
ind0 = np.arange(0)
ind200 = np.arange(0)
ind500 = np.arange(0)
ind1000 = np.arange(0)

for i in range(numArticles):
    #return indicies of articles with words: <200
    if length[i] < 200:
        ind0 = np.append(ind0,i)
    #return indicies of articles with words: 200-500
    if (length[i] >= 200) and (length[i] < 500):
        ind200 = np.append(ind200,i)
    #return indicies of articles with words:500-1000
    if (length[i] >= 500) and (length[i] < 1000):
        ind500 = np.append(ind500,i)
    #return indicies of articles with words: >1000
    if length[i] > 1000:
        ind1000 = np.append(ind1000,i)

for art in range(len(ind0)):
    for sum in range(maxSummaryLength):
        avgHolder[0][sum] += rougeScores[ind0[art]][sum]/len(ind0)

for art in range(len(ind200)):
    for sum in range(maxSummaryLength):
        avgHolder[1][sum] += rougeScores[ind200[art]][sum]/len(ind200)
  
for art in range(len(ind500)):
    for sum in range(maxSummaryLength):
        avgHolder[2][sum] += rougeScores[ind500[art]][sum]/len(ind500)

for art in range(len(ind1000)):
    for sum in range(maxSummaryLength):
        avgHolder[3][sum] += rougeScores[ind1000[art]][sum]/len(ind1000)
        
        
#plot a grouping of articles
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x_axis = np.arange(10)+1
y_axis = avgHolder[3]
ax.bar(x_axis,y_axis)
plt.show()


#Comparison Summary of NLP Article
NLPTextHolder = []
NLPTextHolder.append(nlp.section("Methods: Rules, statistics, neural networks"))
NLPTextHolder.append(nlp.section("Statistical methods"))
NLPTextHolder.append(nlp.section("Neural networks"))
freqNLPList = []
NLPnumDocs = {}

#begin loop
for i in range(len(NLPTextHolder)):
    text = ""

    #lowercase all text
    for j in range(0,len(NLPTextHolder[i])):
        text += NLPTextHolder[i][j]
    NLPTextHolder[i] = text.lower()
    
    # tokenize the text, clean tokens
    tokens =[t for t in NLPTextHolder[i].split()]    
    clean_token =tokens[:]
    
    #define irrelevant words that include stop words , punctuations and numbers
    stopword = set(stopwords.words('english') + list(punctuation) + list("0123456789") )
    for token in tokens:
        if token in stopword:
            clean_token.remove(token)
    
    #append the word freq list to the end of freqList
    freqNLPList.append(nltk.FreqDist(clean_token))
#    textDocs.append(text)
    
#find the number of documents each word is in
for doc in range(len(NLPTextHolder)):
    for word in freqNLPList[doc]:
        if word in NLPnumDocs:
            NLPnumDocs[ word ] += 1
        else:
            NLPnumDocs[ word ] = 1

summaries = [[],[],[]]
#loop over each document to get variable length summaries
for doc in range(len(NLPTextHolder)):
    
    #do upkeep stuff
    top_words=[]
    top_words=freqNLPList[doc].most_common(100)
    sentences = sent_tokenize(NLPTextHolder[doc])
    
    #get and store the rouge-l score for each summary length option
    ranking = defaultdict(int)
    for i, sent in enumerate(sentences):
        wordCount = 1
        for word in word_tokenize(sent.lower()):
            if word in freqNLPList[doc]:
                ranking[i]+= (freqNLPList[doc][word] * np.log(3/NLPnumDocs[word]))
                wordCount += 1
        ranking[i] = ranking[i]/wordCount
        top_sentences = nlargest(3, ranking, ranking.get)
        sorted_sentences = [sentences[j] for j in sorted(top_sentences)]
        summaries[doc] = sorted_sentences
        
print(summaries[0])
print(summaries[1])
print(summaries[2])