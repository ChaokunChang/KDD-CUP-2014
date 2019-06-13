# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:42:05 2016

@author: attam
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:58:31 2016

@author: attam
"""

import csv
import math
from matplotlib import pyplot as plt
import pandas as pd
#import textblob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from collections import Counter
import re
from collections import Counter
from collections import defaultdict
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
import string
from nltk.corpus import stopwords
import re
#from stop_words import get_stop_words
#from nltk_data.corpora import stopwords
#s=set(stopwords.words('english'))
#s    
#function having nltk defined stopwords
def removestopwords(text):
    articles = {'i':'','me':'','my':'','myself':'','we':'','our':'','ours':'','ourselves':'','you':'','your':'','yours':'','yourself':'','yourselves':'','he':'','him':'','his':'','himself':'','she':'','her':'','hers':'','herself':'','it':'','its':'','itself':'','they':'','them':'','their':'','theirs':'','themselves':'','what':'','which':'','who':'','whom':'','this':'','that':'','these':'','those':'','am':'','is':'','are':'','was':'','were':'','be':'','been':'','being':'','have':'','has':'','had':'','having':'','do':'','does':'','did':'','doing':'','a':'','an':'','the':'','and':'','but':'','i':'','for':'','because':'','as':'','until':'','while':'','of':'','at':'','by':'','for':'','with':'','about':'','against':'','between':'','into':'','through':'','during':'','before':'','after':'','above':'','below':'','to':'','from':'','up':'','down':'','in':'','out':'','on':'','off':'','over':'','under':'','again':'','further':'','then':'','once':'','here':'','there':'','when':'','where':'','why':'','how':'','all':'','any':'','both':'','each':'','few':'','more':'','most':'','other':'','some':'','such':'','no':'','nor':'','not':'','only':'','own':'','same':'','so':'','than':'','too':'','very':'','st':'','can':'','will':'','just':'','don':'','should':'','now':'','d':'','ll':'','more':'','ve':'','y':'','ain':'','aren':'','couldn':'','didn':'','doesn':'','hadn':'','hasn':'','haven':'','isn':'','ma':'','mightn':'','mustn':'','needn':'','shan':'','shouldn':'','wasn':'','weren':'','won':'','wouldn':'','ii':'','need':'','students':''}
    type(articles)
    rest = []
    for word in text.split():
        if word not in articles:
            rest.append(word)
    return ' '.join(rest)
    
 
def clean_essay(string2):
    string2 = re.sub(r"\\t", " ", string2)   
    string2 = re.sub(r"\\n", " ", string2)   
    string2 = re.sub(r"\\r", " ", string2)   
    string2 = re.sub(r"[^A-Za-z0-9\']", " ", string2)
    string2 = re.sub(r"\d","",string2)
    string2 = re.sub(r"\s{2,}", " ", string2)
    #for i in en_stop:
     #   if i in string.split()not in en_stop:
      #      string = re.sub(i, " ", string)
    string2 = string2.lower()
    return string2.strip()
    string2.strip()
   
o = pd.read_csv(r'C:\Users\attam\Desktop\UMD\Courses_sem3\ENPM808L\Project\KDD_Cup_2014\outcomes.csv')
e = pd.read_csv(r'C:\Users\attam\Desktop\UMD\Courses_sem3\ENPM808L\Project\KDD_Cup_2014\essays.csv')
c = pd.merge(e, o, how='left', on='projectid')
c=c.dropna(axis=0)

#keeping the exciting projects only
f=c[(c['is_exciting'] == 't')]
words = Counter()
set(stopwords.words('english'))
en_stop=set(stopwords.words('english'))
en_stop.update(['students','need','need','school','reading','2','3','4','30','5','1','6','10','20','25','8','15','40','12','35','7','50','24','60','9','11','16','32','13','28','18','14','22','100','36','26'])

for i,row in f.iterrows():    
    row['need_statement']=clean_essay(row['need_statement'])
    words.update(row['need_statement'].split())
    row['need_statement']
    row
for i,row in f.iterrows():    
    row['short_description']=clean_essay(row['short_description'])
    words.update(row['short_description'].split())
for i,row in f.iterrows():    
    row['title']=clean_essay(row['title'])
    words.update(row['title'].split())
for i,row in f.iterrows():    
    row['essay']=clean_essay(row['essay'])
    words.update(row['essay'].split())
x=words.most_common()

with open('C:/Users/attam/Desktop/output_common_words.csv','w',newline='') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['name','num'])
    for row in x:
        if row[0] not in en_stop:
            #print (row[0])
            csv_out.writerow(row)
x   

