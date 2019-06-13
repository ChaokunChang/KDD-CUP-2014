# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:59:51 2016

@author: bbux
"""
import sys
import re
import gensim

if __name__ == "__main__":
    file = sys.argv[1]
    n = int(sys.argv[2])
    ldamodel = gensim.models.ldamodel.LdaModel.load(file)
    topics = ldamodel.print_topics(n, num_words=10)
    for tid, topic in topics:
        terms = sorted(re.findall(r'"(.*?)"', topic))
        print(str(tid) + "," + ",".join(terms))
