#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Name: kmer2vec

Created on Friday July 15 2022

Author: Ruohan Ren
"""

# open the fasta file of promoter sequences (in dataset_pro)
f=open("promoter.fasta","r")

if f.mode == "r":
    contents = f.read()

text = list(contents)

name=[[]for i in range(300000)]
dna=[[]for i in range(300000)]
n=0
nn=0    
i=0

# process 
while text[i]!='!':
    if text[i]=='>':
        m=1
        nn=nn+1
    else:
        m=0
    if m==1:
        j=i+1
        
        while text[j]!='\n':
            name[nn].append(text[j])
            j=j+1
            
        i=j
        m=0
    else:
        j=i
        
        while text[j]!='>' and text[j]!='!':
            if text[j]=='\n':
                
                j=j+1
                
            else:
                dna[nn].append(text[j])
                j=j+1
        i=j-1
    i=i+1
    

for i in range(1,nn+1):
    dna[i].append('!')


kmer=3
  
dna2=[[]for i in range(300000)]

for i in range(1,nn+1):
    for j in range(0,len(dna[i])-kmer):
        for k in range(j,j+kmer):
            dna2[i].append(dna[i][k])
        dna2[i].append(' ')
   
sentence=[[]for i in range(300000)]

for i in range(1,nn+1):
    sentence[i]="".join(dna2[i])


f=open("5.txt","w")


for i in range(1,nn+1):
    f.write(sentence[i])


from gensim.models import word2vec
import numpy as np
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  
sentences = word2vec.Text8Corpus('5.txt')  
model = word2vec.Word2Vec(sentences, sg=1,  window=24,  min_count=1,  negative=5, sample=0.001, hs=0, workers=1,epochs=2,vector_size=100,seed=666)  
model.save('word2vec_test1.model') 


model = word2vec.Word2Vec.load('word2vec_test1.model')


x={}
for word in model.wv.key_to_index.keys ():  # check all the words
     print(word)
     x[word]=model.wv[word]
print(x)

np.save('word2vec.npy',x)






