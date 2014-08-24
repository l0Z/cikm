#coding:utf-8
'''
Created on 2014年8月24日

@author: ZS
四种表示，7个类别
'''
import numpy as np
import scipy.sparse as sp
from scipy import io
from itertools import islice
from collections import Counter
import cPickle as pickle
import scipy
import random
import time
import sys
class querylog():
    def __init__(self,):
        return 
    def gen_clickgraph(self,lines):
        #tri tuples
        self.queries=set([i[1] for i in lines])
        self.clicks=set([i[2] for i in lines])-set(['-',])
        self.queriesd=dict((i,idx) for idx,i in enumerate(self.queries))
        self.clicksd=dict((i,idx) for idx,i in enumerate(self.clicks))
        QN=len(self.queries)
        CN=len(self.clicks)
        self.clickgraph=sp.lil_matrix((QN,CN))
        for iline in lines:
            ilabel,iquery,iclick=iline
            qidx=self.queriesd.get(iquery,-1) 
            
            cidx=-1           
            if not iclick.strip()=='-':
                cidx=self.clicksd.get(iclick.strip(),-1)
            
            if qidx>=0 and cidx>=0:
                self.clickgraph[qidx,cidx]=self.clickgraph[qidx,cidx]+1
 
    def gen_textfeatures(self,lines):
        #bi tuples
        self.count=0
        N=len(lines)
        M=len(self.clicks)+len(self.ngramsd[0])+len(self.ngramsd[1])+len(self.ngramsd[2])+len(self.ngramsd[3])   
        self.trainv=sp.lil_matrix((N,M))
        self.target=sp.lil_matrix((N,len(self.labels)))
        
        for iline in lines:
            iquery,ilabel=iline
            
            ilabels=list(ilabel)
          
            words=iquery.split()
            words=[int(i) for i in words]
            ngrams_f=[]
            offset=0
            for j in xrange(1,5):
                ngrams=[ tuple(words[i:i+j]) for i in xrange(len(words)-j)]
                #print ngrams,len(self.ngramsd[j-1]),list(self.ngramsd[j-1][:10])
                ngramsidx=[self.ngramsd[j-1].get(igram,-1) for igram in ngrams]
                ngrams_f.extend([i+offset for i in ngramsidx if i>=0])
                offset=offset+len(self.ngramsd[j-1])           
            self.trainv[self.count,ngrams_f]=1
            
            #
            #             self.clickv[self.count,ngrams_f]=1
            
            for labelid in ilabels:
                self.target[self.count,labelid]=1
            self.count=self.count+1


def gen_dict(lines):
    ngrams=[Counter() for  i in xrange(4) ]
    for iline in lines:
        words=iline.split()            
        words=[int(i) for i in words]
        #print words                  
        for j in xrange(1,5):
            #print [ tuple(words[i:i+j]) for i in xrange(max(0,len(words)-j))]
            ngrams[j-1].update([ tuple(words[i:i+j]) for i in xrange(len(words)-j)])
        
    for j in xrange(0,4):
        num=len(ngrams[j])
        print num
        ngrams[j]=ngrams[j].most_common()
        print ngrams[j][num/3],ngrams[j][num/2],ngrams[j][num*2/3]
        ngrams[j]= set([i[0] for i in ngrams[j]])
    return ngrams

def test():
    #gen query dict
    lbqueries=pickle.load( open('lbqueries','rb'))
    qs=set([i[1] for i in lbqueries ])
    ngrams=gen_dict(qs)
    ngramsd=[{} for  i in xrange(4) ]
    for j in xrange(4):
        ngramsd[j]=dict( (igram,idx) for idx,igram in enumerate(ngrams[j]))
        print j,len(ngramsd[j])
    pickle.dump(ngrams, open('qngrams.pkl','wb'), protocol=2)
    pickle.dump(ngramsd, open('qngramsd.pkl','wb'), protocol=2  )  
    
#     cs=set([i[2] for i in lbqueries ])-set(['-',])
#     cngrams=gen_dict(cs)
#     for j in xrange(4):
#         ngrams[j]=ngrams[j]|cngrams[j]
#         ngramsd[j]=dict( (igram,idx) for idx,igram in enumerate(ngrams[j]))
#         print j,len(ngramsd[j])
#     pickle.dump(ngrams, open('qcngrams.pkl','wb'), protocol=2)
#     pickle.dump(ngramsd, open('qcngramsd.pkl','wb'), protocol=2  )  
    
    simp_log=querylog()
    simp_log.labels=['UNKNOWN', 'TEST', 'VIDEO', 'OTHER', 'GAME', 'NOVEL', 'TRAVEL', 'LOTTERY', 'ZIPCODE']
    simp_log.labelsd=dict((ilabel,idx) for idx,ilabel in enumerate(simp_log.labels))

    simp_log.ngramsd=ngramsd
    simp_log.gen_clickgraph(lbqueries)
    scipy.io.savemat('clickgraph.mat',{'clickgraph':simp_log.clickgraph})
    squeries=dict((i[1],set([])) for i in lbqueries )
    for i in lbqueries:
        squeries[i[1]].update(i[0])
    simp_log.gen_textfeatures(squeries.items())
    scipy.io.savemat('text.mat',{'trainv':simp_log.trainv,'target':simp_log.target})
    
if __name__=='__main__':
    test()           

        
