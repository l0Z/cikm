#coding:utf-8
'''
Created on 2014年7月21日

@author: ZS
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
from sklearn import metrics,linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score,mean_squared_error
def classify(traindata,testdata,traintarget,testtarget,alpha=0.1):

    #loss: log, 
    clf=linear_model.LogisticRegression(penalty='l1',class_weight='auto')
    #clf=linear_model.SGDClassifier(loss='hinge', penalty="l1",alpha=alpha)
    #clf=RandomForestClassifier(n_estimators=10)  
    clf.fit(traindata,traintarget)
    print 'train done! '
    #print 'score',clf.score(testdata,testtarget)
    #print 'sparsity',sp.csr_matrix(clf.coef_).nnz
    return clf
    
class querylogs():
    def __init__(self,):
#         self.wordD={}
        self.words=set([])#Counter()
        self.ngrams=[Counter() for i in xrange(4)]
        self.ngramsd=[{} for  i in xrange(4) ]
        self.queries=Counter()
        self.queriesd={}
        self.clicks=Counter()
        self.clickgraph=[]
        self.sessions=[]
        self.labels=Counter()
        self.sessionNum=0

    def gen_testfeature(self,d):
        lines=open(d,'rb').readlines()
        for iline in lines:
            words=iline.split()            
            words=[int(i) for i in words]
            #print words                  
            for j in xrange(1,5):
                #print [ tuple(words[i:i+j]) for i in xrange(max(0,len(words)-j))]
                self.ngrams[j-1].update([ tuple(words[i:i+j]) for i in xrange(len(words)-j)])
            self.words.update(words)
        for j in xrange(0,4):
#             num=len(self.ngrams[j])/2
#             print num
#             self.ngrams[j]=self.ngrams[j].most_common()
#             print self.ngrams[j][num/3]
            self.ngrams[j]= self.ngrams[j].keys()
            self.ngramsd[j]=dict( (igram,idx) for idx,igram in enumerate(self.ngrams[j]))
        pickle.dump(self.ngrams, open('ngrams.pkl','wb'), protocol=2)
        pickle.dump(self.ngramsd, open('ngramsd.pkl','wb'), protocol=2)
        
    def gen_queryfeature(self,iquery,shape):
        #words  0-1 feauture
        words=iquery.split()
        words=[int(i) for i in words]
        ngrams_f=[]
        offset=0
        for j in xrange(1,5):
            ngrams=[ tuple(words[i:i+j]) for i in xrange(len(words)-j)]
            #print ngrams,len(self.ngramsd[j-1]),list(self.ngramsd[j-1][:10])
            ngramsidx=[self.ngramsd[j-1].get(igram,-1) for igram in ngrams]
            ngrams_f.extend([i+offset for i in ngramsidx if i>=0])
            offset=offset+len(self.ngrams[j-1])
        
        col=ngrams_f 
        #print len(col)
        data=np.ones(len(col))     
        #click distribution   
#         queryid=self.queriesd.get(iquery,-1)
#         click_distri=self.clickgraph[queryid,:]
#         
#         col= np.hstack((col,click_distri.indices))
#         data=np.hstack((data,click_distri.data))
        row=np.zeros(len(col))
        return sp.coo_matrix((data, (row,col)),shape=shape)
        # sp.vstack([a,b,c..])
    
    def gen_trainv(self,d,N):
        #load feature dictionary
        self.ngramsd=pickle.load( open('ngramsd.pkl','rb'))
        #collect trainv,target
        gen_queryfeature(self,iquery,shape)
        
        # ignore unknown and test labelled data
        
    def collect_traindata(self,d,tN):
#         self.wordsd=dict((i,idx) for idx,i in enumerate(self.words))
#         self.clicksd=dict((i,idx) for idx,i in enumerate(self.clicks))
#         self.queriesd=dict((i,idx) for idx,i in enumerate(self.queries))
        self.labelsd=dict((ilabel,idx) for idx,ilabel in enumerate(self.labels))
        print self.labelsd
        QN=len(self.queries)
        CN=len(self.clicks)
        self.clickgraph=sp.lil_matrix((QN,CN))
        N=52810489#self.datanum#sum(self.queries.values()) # num of data point
        M=len(self.clicks)+len(self.ngrams[0])+len(self.ngrams[1])+len(self.ngrams[2])+len(self.ngrams[3])
        self.trainv=sp.lil_matrix((N,M))
        self.trainvshape=(N,M)
        self.target=sp.lil_matrix((N,len(self.labels)))
        self.sessionids=np.zeros(N)
        self.count=0
        with open(d,'rb') as f:
            while True:
                next_n_lines = list(islice(f, tN))
                if not next_n_lines:
                    break
                self.unideal2(next_n_lines)
                print self.trainv.nnz,self.target.nnz#,self.clickgraph.nnz
                #print 'a',self.sessionNum
        
        # label the ones within one session the same label
#     def train(self,):
#         #divide train and sample test
#     def train_classify(self,):
#         #gen test, train set 
    
    def sample(self,d,N):
        f1=open('sample.txt','wb')
        with open(d,'rb') as f:
            while True:
                next_n_lines = list(islice(f, N))
                if not next_n_lines:
                    break
                isession=[]
                for iline in next_n_lines:
                    if len(iline)<2:
                        #new session
                        p=random.randint(1,80)
                        if p==66:
                            #lucky number! retain this session!
                            f1.write(''.join(isession))
                            f1.write('\n')
                        isession=[]                        
                    else:
                        isession.append(iline)    
        f1.close()
    def gen_dict(self,d,N):
        with open(d,'rb') as f:
            while True:
                next_n_lines = list(islice(f, N))
                if not next_n_lines:
                    break
                self.unideal(next_n_lines)
                #print 'N'
        print self.labels
        print sum(self.clicks.values()),sum(self.clicks.values())/len(self.clicks)
        print len(self.words),len(self.clicks),len(self.queries)
        print sum(self.queries.values()),sum(self.queries.values())/len(self.queries)
        
        self.datanum=sum(self.queries.values())
        self.labelsd=dict((ilabel,idx) for idx,ilabel in enumerate(self.labels))
        self.words=set(self.words.keys())
        self.clicks=set(self.clicks.keys())
        self.queries=set(self.queries.keys())
#         for j in xrange(0,3):
#             #self.ngrams[j].update([ tuple(words[i:i+j]) for i in xrange(len(words)-j)])
#             self.ngrams[j]= self.ngrams[j].most_common(len(self.ngrams)/3)
#             self.ngrams[j]=set([i[0] for i in self.ngrams[j]])
#         QN=len(self.queries)
#         CN=len(self.clicks)
#         self.clickgraph=sp.lil_matrix((QN,CN))
#         N=sum(self.queries.values()) # num of data point
#         M=len(self.clicks)+len(self.words)+len(self.ngrams[0])+len(self.ngrams[1])+len(self.ngrams[2])
#         self.trainv=sp.csr_matrix((N,M))
#         self.target=sp.lil_matrix((N,len(self.labels)))
#         self.sessionids=np.zeros(N)
#         self.count=0
  
        
    def unideal(self,lines):
        for iline in lines:
            if len(iline)<2:
#                 self.sessionNum=self.sessionNum+1
                continue
            #print iline.split('\t')
            words=[]
            ilabel,iquery,iclick=iline.split('\t')
            ilabels=ilabel.split(' | ')
            for i in ilabels:
                self.labels.update([i[6:],])
            iwords=iquery.split()
            self.queries.update([iquery,])
            words.extend(iwords)
            if not iclick.strip()=='-':
                self.clicks.update([iclick.strip(),])
                words.extend(iclick.strip().split())
            
            words=[int(i) for i in words if i!='-']
                         
#             for j in xrange(0,3):
#                 self.ngrams[j].update([ tuple(words[i:i+j]) for i in xrange(len(words)-j)])
            self.words.update(words)
            #print self.labels,self.queries,self.clicks,self.words   
    def unideal2(self,lines):
        for iline in lines:
            if len(iline)<2:
                self.sessionNum=self.sessionNum+1
                continue
            #print iline.split('\t')
            self.sessionids[self.count]=self.sessionNum
#             words=[]
            ilabel,iquery,iclick=iline.split('\t')
            qidx=self.queriesd.get(iquery,-1) 
            
            cidx=-1           
            ilabels=ilabel.split(' | ')
#             words.extend(iquery.split())
            if not iclick.strip()=='-':
#                 words.extend(iclick.strip().split())
                cidx=self.clicksd.get(iclick.strip(),-1)
            
            if qidx>=0 and cidx>=0:
                self.clickgraph[qidx,cidx]=self.clickgraph[qidx,cidx]+1
            
#             words=[int(i) for i in words]
#             wordset=set(words)
#             #01 feature
            words=iquery.split()
            words=[int(i) for i in words]
            ngrams_f=[]
            offset=0
            for j in xrange(1,5):
                ngrams=[ tuple(words[i:i+j]) for i in xrange(len(words)-j)]
                #print ngrams,len(self.ngramsd[j-1]),list(self.ngramsd[j-1][:10])
                ngramsidx=[self.ngramsd[j-1].get(igram,-1) for igram in ngrams]
                ngrams_f.extend([i+offset for i in ngramsidx if i>=0])
                offset=offset+len(self.ngrams[j-1])
            
    
            self.trainv[self.count,ngrams_f]=1
            
            for i in ilabels:
                labelid=self.labelsd[i[6:]]
                if labelid<2:
                    continue
                self.target[self.count,labelid]=1
                
#             if self.trainv.shape[0]==1:
#                 self.trainv=self.gen_queryfeature(iquery, self.trainvshape)
#             else:
#                 self.trainv=sp.vstack([self.trainv,self.gen_queryfeature(iquery, self.trainvshape)])
#             
            #print labelid,self.trainv.nnz
            #print qidx,cidx
            self.count=self.count+1
       
    def uni2vector(self,lines):
        for iline in lines:
            if len(iline)<2:
                self.sessionNum=self.sessionNum+1
                continue
            #print iline.split('\t')
            ilabel,iquery,iclick=iline.split('\t')
            ilabels=ilabel.split(' | ')
            for i in ilabels:
                self.labels.add(i[6:])
            self.queries.update([iquery,])
            if not iclick=='-':
                self.clicks.update([iclick.strip(),])
                self.words.update(iclick.strip().split())
            self.words.update(iquery.split())
            #print self.labels,self.queries,self.clicks,self.words             
    def relabel(self):
        '''
        relabel the unknown queries to enrich training set
        based on click graph features..maybe
        '''
    def test_classify(self,p=0.8):
        '''
        '''
        labeled_rows=list(set(sp.coo_matrix(self.target[:,]).row)) #retain nonzero rows
        print len(labeled_rows)
        trainv=sp.csr_matrix(self.trainv)[labeled_rows,:]
        target=sp.csr_matrix(self.target[:,2:])[labeled_rows,:]
        print trainv.shape,target.shape
        num,d=trainv.shape
        self.clfs=[]
        for i in xrange(7):
            traindata=trainv[:int(num*p),:]
            traintarget=target[:int(num*p),i]
            print traintarget.nnz
            traintarget=traintarget.todense()
            testdata=trainv[int(num*p):,:]
            testtarget=target[int(num*p):,i].todense()
            #print traindata.shape,traintarget.shape,testdata.shape,testtarget.shape
            self.clfs.append(classify(traindata,testdata,traintarget,testtarget,alpha=0.1))
            
        return self.clfs
    def gen_submission(self,d):
#generate testdata vectors
#         lines=open(d,'rb').readlines()
#         self.testv=sp.lil_matrix((len(lines),self.trainv.shape[1]))
#         self.count=0
#         for iline in lines:
#             words=iline.split()            
#             words=[int(i) for i in words]
#             offset=0
#             ngrams_f=[]
#             for j in xrange(1,5):
#                 ngrams=[ tuple(words[i:i+j]) for i in xrange(len(words)-j)]
#                 #print ngrams,len(self.ngramsd[j-1]),list(self.ngramsd[j-1][:10])
#                 ngramsidx=[self.ngramsd[j-1].get(igram,-1) for igram in ngrams]
#                 ngrams_f.extend([i+offset for i in ngramsidx if i>=0])
#                 offset=offset+len(self.ngrams[j-1])
#             
#             self.testv[self.count,ngrams_f]=1
#             self.count=self.count+1
        
        testlabels=sp.lil_matrix((self.testv.shape[0],7))
        for i in xrange(7):
            ilabel=self.clfs[i].predict(self.testv)
            ilabel.shape=self.testv.shape[0],1
            testlabels[:,i]=ilabel#predict_proba(self.testv)
        lines=open(d,'rb').readlines()
        self.labels=set(['VIDEO', 'OTHER', 'GAME', 'NOVEL', 'TRAVEL', 'LOTTERY', 'ZIPCODE'])
        with open('result.txt','wb') as f:
            for i in xrange(len(lines)):
                iline=lines[i]+'\t'
                labelidxs=testlabels[i,:].rows[0]
                iline=iline+'CLASS='+self.labels[labelidxs[0]]
                for ilabel in labelidxs[1:]:
                    iline=iline+' | CLASS='++self.labels[ilabel]
                iline=iline+'\r\n'   
                f.write(iline)
              
        
def test_classify_all():
    baidulogs=querylogs()
    baidulogs.testv=scipy.io.loadmat('testv.mat')['testv']
    baidulogs.ngramsd=pickle.load( open('ngramsd.pkl','rb'))
    baidulogs.target=scipy.io.loadmat('baidum.mat')['target']
    baidulogs.trainv=scipy.io.loadmat('quickbaidum.mat')['trainv']
    
    baidulogs.test_classify(p=1)# use p percentage data to train
    
    baidulogs.gen_submission('testdata.txt')
def test1():
    d='sample.txt'
    baidulogs=querylogs()
    #baidulogs.sample(d,1000000)
    baidulogs.gen_dict(d, 1000000) 
def testdata():
    d='testdata.txt'
    baidulogs=querylogs()
    baidulogs.gen_testfeature(d)
   
def gen_clickgraph():
    baidulogs.clicks=pickle.load(open('bigbaidulogsclicks','rb'))
    baidulogs.queries=pickle.load(open('bigbaidulogsqueries','rb'))  
    
def test():
    d='train.txt'
    baidulogs=querylogs()
#     baidulogs.gen_dict(d, 1000000)  
#     #wordsdict,
#     pickle.dump(baidulogs.words,open('baidulogswords','wb'),protocol=2)
#     #clickdict
#     pickle.dump(baidulogs.clicks,open('baidulogsclicks','wb'),protocol=2)   
#     pickle.dump(baidulogs.queries, open('baidulogsqueries','wb'),protocol=2)
#     print 'pickle down'
    #baidulogs.words=pickle.load(open('baidulogswords','rb'))
#     baidulogs.clicks=pickle.load(open('baidulogsclicks','rb'))
#     baidulogs.queries=pickle.load(open('baidulogsqueries','rb'))
    baidulogs.labels=['UNKNOWN', 'TEST', 'VIDEO', 'OTHER', 'GAME', 'NOVEL', 'TRAVEL', 'LOTTERY', 'ZIPCODE']
    baidulogs.ngrams=pickle.load( open('ngrams.pkl','rb'))
    baidulogs.ngramsd=pickle.load( open('ngramsd.pkl','rb'))
    #return
#     baidulogs=pickle.load(open('baidulogs','rb'))
    print 'load!'
#     m=scipy.io.loadmat('baidum.mat')
#     baidulogs.trainv=m['trainv']
#     baidulogs.target=m['target']
#     baidulogs.sessionids=m['session']
#     baidulogs.test_classify()
    baidulogs.collect_traindata(d, 100000)
    scipy.io.savemat('quickbaidum.mat',{'trainv':baidulogs.trainv,'target':baidulogs.target,'session':baidulogs.sessionids,'graph':baidulogs.clickgraph})
#     #np.save('target',baidulogs.target)
#     #pickle.dump(baidulogs, open('baidulogs','wb'))
#     
#     print 'pickle down 2'
#     print sum(baidulogs.clicks.values())/len(baidulogs.clicks)
#     print len(baidulogs.words),len(baidulogs.clicks),baidulogs.labels
#     for i in xrange(4):
#         print baidulogs.ngrams[i].most_common()[:20],np.median(baidulogs.ngrams[i].values())    

def testbaidu():
    baidulogs=pickle.load(open('baidulogsclicks','rb'))
    print baidulogs.clicks.most_common()[:20],np.median(baidulogs.clicks.values())
    print len(baidulogs.queries),baidulogs.queries.most_common()[:20],np.median(baidulogs.queries.values())    
    print len(baidulogs.words),baidulogs.words.most_common()[:20],np.median(baidulogs.words.values())    
# def fobos():
#     sol_half = self.sol - self.gamma * self.f2.grad(self.sol)
#     self.sol= cmp(sol_half,0)*[ ]

if __name__=='__main__':
    import cProfile as profile
    #testdata()
    test()
    #profile.run('test_classify_all()','1.txt')   
    
    

    
