#coding:utf-8
'''
Created on 2014年8月26日

@author: ZS

go to zhaoshi@162.105.71.134:~/cikm$
run: python gen_submission.py  trialname

'''
import sys
import numpy as np
def gen_submission(labels,d='testdata.txt'):
    #labels[0] array like 0-1 label labels[1] array like  decision_functions
    lines=open(d,'rb').readlines()
    selflabels=['UNKNOWN','TEST','VIDEO', 'OTHER', 'GAME', 'NOVEL', 'TRAVEL', 'LOTTERY', 'ZIPCODE']
#     with open(sys.argv[1]+'decisionfunctionresult.txt','wb') as f2:
#         for i in xrange(len(lines)):
#             idf=[labels[1][j][i] for j in xrange(7)]
#             iline=' '.join(str(i)  for i in idf)
#             f2.write(iline+'\n')
    with open(sys.argv[1]+'result.txt','wb') as f:
        for i in xrange(len(lines)):
            iline=lines[i].strip()+'\t'
            labelidxs=[ int(j)  for j in xrange(len(labels[0])) if labels[0][j][i]==1]
            #labelidxs=testlabels[i,:].rows[0]
            
            #to make sure data has at least one label, in case that all clfs give negative results
            if not len(labelidxs):
                maxi=np.argmax([ j  for j in xrange(len(labels[0])) if labels[1][j][i]])
                labelidxs.append(maxi)
                
            iline=iline+'CLASS='+selflabels[labelidxs[0]+2]
            for ilabel in labelidxs[1:]:
                iline=iline+' | CLASS='+selflabels[ilabel+2]
            iline=iline+'\r\n'   
            f.write(iline)
def pre2label(preds):
    dfs=np.array(preds)
    dfs=dfs.T
    labels=np.array([[1 if j>0 else 0 for j in preds[i]] for i in xrange(7)]).T
    return labels,dfs
    
def test():
    if len(sys.argv)<2:
        sys.argv.append('jj')
    #labels='' ###read your labels
    preddirs=['pred'+str(i) for i in xrange(7)]
    labels=pre2label(open(ipred,'rb').readlines() for ipred in preddirs)
    gen_submission(labels)           
if __name__=='__main__':
    test()             
