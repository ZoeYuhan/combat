import numpy as np
import pandas as pd
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

train=pd.read_csv("/User/labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)   #Load labeled train data
test=pd.read_csv("/User/testData.tsv",header=0,delimiter="\t",quoting=3)            #Load test data
y=train["sentiment"]          #Train data result
  
traindata=[]
for i in xrange(0,len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_word(train["review"][i],False)))   #Pick up review data from traindata and conver to word
testdata=[]
for i in xrange(0,len(test["review"])):
    testdata.append(" ".join(KaggelWord2VecUtility.review_to_word(test["review][i],False)))      #Pick up review data from testdata and covert to word
    
tfv=TfidfVectorizer(min_df=3,max_feature=None,strip_accents="unicode",analyzer="word",token_pattern=r"\w{1,}",ngram_range=(1,2),use_idf=1,smooth_idf=1,sublinear_tf=1,stop_words="english")

X_all=traindata+testdata

tfv.fit(X_all)
X_all=tfv.transform(X_all)
lentrain=len(traindata)

X=X_all[:lentrain]
X_test=[lentrain:]

model=LogisticRegression(penalty="L2",dual=True,tol=0.001,C=1,fit_intercept=True,intercept_scaling=1.0,class_wight=None,random_state=None)
model.fit(X,y)
print "20 Fold CV Score:",np.mean(cross_validation.cross_val_score(model,X,y,cv=20,scoring="roc_auc"))

result=model.predict_proba(X_test)[:,1]
output=pd.DataFrame(data={"id":test["id"],"semtiment":result})

output=pd.to_csv("/User/bag_of_word_model.csv",index=False,quoting=3)
