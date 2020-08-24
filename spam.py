import os
import pandas as pd
import numpy as np


os.chdir("C:\\Users\\user\\Documents\\Python\\Practises\\S.C")

#message = pd.read_csv("Spam SMS Collection",names = ['label','message'])
#delimit = pd.read_csv("Spam SMS Collection",names = ['label','message'],delimiter = '\n')

sep = pd.read_csv("Spam SMS Collection",names = ['label','message'],sep = '\t')

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
corpus = []

for i in range(len(sep['message'])):
    review = re.sub('[^a-zA-Z]',' ',sep['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    comment = ' '.join(review)
    corpus.append(comment)


word = [nltk.word_tokenize(word) for word in corpus]    

from sklearn.feature_extraction.text import CountVectorizer

CV = CountVectorizer(max_features= 4500)
X = CV.fit_transform(corpus).toarray()
X = pd.DataFrame(X)

import pickle

obj = pickle.dump(CV,open('BOW.pkl','wb')) 

Y = sep['label']

FullRaw = pd.concat([X,Y],axis =1)
FullRaw['label'] = np.where(FullRaw['label'] == 'ham',1,0)

from sklearn.model_selection import train_test_split
    
Train,Test = train_test_split(FullRaw,test_size = 0.3, random_state =123)

Train_X = Train.drop(['label'],axis =1)
Train_Y = Train['label']
Test_X = Test.drop(['label'],axis =1)
Test_Y = Test['label']

from sklearn.naive_bayes import MultinomialNB

M1 = MultinomialNB().fit(Train_X,Train_Y)

Test_Pred = M1.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_Pred,Test_Y)

sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

model = pickle.dump(M1,open('model.pkl','wb'))  

  
