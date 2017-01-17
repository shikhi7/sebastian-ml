import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
count=CountVectorizer(ngram_range=(1,1))
docs=np.array(['The sun is shining','The weather is sweet','The sun is shining and the weather is sweet'])
bag=count.fit_transform(docs)
#So here is the deal, we take all the unique words, give them all some special mapping
#and now to calculate feature vectors, we mark the occurences of the words
#Thus the vectors may be sparse.
# print(count.vocabulary_)
# print(bag.toarray())
#Mention of raw term frequencies. n-gram method. Taking 1 word or 2 words et al.
#Term frequency and inverse document frequency: Downweight frequently occuring words in feature vectors
# tf-idf=tf*idf
#idf=log((n_d)/(1+df(d,t)))
#We also have tf-idf library
tfidf= TfidfTransformer()
bag=tfidf.fit_transform(bag)
# print(bag.toarray())

#Cleaning the data, using regex library
#First remove all the html tags, then collect emoticons, remove any non word character from original data, convert to lower case, append the emoticons.
def preprocessor(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    #Removing non word characters
    text=re.sub('[\W+]',' ',text.lower()) + ' '.join(emoticons).replace('-','')
    return text
df=pd.read_csv('./movie_data.csv')
#test1
# print(df.head(n=5))
#print(preprocessor(df.loc[0,'review']))
#test2
# print(preprocessor("</a>This :) is :( a test =)!"))
#Applying preprocessor to all the reviews
df['review']=df['review'].apply(preprocessor)
# Apply method applies functions along the axis. Available exclusively in pandas.
#test
print(df.head(n=3))
#Exporting cleaned data as a csv
df.to_csv('./cleaned_movie_data.csv',index=False)
