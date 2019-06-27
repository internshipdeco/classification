import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv("Train.csv")
print(df.GL_Code.unique())

X = []

for row in df.Item_Description:
    row = row.split(" - ")
    X.append(row[2])
corpus = []
for i in range (0, len(df)):
    review = re.sub('[^a-zA-Z]',' ', df['Clean_text'][i])
    review = review.lower()
    review = nltk.word_tokenize(review)
    #ps = PorterStemmer()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
df["Clean_text"] = corpus

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
#df["final_score"] = df[["score_1","score_2","score_3","score_4","score_5"]].mean(axis=1).astype(int)
y =df[""].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df["Product_Category"] = labelencoder.fit_transform(df["Product_Category"])

X = df.GL_Code.to_frame()
y = df.Product_Category.to_frame()

from sklearn.svm import LinearSVR
regr = LinearSVR()
regr.fit(X, y)


df_test = pd.read_csv("Test.csv")
X_test = []
for row in df_test.Item_Description:
    row = row.split(" - ")
    X_test.append(row[2])

df_test["Clean_text"] = X_test
corpus_test = []
for i in range (0, len(df_test)):
    review = re.sub('[^a-zA-Z]',' ', df_test['Clean_text'][i])
    review = review.lower()
    review = nltk.word_tokenize(review)
    #ps = PorterStemmer()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)
df_test["Clean_text"] = corpus_test

X_test = cv.fit_transform(corpus_test).toarray()
y_pred = regr.predict(X_test).astype(int)

dt = labelencoder.inverse_transform(y_pred)
dt.reshape(-1,1)
