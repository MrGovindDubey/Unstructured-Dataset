import pandas as pd
import numpy as np
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



fd = pd.read_csv("data.csv")


# sub_category  = ['abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv', 'abcv']




import matplotlib.pyplot as plt

# Count the number of troll and non-troll comments
sub_category = fd['sub_category'].value_counts()

# Plot the distribution
plt.figure(figsize=(6, 6))
plt.pie(sub_category, labels=category, autopct='%1.1f%%')
plt.title('Distribution of Troll vs. Non-Troll Comments')
plt.show()



corpus = []
print(len(fd['crime_description']))
for i in range(0, len(fd['crime_description'])):
    review = re.sub('[^a-zA-Z]', ' ', fd['crime_description'][i])
    review= review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords_hindi =  ["से", "के", "है", "का", "की", "को", "में", "की", "लिए", "पर", "और", 'kya', 'tum', 'hai', 'ki', 'ka', 'mein', 'se', 'ko', 'hain', 'kar', 'raha', 'rahi']
    all_stopwords.update(all_stopwords_hindi) 
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        
    review =' '.join(review)
    corpus.append(review)
    
    


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 15000)
X = cv.fit_transform(corpus).toarray()
y = fd.iloc[:, 0].values




X.shape
y.shape
print(X.shape)
# print(y[: 5])





# Import the necessary libraries
from wordcloud import WordCloud

# Create a WordCloud for troll comments
troll_corpus = " ".join([corpus[i] for i in range(len(corpus)) if y[i] == 1])

# Generate the WordCloud
wordcloud = WordCloud(width=800, height=400).generate(troll_corpus)

# Plot the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud for Troll Comments")
plt.show()




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)





import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

model = Sequential(
    [               
        tf.keras.Input(shape=(15000,)),
        Dense(units=25, activation="relu", name='layer1'),
        Dense(units=15, activation="relu", name='layer2'),
        Dense(units=57, activation="linear", name='layer3'),
    ], name = "my_model" 
)



model.compile(loss= SparseCategoricalCrossentropy(from_logits = True), metrics = ["accuracy"], optimizer= Adam())

y_train = y_train.reshape(-1)
history =  model.fit(X_train, y_train, epochs = 15)





y_pred = model(X_test)
print(y_pred[0])
y_label = np.argmax(y_pred, axis=1)
print(y_label[0])