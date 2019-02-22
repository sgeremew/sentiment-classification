#
# Samuel Geremew
# February 19th, 2019
# CS484-001
# Domeniconi
#


import re
import nltk
import numpy as np
from nltk.stem import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


#------------------------------------------------------------------------------
# This program will focuses on using K Nearest-Neighbor Algorithm to determine 
# the sentiment of movie reviews. First it will take in a user determined file 
# that contains any number of reviews that each end in an #EOF and new line. 
# Then the program will preprocess the text by removing stop words, stemming, 
# removing punctuation, and a number of other preprocess steps. This 
# preprocessing is an attempt to reduce the dimensionality of the reviews 
# in order to make the algorithm run more efficiently. After preprocessing 
# what is left will be converted from a sparse matrix into a dense matrix 
# in order to again improve efficiency and time. Then it will use the KNN 
# Algorithm to determine the sentiment of the reviews and it will score 
# the accuracy.
#------------------------------------------------------------------------------


print("\n\n\n----------------------------\tHW 1\t----------------------------|")
print("\nSamuel Geremew\nCS484\n\n")


# Examining the first few reviews
reviews_training_data = []
reviews_test_data = []
training_scores = []

#
# Creating two array's where each element is a one line review
#

train_data_file = open('train-data-20.dat','r')
test_data_file = open('test-data-20.dat','r')

# only two records
for i in range(0,2):
	reviews_training_data.append(train_data_file.readline().strip())
for i in range(0,2):
	reviews_test_data.append(test_data_file.readline().strip())

# for line in train_data_file:
# 	reviews_training_data.append(line.strip())
# for line in test_data_file:
# 	reviews_test_data.append(line.strip())

train_data_file.close()
test_data_file.close()

#
# close the files, all data are now stored in the arrays
#



for review in reviews_training_data:
	training_scores.append(review[:2])
print("Here are the true scores in an array\n")
print(training_scores)
print("\n")


#
#
#	Preprocessing
#
#	1.) Remove special characters and punctuations
#	2.) Stemming: reduce the words to their roots
#	3.) Remove stop words: words with little meaning when tokenized

# Creating a stemmed version of our document vectorizer
stemmer = PorterStemmer()
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

# Replace special characters that are not needed with a space or just remove 
# them
REMOVE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
SPACES = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\t)")

# function for character replacement and elimination
def preprocess_reviews(reviews):

    reviews = [REMOVE.sub("", line.lower()) for line in reviews]
    reviews = [SPACES.sub(" ", line) for line in reviews]
 
    return reviews


#	1.) Remove special characters and punctuations
#
reviews_training_data = preprocess_reviews(reviews_training_data)
reviews_test_data = preprocess_reviews(reviews_test_data)


#	2.) Stemming
#	3.) Remove stop words
#
#	Remove stop words and then we'll do some stemming
scvectorizer = StemmedCountVectorizer(stop_words = 'english')




#
# learns vocab of training set
#
scvectorizer.fit(reviews_training_data)


#
# transforms the document word vectors into sparse document matrices
#
train_sparse = scvectorizer.transform(reviews_training_data) 
test_sparse = scvectorizer.transform(reviews_test_data)

names = scvectorizer.get_feature_names()
print("Here are the vocabulary terms \"aka the dimensions\"\n")
print(names)
print("\nHere is the sparse matrix printed in an array format\n")
print(train_sparse.toarray())


#
# Transform both train and test data to array's
#
train_array = train_sparse.toarray()
test_array = test_sparse.toarray()











print("\n\tprogram ending...\n")

