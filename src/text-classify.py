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



















print("\n\tprogram ending...\n")

