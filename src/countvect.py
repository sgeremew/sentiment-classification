from sklearn.feature_extraction.text import CountVectorizer

# corpus = [
# 'This is the first document.',
# 'This document is the second document.',
# 'And this is the third one.',
# 'Is this the first document?']
array = []
file = open('train-data-20.dat','r')

# only two records
for i in range(0,2):
	array.append(file.readline().strip())

file.close()

vectorizer = CountVectorizer()

vectorizer.fit(array)

X = vectorizer.transform(array)

print(vectorizer.get_feature_names())
print(X.toarray())