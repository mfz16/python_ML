from sklearn.feature_extraction.text import CountVectorizer
import os.path
vectorizer = CountVectorizer(min_df=1)

content = ["How to format my hard disk", " Hard disk format problems "]

X = vectorizer.fit_transform(content)
print(vectorizer.get_feature_names())

print(X.toarray().transpose())
posts = [open(os.path.join("d:/datasets/Posts/", f)).read() for f in os.listdir("d:/datasets/Posts/")]
print (posts)

X_train = vectorizer.fit_transform(posts)
print (X_train)
num_samples, num_features = X_train.shape
print("#samples: %d, #features: %d" % (num_samples,num_features))