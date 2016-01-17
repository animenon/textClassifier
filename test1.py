import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
vectorizer = TfidfVectorizer(max_features=9000)
train_data_features = vectorizer.fit_transform(train["Summary"])
test_data_features=vectorizer.fit_transform(test["Summary"])
res= OneVsOneClassifier(LinearSVC(random_state=0))
res_fit=res.fit(train_data_features, train["Topic"]).predict(test_data_features)
result = pd.DataFrame( data={"id":test["ID"],"topic":res_fit} )
print(result)
result.to_csv('output.txt', sep="\t")
