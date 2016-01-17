import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import metrics
#t=pd.read_csv("train.txt",sep="\t")
train=pd.read_csv("train.csv")
#print(train)
test=pd.read_csv("test.csv")
vectorizer = TfidfVectorizer(max_features=9000)
#vectorizer.get_feature_names()
train_data_features = vectorizer.fit_transform(train["Summary"])
test_data_features=vectorizer.fit_transform(test["Summary"])
#pd.DataFrame(data={"id", "topic"})
#pd.to_csv("my_csv.csv")
res= OneVsOneClassifier(LinearSVC(random_state=0))
res_fit=res.fit(train_data_features, train["Topic"]).predict(test_data_features)

result = pd.DataFrame( data={"id":test["ID"],"topic":res_fit} )
print(result)
#met=metrics.accuracy_score(test["ID"], res_fit)
#print("accuracy:   %0.3f" % met)
result.to_csv('output1.csv', sep="\t")
