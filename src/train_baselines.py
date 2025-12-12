# src/train_baselines.py
import argparse, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support

def main(dataset_path="data/dataset.csv", out_dir="models/baselines"):
    df = pd.read_csv(dataset_path)
    X = df["text"]; y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    # NB pipeline
    nb_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
                        ("clf", ComplementNB())])
    nb_pipe.fit(X_train, y_train)
    y_pred = nb_pipe.predict(X_test)
    print("NB report\n", classification_report(y_test,y_pred,digits=4))
    joblib.dump(nb_pipe, f"{out_dir}/nb.joblib")
    # SVM pipeline
    svm_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
                         ("clf", LinearSVC())])
    svm_pipe.fit(X_train, y_train)
    y_pred = svm_pipe.predict(X_test)
    print("SVM report\n", classification_report(y_test,y_pred,digits=4))
    joblib.dump(svm_pipe, f"{out_dir}/svm.joblib")

if __name__ == "__main__":
    main()
