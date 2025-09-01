import re

import pandas as pd
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

DetectorFactory.seed = 0
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC

svc = LinearSVC(class_weight='balanced', max_iter=10000)
csv = pd.read_csv("multilingual_mobile_app_reviews_2025.csv")
vectorizer =  TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
clf = LogisticRegression(class_weight='balanced', max_iter=10000)
csv = csv.dropna(subset=['review_text'])
def clean_text(text, lang):
    text = text.lower()
    text = re.sub(r"[^a-zA-Záéíóúãõâêîôûç ]", " ", text)
    text = re.sub(r"\s+", " ",text).strip()
    lang_map = {'en': 'english', 'pt': 'portuguese', 'es': 'spanish'}
    if lang in lang_map:
        sw = set(stopwords.words(lang_map[lang]))
        tokens =  [w for w in text.split() if w not in sw]
        return " ".join(tokens)
    return text

def safe_detect(text):
    try:
        return detect(str(text))
    except:
        return "unknown"


csv['language'] = csv['review_text'].apply(safe_detect)
csv['clean_text'] = csv.apply(
    lambda row: clean_text(row['review_text'], row['language']), axis=1
)

csv = csv[csv['review_text'].str.strip().astype(bool)]
csv = csv[csv['review_text'].str.split().str.len() >=3]
csv = csv[csv['language'].isin(['en', 'pt', 'es'])]

csv['pos/neg'] = csv['rating'].apply(lambda x: 'pos' if x >=4 else 'neg')
csv_pos = csv[csv['pos/neg'] == 'pos']
csv_neg = csv[csv['pos/neg'] == 'neg']

csv_pos_sampled = pd.concat([csv_pos]*2, ignore_index=True)
csv_balanced = pd.concat([csv_pos_sampled, csv_neg], ignore_index=True)
csv_balanced = csv_balanced.sample(frac=1, random_state = 42).reset_index(drop=True)

X = csv_balanced['clean_text']
y = csv_balanced['pos/neg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

param_grid_logreg = {'C':[0.01, 1, 10],'penalty':['l2'], 'class_weight':['balanced',None]}
grid_logreg = GridSearchCV(LogisticRegression(max_iter=10000), param_grid_logreg, scoring='f1_macro', cv=3, n_jobs=1)
grid_logreg.fit(X_train_vect, y_train)
best_logreg = grid_logreg.best_estimator_
y_pred_logreg = best_logreg.predict(X_test_vect)
print('logisticRegression:', classification_report(y_test, y_pred_logreg))
print('melhor param', grid_logreg.best_params_)

paramgrid = {'C':[0.01, 1, 10], 'class_weight':['balanced',None]}
gridsvc = GridSearchCV(LinearSVC(class_weight='balanced', max_iter=10000), paramgrid, scoring='f1_macro', cv=3, n_jobs=1)
gridsvc.fit(X_train_vect, y_train)
y_pred_svc = gridsvc.predict(X_test_vect)
print('LinearSVC:', classification_report(y_test, y_pred_svc))
print('melhor param', gridsvc.best_params_)

clf.fit(X_train_vect, y_train)
svc.fit(X_train_vect, y_train)

y_pred = clf.predict(X_test_vect)
y_predSVC = svc.predict(X_test_vect)

print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_predSVC))














