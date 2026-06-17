import json
from preprocessing import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

# Load intents
with open("intent.json", "r") as f:
    data = json.load(f)

# Extract patterns and tags
patterns = []
tags = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])


#no we willl clean each pattern
cleaned_pattern = [" ".join(preprocess(pattern)) for pattern in patterns]


#so now we will convert patterns into number for ml to understand using tf*idf

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(cleaned_pattern)

##NOW WE WILL TRAIN THE MODEL USING LOGICTIC REG or SVM
#model = SVC(kernel="linear", probability=True)   #COMMENT OUT UNWANTED MODEL
model= LogisticRegression(max_iter=1000)
model.fit(X, tags)


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved!")