import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import nltk
nltk.download('punkt')

# load the dataset
df = pd.read_csv(r"D:\college PDF\spam.csv.csv", encoding='latin-1')

#Remove duplicates
df = df.drop_duplicates(keep='first')

# Prepare features and target
x = df['EmailText'].values
y = df['Label'].values
porter_stemmer = PorterStemmer()

#Text preprocessor function
def preprocessor(text):
    text = text.lower()
    text = re.sub("\W", " ", text)
    text = re.sub("\s+(in|the|all|for|and|on)\s+", " _connector_ ", text)
    words = re.split("\s+", text)
    stemmed_words = [porter_stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Tokenzer function
def tokenizer(text):
    text = re.sub("(\W)", r" \1 ", text)
    return re.split("\s+", text)

# Vectorize the text
vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2), min_df=0.006, preprocessor=preprocessor)
x = vectorizer.fit_transform(x)

# Resample data to handle class imbalance
ros = RandomOverSampler(random_state=42)
print('Original dataset shape:', Counter(y))
x, y = ros.fit_resample(x, y)
print('Modified dataset shape:', Counter(y))

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train Multinomial Naive Bayes model
clf = MultinomialNB()
clf.fit(x_train, y_train)

#Test and print Naive Bayes accuracy
y_pred_NB = clf.predict(x_test)
NB_Acc = clf.score(x_test, y_test)
print('Naive Bayes Accuracy score= {:.4f}'.format(NB_Acc))

# get prediction for a sample input
sample = input('Enter a message:')
data = vectorizer.transform([sample]).toarray()
print('Prediction:', clf.predict(data))

#Train Support Vector Classifier
model = SVC(C=1, kernel='linear')
model.fit(x_train, y_train)

# Test SVC model
accuracy = metrics.accuracy_score(y_test, model.predict(x_test))
accuracy_percentage = 100 * accuracy
print(f'SVC Accuracy Percentage: {accuracy_percentage:.2f}%')

#Hyperparameter tuning for SVC using GridSearchCV
params = {"C": [0.2, 0.5], "kernel": ['linear', 'sigmoid']}
cval = KFold(n_splits=2)
TunedModel = GridSearchCV(SVC(), params, cv=cval)
TunedModel.fit(x_train, y_train)

# Test tuned SVC model
accuracy = metrics.accuracy_score(y_test, TunedModel.predict(x_test))
accuracy_percentage = 100 * accuracy
print(f'Tuned Model Accuracy Percentage: {accuracy_percentage:.2f}%')

# Train and tune Multinomial Naive Bayes using GridSearchCV
nb_params = {'alpha': [0.1, 0.5, 1.0], 'fit_prior': [True, False]}
TunedModel1 = GridSearchCV(MultinomialNB(), nb_params, cv=cval)
TunedModel1.fit(x_train, y_train)

#Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, TunedModel.predict(x_test)), annot=True, fmt="g")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Print classification report
print(classification_report(y_test, TunedModel.predict(x_test)))

# Testing the model with sample emails
mail1 = [
    "Hey, you have won a car !!!!. Conrgratzz",
    "Dear applicant, Your CV has been received. Best regards",
    "You have received $1000000 to your account",
    "Join with our WhatsApp group",
    "Kindly check the previous email. Kind Regards"
]

for mail in mail1:
    is_spam = TunedModel.predict(vectorizer.transform([mail]).toarray())
    print(mail + " : " + str(is_spam))

