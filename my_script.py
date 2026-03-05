import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# load the dataset
df = pd.read_csv("spam.csv", encoding='latin1')

# keep only the requied column
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

# display dataset infomation
print(df.info())

# ceck for missing values
print(df.isnull().sum())

# remove duplicates
df.drop_duplicates(inplace=True)
print(f"Dataset Shape after removing duplicates: {df.shape}")


def data_cleaning(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # remove non-alphabetic characters
    text = text.lower()  # convert to lowercase
    text = text.split()  # tokenize
    text = ' '.join(text)  # rejoin
    return text

# apply text clean
df['text'] = df['text'].apply(data_cleaning)

# features and labels
X = df['text']
y = df['label']

# convert text data into tf-idf features row
tf_idf = TfidfVectorizer()
X = tf_idf.fit_transform(X)
X = pd.DataFrame(X.toarray(), columns=tf_idf.get_feature_names_out())

# encode labels 
le = LabelEncoder()
y = le.fit_transform(y)

# plot distribution of labels w ,h ratio
plt.figure(figsize=(16,6))
plt.title('Distribution of Target Labels')
sns.countplot(data=df, x=y, color='#3c32a8')
plt.show()

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42, stratify=y)

# train and evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f'\n{model_name}:')
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
    print(f"Precision Score: {precision_score(y_test, y_pred)}")
    print(f"Recall Score: {recall_score(y_test, y_pred)}")
    print(f"F1-Score: {f1_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # confusion matrix
    plt.figure(figsize=(10,6))
    plt.title(f'{model_name} - Confusion Matrix')
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', linewidths=2, linecolor='pink')
    plt.show()

# evaluate naive bayes models
evaluate_model(GaussianNB(), X_train, y_train, X_test, y_test, "Gaussian Naive Bayes")
evaluate_model(BernoulliNB(), X_train, y_train, X_test, y_test, "Bernoulli Naive Bayes")
evaluate_model(MultinomialNB(), X_train, y_train, X_test, y_test, "Multinomial Naive Bayes")