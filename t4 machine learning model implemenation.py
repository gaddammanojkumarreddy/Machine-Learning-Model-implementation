import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# Load dataset (using a sample dataset from UCI)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Text preprocessing
def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([c for c in msg if c not in string.punctuation])
    msg = ' '.join([word for word in msg.split() if word not in stopwords.words('english')])
    return msg

# Apply cleaning
df['clean_msg'] = df['message'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_msg'])
y = df['label_num']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
