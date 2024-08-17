Logistic Regression 

train time 14.7s
accuracy: 0.76

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example data
X = ["I loved the book!", "It was okay.", "Not my type of book.", "Fantastic read, highly recommend!"]
y = ["positive", "neutral", "negative", "positive"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a pipeline
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

Naive Bayes

train time: 7.1s
accuracy: 0.73

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example data
X = ["I loved the book!", "It was okay.", "Not my type of book.", "Fantastic read, highly recommend!"]
y = ["positive", "neutral", "negative", "positive"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

```

Support Vector Machines (SVM)

train time 2m:28s
accuracy: 0.79

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example data
X = ["I loved the book!", "It was okay.", "Not my type of book.", "Fantastic read, highly recommend!"]
y = ["positive", "neutral", "negative", "positive"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a pipeline
model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```


Sales analysis

import pandas as pd

# Example DataFrame
data = {
    'book_title': ['Book A', 'Book A', 'Book A', 'Book B', 'Book B', 'Book C', 'Book C', 'Book C'],
    'sentiment': ['positive', 'negative', 'positive', 'neutral', 'positive', 'negative', 'positive', 'neutral']
}

df = pd.DataFrame(data)

# Step 1: Filter to include only positive reviews
positive_reviews = df[df['sentiment'] == 'positive']

# Step 2: Group by title to get the count of positive reviews per book
positive_count = positive_reviews.groupby('book_title').size().reset_index(name='positive_count')

# Step 3: Group the entire dataset by title to get the total count of reviews per book
total_count = df.groupby('book_title').size().reset_index(name='total_count')

# Step 4: Merge the positive count and total count DataFrames
merged_df = pd.merge(positive_count, total_count, on='book_title')

# Step 5: Calculate the percentage of positive reviews
merged_df['positive_percentage'] = (merged_df['positive_count'] / merged_df['total_count']) * 100

# Display the result
print(merged_df)


References
Cem Dilmegani, Ezgi Alp, PhD., 2024; 6 Approaches for Sentiment Analysis Machine Learning in 2024. Available at:
https://research.aimultiple.com/sentiment-analysis-machine-learning/ (Accessed: 17 August 2024)