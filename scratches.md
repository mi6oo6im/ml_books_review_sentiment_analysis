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

Datasets:
Ansh Tanwar, 2023; Top 100 Bestselling Book Reviews on Amazon. Available at:  https://www.kaggle.com/datasets/anshtanwar/top-200-trending-books-with-reviews/data (Accessed: 10 August 2024)

Trrishan, 2021; Top 100 Science Fiction Books and their Reviews. Available at: https://www.kaggle.com/datasets/notkrishna/top-100-science-fiction-books-and-their-reviews (Accessed: 10 August 2024)

Wikipedia, 2024; List of best-selling books. Available at: https://en.wikipedia.org/wiki/List_of_best-selling_books (Accessed: 10 August 2024)


Articles:
Cem Dilmegani, Ezgi Alp, PhD., 2024; 6 Approaches for Sentiment Analysis Machine Learning in 2024. Available at:
https://research.aimultiple.com/sentiment-analysis-machine-learning/ (Accessed: 17 August 2024)

US Library of congress, 2023; Frequently Asked Questions: History, Humanities & Social Sciences: Can you tell me how many copies of a book were sold or printed? Available at:
https://ask.loc.gov/history-humanities-social-sciences/faq/383956 (Accessed: 18 August 2024)

Harvard Library, 2005; Where can I find detailed book sales figures or statistics? Available at: https://ask.library.harvard.edu/faq/81944 (Accessed: 18 August 2024)

SciKit-Learn, Getting Started. Available at: https://scikit-learn.org/stable/getting_started.html (Accessed: 10 August 2024)

Natural Language Toolkit - NLTK. Available at: https://www.nltk.org/ (Accessed: 10 August 2024)