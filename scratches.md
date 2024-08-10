# Sample DataFrame
data = {
    'review': ["I loved the book!", "It was okay.", "Not my type of book.", "Fantastic read, highly recommend!"],
    'stars_given': [5, 3, 2, 5]  # Assuming stars 1-2 = Negative, 3 = Neutral, 4-5 = Positive
}

df = pd.DataFrame(data)

# Convert stars to sentiment labels
df['sentiment'] = df['stars_given'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Logistic Regression
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))