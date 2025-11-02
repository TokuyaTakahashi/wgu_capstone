import glob
import pandas as pd
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier


# Lemmatizing words for better features
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = text.split(' ')
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(lemmatized_tokens)


# Removal of previously created models (useful when attempting and tuning different models)
files_to_delete = glob.glob(os.path.join('trained_models/', '*'))
for file_path in files_to_delete:
    try:
        # Check if it's a file (and not a subfolder)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"  [DELETED] {file_path}")
        else:
            print(f"  [SKIPPED] {file_path} (it is a subfolder)")

    except Exception as e:
        print(f"  [FAILED]  Could not delete {file_path}. Error: {e}")

df = pd.read_csv('aa_dataset-tickets-multi-lang-5-2-50-version.csv')
print(f'Preview of the data \n{df.head()}')

# Determine the number of languages
print(df['language'].value_counts())

# Drop rows where language is not English
df = df[df['language'] == 'en']
print(f'Dataframe after dropping non-English rows: {len(df)} rows')

print(df['queue'].value_counts())
print(f'\nTotal number of classes: {df["queue"].nunique()}')
print(f'Average samples per class: {df.shape[0] / df["queue"].nunique():.2f}')

# Check for class imbalance
queue_counts = df['queue'].value_counts()
print(f'\nClass distribution:')
for queue, count in queue_counts.items():
    percentage = (count / len(df)) * 100
    print(f'  {queue}: {count} ({percentage:.1f}%)')

# Handle missing values in subject and body columns
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')

# # Combine subject and body columns into input_text
df['input_text'] = df['subject'] + ' ' + df['body']
print(f'Created input_text column with {len(df)} rows')
df['input_text'] = df['input_text'].apply(clean_text)

# Remove any rows where input_text is empty or only whitespace
df = df[df['input_text'].str.strip() != '']
print(f'Dataframe after removing empty input_text: {len(df)} rows')

X = df['input_text'].copy()

# y is your 'Routing Group' column
y = df['queue'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train_encoded = LabelEncoder().fit_transform(y_train)

# Define the tuned vectorizer (this is your feature engineering)
tuned_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),  # Use 1- and 2-word n-grams
    max_df=0.90,
    min_df=5,
    max_features=10000
)

# Pipeline 1: ComplementNB
pipe_nb = Pipeline([
    ('vectorizer', tuned_vectorizer),
    ('smote', SMOTE(random_state=42)),
    ('model', ComplementNB())
])

# Param grid for ComplementNB tuning
param_grid_nb = {
    'model__alpha': [0.1, 0.5, 0.75, 1.0],
    'model__norm': [True, False],
    'model__fit_prior': [True, False]
}

# Pipeline 2: LinearSVC
pipe_svc = Pipeline([
    ('vectorizer', tuned_vectorizer),
    ('smote', SMOTE(random_state=42)),
    ('model', LinearSVC())
])

# Param grid for LinearSVC tuning
param_grid_svc = {
    'model__C': [0.1, 1, 10, 100],
}

# Pipeline 3: Random Forest
pipe_rf = Pipeline([
    ('vectorizer', tuned_vectorizer),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Param grid for random forest tuning
param_grid_rf = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [20, 30, None],
    'model__min_samples_leaf': [1, 2, 4],
    'model__min_samples_split': [2, 5, 10]
}

# Pipeline 4: XG Boost
pipe_xg_boost = Pipeline([
    ('vectorizer', tuned_vectorizer),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(objective='multi:softprob', random_state=42, eval_metric='mlogloss', n_jobs=-1))
])

param_grid_vectorizer = {
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'vectorizer__max_features': [10000, 15000],
    'vectorizer__max_df': [0.6, 0.7, 0.8, 0.9],
    'vectorizer__min_df': [5]
}

# --- 4. Create a list of the tuners you want to run ---
tuners_to_run = [
    {
        "name": "complement_nb",
        "estimator": pipe_nb,
        "params": {**param_grid_vectorizer, **param_grid_nb}
    },
    {
        "name": "linear_svc",
        "estimator": pipe_svc,
        "params": {**param_grid_vectorizer, **param_grid_svc}
    },
    {
        "name": "random_forest",
        "estimator": pipe_rf,
        "params": {**param_grid_vectorizer, **param_grid_rf}
    },
    {
        "name": "xg_boost",
        "estimator": pipe_xg_boost,
        "params": {**param_grid_vectorizer}
    }
]

joblib.dump(X_test, f"trained_models/X_test.pkl")
joblib.dump(y_test, f"trained_models/y_test.pkl")

for tuner in tuners_to_run:
    print(f"--- Tuning Model: {tuner['name']} ---")

    # Set up the Randomized Search for *this* model
    random_search = RandomizedSearchCV(
        estimator=tuner["estimator"],
        param_distributions=tuner["params"],
        n_iter=5,
        cv=3,
        scoring='f1_weighted',
        random_state=42,
        verbose=1
    )
    if tuner['name'] == 'xg_boost':
        random_search.fit(X_train, y_train_encoded)
    else:
        random_search.fit(X_train, y_train)
    # Save model for predictions
    joblib.dump(random_search.best_estimator_, f"trained_models/{tuner['name']}_ticket_classifier.pkl")

