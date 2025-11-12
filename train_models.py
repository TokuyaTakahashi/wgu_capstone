import glob
import os
import joblib
import nltk
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nlpaug.augmenter.word as naw

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')


# Lemmatizing words for better features
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = text.split(' ')
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(lemmatized_tokens)


os.makedirs("trained_models", exist_ok=True)
# Removal of previously created models (useful when attempting and tuning different models)
files_to_delete = glob.glob(os.path.join('trained_models/', '*'))
for file_path in files_to_delete:
    try:
        # Check if it's a file (and not a subfolder)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"  [DELETED] {file_path}")
        else:
            print(f"  [SKIPPED] {file_path} ")

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

# Subject + Body column
X = df['input_text'].copy()

# Routing Group column
y = df['queue'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

df_train = pd.DataFrame({
    'input_text': X_train,
    'queue_encoded': y_train
})

# Identify minority classes
class_counts = df_train['queue_encoded'].value_counts()
minority_classes = class_counts[class_counts < 1000].index
minority_df = df_train[df_train['queue_encoded'].isin(minority_classes)]

print(f"Original training size: {len(df_train)}")
print(f"Found {len(minority_classes)} minority classes to augment.")

aug = naw.SynonymAug(aug_src='wordnet', stopwords=ENGLISH_STOP_WORDS)

# Generate new minority samples by replacing words with synonyms to reduce imbalance
new_texts = aug.augment(minority_df['input_text'].tolist())
new_labels = minority_df['queue_encoded'].tolist()

augmented_df = pd.DataFrame({
    'input_text': new_texts,
    'queue_encoded': new_labels
})

# Combine newly created samples to existing dataset
df_train_final = pd.concat([df_train, augmented_df], ignore_index=True)

X_train_augmented = df_train_final['input_text']
y_train_augmented = df_train_final['queue_encoded']

print(f"New augmented training size: {len(df_train_final)}")

# Define the default vectorizer
tuned_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.90,
    min_df=5,
    max_features=100000
)

# Pipeline 1: ComplementNB
pipe_nb = Pipeline([
    ('vectorizer', tuned_vectorizer),
    ('over_sampler', SMOTE(random_state=42)),
    ('model', ComplementNB())
])

# Param grid for ComplementNB tuning
param_grid_nb = {
    'model__alpha': [0.1, 0.5, 0.75, 1.0],
    'model__norm': [True, False],
    'model__fit_prior': [True, False]
}

# Pipeline 2: LogisticRegression
pipe_log_reg = Pipeline([
    ('vectorizer', tuned_vectorizer),
    ('over_sampler', SMOTE(random_state=42)),
    ('model', LogisticRegression(solver='saga', n_jobs=-1))
])

# Param grid for LogisticRegression tuning
param_grid_logreg = {
    'model__C': [0.1, 1, 10]
}

# Pipeline 3: Random Forest
pipe_rf = Pipeline([
    ('vectorizer', tuned_vectorizer),
    ('over_sampler', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Param grid for random forest tuning
param_grid_rf = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__max_depth': [1, 20, None],
    'model__min_samples_leaf': [1, 20, 200],
    'model__min_samples_split': [2, 10, 20],
    'model__criterion': ['log_loss', 'entropy'],
    'model__class_weight' : ['balanced_subsample', 'balanced']
}

param_sampler = {
    'over_sampler': [
        SMOTE(random_state=42),
        ADASYN(random_state=42),
        ADASYN(random_state=42, sampling_strategy='not majority'),
        RandomOverSampler(random_state=42),
        'passthrough'
    ],
}


# parameters for tuning the vectorizer
param_grid_vectorizer = {
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'vectorizer__max_features': [10000, 15000],
    'vectorizer__max_df': [0.6, 0.7, 0.8, 0.9],
    'vectorizer__min_df': [5]
}

tuners_to_run = [
    {
        "name": "complement_nb",
        "estimator": pipe_nb,
        "params": {**param_grid_vectorizer, **param_sampler, **param_grid_nb}
    },
    {
        "name": "log_reg",
        "estimator": pipe_log_reg,
        "params": {**param_grid_vectorizer, **param_sampler, **param_grid_logreg}
    },
    {
        "name": "random_forest",
        "estimator": pipe_rf,
        "params": {**param_grid_vectorizer, **param_sampler, **param_grid_rf}
    }
]

joblib.dump(X_test, f"trained_models/X_test.pkl")
joblib.dump(y_test, f"trained_models/y_test.pkl")

# Hyperparameter tuning
for tuner in tuners_to_run:
    print(f"--- Tuning Model: {tuner['name']} ---")

    # Set up the Randomized Search
    random_search = RandomizedSearchCV(
        estimator=tuner["estimator"],
        param_distributions=tuner["params"],
        n_iter=15,
        cv=5,
        scoring='f1_macro',
        random_state=42,
        verbose=1,
        error_score='raise'
    )
    random_search.fit(X_train_augmented, y_train_augmented)
    # Save model for predictions
    joblib.dump(random_search.best_estimator_, f"trained_models/{tuner['name']}_ticket_classifier.pkl")

