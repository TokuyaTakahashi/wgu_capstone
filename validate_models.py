import joblib
import os
from sklearn.metrics import classification_report

X_test = joblib.load(os.path.join('trained_models', 'X_test.pkl'))
y_test = joblib.load(os.path.join('trained_models', 'y_test.pkl'))


# Validate models by getting their classification report and printing out f1-weighted scores
def evaluate_models(models_to_test, X_test, y_test, model_dir='trained_models'):
    # This dictionary will store the full report for each model
    all_reports = {}

    print(" Starting Model Evaluation ")

    # Loop through the dictionary you provide
    for name, filename in models_to_test.items():
        model_path = os.path.join(model_dir, filename)
        model = joblib.load(model_path)
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        all_reports[name] = report
        f1 = report['weighted avg']['f1-score']
        print(f'{name} Weighted F1-Score: {f1:.4f}')
    print(" Evaluation Complete ")

    return all_reports


models_to_evaluate = {
    'CNB': 'complement_nb_ticket_classifier.pkl',
    'LogReg': 'log_reg_ticket_classifier.pkl',
    'RFC': 'random_forest_ticket_classifier.pkl'
}

all_reports = evaluate_models(models_to_evaluate, X_test, y_test)
