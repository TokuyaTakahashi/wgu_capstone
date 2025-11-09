import plotly.express as px
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report

cnb = joblib.load(os.path.join('trained_models', 'complement_nb_ticket_classifier.pkl'))
log_reg = joblib.load(os.path.join('trained_models', 'log_reg_ticket_classifier.pkl'))
rfc = joblib.load(os.path.join('trained_models', 'random_forest_ticket_classifier.pkl'))
X_test = joblib.load(os.path.join('trained_models', 'X_test.pkl'))
y_test = joblib.load(os.path.join('trained_models', 'y_test.pkl'))
cnb_predictions = cnb.predict(X_test)
log_reg_predictions = log_reg.predict(X_test)
rfc_predictions = rfc.predict(X_test)
cnb_report = classification_report(y_test, cnb_predictions, output_dict=True)
log_reg_report = classification_report(y_test, log_reg_predictions, output_dict=True)
rfc_report = classification_report(y_test, rfc_predictions, output_dict=True)
# print(cnb_report)
print(f'CNB outcome: {cnb_report['weighted avg']}')
print(f'LogReg outcome: {log_reg_report['weighted avg']}')
print(f'RFC outcome: {rfc_report['weighted avg']}')
