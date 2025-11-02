import numpy as np
import pandas as pd
import joblib
import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def apply_prediction():
    if st.session_state.ticket_input:
        full_path = os.path.join('trained_models', trained_models[st.session_state.sidebar])
        loaded_pipeline = joblib.load(full_path)
        ticket_desc = st.session_state.ticket_input
        st.session_state.prediction = loaded_pipeline.predict([ticket_desc])
    else:
        st.session_state.prediction = "None"


trained_models = {"Naive Bayes": "complement_nb_ticket_classifier.pkl",
                  "Linear Support Vector": "linear_svc_ticket_classifier.pkl",
                  "Random Forest": "random_forest_ticket_classifier.pkl",
                  "XG Boost": "xg_boost_ticket_classifier.pkl"}

routing_group_styles = {
    "Billing and Payments": ("💵", "green"),
    "Customer Service": ("💁", "blue"),
    "General Inquiry": ("❔", "gray"),
    "Human Resources": ("💼", "violet"),
    "IT Support": ("💻", "blue"),
    "Product Support": ("🛍️", "orange"),
    "Returns and Exchanges": ("↩️", "red"),
    "Sales and Pre-Sales": ("📈", "green"),
    "Service Outages and Maintenance": ("⚠️", "red"),
    "Technical Support": ("🔧", "orange"),
    "None": ("❌", "grey")
}

st.set_page_config(layout="centered")

# Sidebar - Select box for model selection
st.sidebar.selectbox('Model Selector', options=trained_models.keys(), key="sidebar", on_change=apply_prediction)

# Sidebar - Legend for all possible routing groups
with st.sidebar:
    st.header("Routing Groups")
    for group, (emoji, color) in routing_group_styles.items():
        st.badge(group, icon=emoji, color=color)

# Load trained models and test data for predictions and accuracy metrics
selected_model = joblib.load(os.path.join('trained_models', trained_models[st.session_state.sidebar]))
X_test = joblib.load(os.path.join('trained_models', 'X_test.pkl'))
y_test = joblib.load(os.path.join('trained_models', 'y_test.pkl'))

encoder = LabelEncoder()
y_test_encoded = encoder.fit_transform(y_test)

# Generate predictions using the selected model
predictions = selected_model.predict(X_test)
class_names = selected_model.named_steps['model'].classes_
vectorizer = selected_model.named_steps['vectorizer']
model = selected_model.named_steps['model']

st.header("Ticket Classifier")
# Main content - Tabs for live prediction, model accuracy, and data visualizations
with st.container():
    tab1, tab2, tab3 = st.tabs(["Live Prediction", f"{st.session_state.sidebar} Performance", "Visualizations"])

    with tab1:
        # Text area to enter ticket input for routing group prediction
        st.text_area('Ticket Description', None, key="ticket_input", on_change=apply_prediction,
                     placeholder="Enter a short description for an IT ticket to get a corresponding routing group")

        # Display prediction as a badge
        if st.session_state.ticket_input and 'prediction' in st.session_state:
            prediction = encoder.inverse_transform(st.session_state.prediction)[0] if model.__class__.__name__ == 'XGBClassifier' else st.session_state.prediction[0]
        else:
            prediction = "None"
        styling = routing_group_styles[prediction]
        st.badge(prediction, icon=f"{styling[0]}", color=f'{styling[1]}', width="stretch")
    with tab2:
        st.subheader("Classification Report")
        if model.__class__.__name__ == 'XGBClassifier':
            report = classification_report(y_test_encoded, predictions, output_dict=True)
        else:
            report = classification_report(y_test, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    with tab3:
        # Show routing group sample data count
        st.subheader("Routing Group Ticket Count")
        st.bar_chart(y_test.value_counts())

        # Plot confusion matrix
        plt.style.use('dark_background')
        st.subheader("Confusion Matrix")
        if model.__class__.__name__ == 'XGBClassifier':
            cm = confusion_matrix(y_test_encoded, predictions, labels=class_names, normalize='true')
        else:
            cm = confusion_matrix(y_test, predictions, labels=class_names, normalize='true')
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='1.0%', cmap='viridis',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
