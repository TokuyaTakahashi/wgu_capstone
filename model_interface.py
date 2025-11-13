import plotly.express as px
import pandas as pd
import joblib
import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


@st.cache_resource
def load_all_models(model_map, model_dir='trained_models'):
    loaded_models = {}
    for name, filename in model_map.items():
        model_path = os.path.join(model_dir, filename)
        loaded_models[name] = joblib.load(model_path)
    return loaded_models


def apply_prediction():
    if st.session_state.ticket_input:
        loaded_pipeline = st.session_state.all_models[st.session_state.sidebar]
        ticket_desc = st.session_state.ticket_input
        st.session_state.prediction = loaded_pipeline.predict([ticket_desc])
        st.session_state.prediction_prob = loaded_pipeline.predict_proba([ticket_desc])
    else:
        st.session_state.prediction = "None"


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


trained_models = {
    "Naive Bayes": "complement_nb_ticket_classifier.pkl",
    "Logistic Regression": "log_reg_ticket_classifier.pkl",
    "Random Forest": "random_forest_ticket_classifier.pkl",
}

if 'all_models' not in st.session_state:
    st.session_state.all_models = load_all_models(trained_models)

st.set_page_config(layout="centered")

# Sidebar - Select box for model selection
st.sidebar.selectbox('Model Selector', options=trained_models.keys(), key="sidebar", on_change=apply_prediction)

# Sidebar - Legend for all possible routing groups
with st.sidebar:
    st.header("Routing Groups")
    for group, (emoji, color) in routing_group_styles.items():
        st.badge(group, icon=emoji, color=color)

# Load trained models and test data for predictions and accuracy metrics
selected_model = st.session_state.all_models[st.session_state.sidebar]
X_test = joblib.load(os.path.join('trained_models', 'X_test.pkl'))
y_test = joblib.load(os.path.join('trained_models', 'y_test.pkl'))
encoder = LabelEncoder()
encoder.fit_transform(y_test)

# Generate predictions using the selected model
predictions = selected_model.predict(X_test)
class_names = selected_model.named_steps['model'].classes_
vectorizer = selected_model.named_steps['vectorizer']
model = selected_model.named_steps['model']
viridis_palette = list(sns.color_palette(palette='viridis',
                                         n_colors=len(class_names)).as_hex())
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
            prediction = st.session_state.prediction[0]
            df_proba = pd.DataFrame({
                "Queue": class_names,
                "Probability": st.session_state.prediction_prob[0]
            })
            fig = px.pie(df_proba, values='Probability', names='Queue', color_discrete_sequence=viridis_palette, title='Prediction Probability')
            fig.update_traces(textposition='outside', textinfo='percent+label',
                              hole=0.6, hoverinfo="label+percent+name")
        else:
            fig = None
            prediction = "None"
        styling = routing_group_styles[prediction]
        st.badge(prediction, icon=f"{styling[0]}", color=f'{styling[1]}', width="stretch")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Classification Report")
        report = classification_report(y_test, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    with tab3:
        # Show routing group sample data count
        st.subheader('Ticket Distribution by Queue')
        df_route_group = pd.DataFrame(y_test.value_counts())
        df_route_group['percent'] = [round(i*100/sum(df_route_group['count']),1) for i in df_route_group['count']]
        color_tgroup = dict(zip(df_route_group.index, viridis_palette))
        # add a color to the dictionary
        color_tgroup['(?)'] = '#e9e9e9'

        # create a list of percentages for labeling
        labels = [f'{i}%' for i in df_route_group['percent']]

        fig = px.treemap(df_route_group, path=[px.Constant('All Routing Groups'), df_route_group.index],
                         values=df_route_group['percent'],
                         color=df_route_group.index,
                         color_discrete_map=color_tgroup,
                         hover_name=labels)
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Plot confusion matrix
        plt.style.use('dark_background')
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions, labels=class_names, normalize='true')
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='1.0%', cmap='viridis',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
