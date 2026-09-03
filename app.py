import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write(
    "This Streamlit demo trains a Random Forest model with SMOTE and shows predictions "
    "for both normal and fraud transactions from the test dataset."
)

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

data = load_data()

st.subheader("Dataset Overview")
st.write("Shape:", data.shape)
st.write("Class distribution:")
st.write(data["Class"].value_counts())

# -------------------------------
# Prepare data
# -------------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Keep original indices for demo
X_test = X_test.copy()
y_test = y_test.copy()

# -------------------------------
# Train model
# -------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_smote, y_train_smote)
    return model

model = train_model(X_train, y_train)

# -------------------------------
# Create index lists for demo
# -------------------------------
fraud_indices = y_test[y_test == 1].index.tolist()
normal_indices = y_test[y_test == 0].index.tolist()

st.subheader("Test Data Demo")
st.write("Fraud cases available in test set:", len(fraud_indices))
st.write("Normal cases available in test set:", len(normal_indices))

tab1, tab2 = st.tabs(["Show Fraud Case", "Show Normal Case"])

# -------------------------------
# Fraud case demo
# -------------------------------
with tab1:
    st.markdown("### Select an actual fraud transaction")

    selected_fraud_index = st.selectbox(
        "Choose a fraud case index",
        fraud_indices
    )

    if st.button("Predict Fraud Case"):
        sample = X_test.loc[[selected_fraud_index]]
        actual = int(y_test.loc[selected_fraud_index])
        prediction = int(model.predict(sample)[0])

        st.write("### Result")
        st.write("Actual Class:", actual)
        st.write("Predicted Class:", prediction)

        if prediction == 1:
            st.error("Fraudulent Transaction Detected")
        else:
            st.warning("Predicted as Legitimate Transaction")

        st.write("### Fraud Transaction Preview")
        st.dataframe(sample)

# -------------------------------
# Normal case demo
# -------------------------------
with tab2:
    st.markdown("### Select an actual normal transaction")

    selected_normal_index = st.selectbox(
        "Choose a normal case index",
        normal_indices[:500]
    )

    if st.button("Predict Normal Case"):
        sample = X_test.loc[[selected_normal_index]]
        actual = int(y_test.loc[selected_normal_index])
        prediction = int(model.predict(sample)[0])

        st.write("### Result")
        st.write("Actual Class:", actual)
        st.write("Predicted Class:", prediction)

        if prediction == 1:
            st.warning("Predicted as Fraudulent Transaction")
        else:
            st.success("Legitimate Transaction")

        st.write("### Normal Transaction Preview")
        st.dataframe(sample)

# -------------------------------
# Footer
# -------------------------------
st.subheader("How to Run")
st.code("streamlit run app_updated.py", language="bash")

st.info(
    "Presentation tip: first show a fraud case from the first tab, then show a normal case "
    "from the second tab. This makes the demo clear and impressive."
)
