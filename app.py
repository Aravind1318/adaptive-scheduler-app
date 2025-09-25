import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Driven Adaptive Scheduler",
    page_icon="⚙",
    layout="wide"
)

# --- Custom Background & Theme ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] { background-color: #121212; color: #EAEAEA; }
[data-testid="stSidebar"] { background-color: #1E1E1E; color: #EAEAEA; }
[data-testid="stHeader"] { background-color: #0A84FF; color: white; }
[data-testid="stMarkdownContainer"] h1, h2, h3, h4 { color: #0A84FF; font-weight: bold; }
label, .stTextInput label, .stNumberInput label, .stTextArea label { color: #EAEAEA !important; font-weight: 500; }
.stTextInput input, .stNumberInput input, textarea { background-color: #1E1E1E !important; color: #EAEAEA !important; border-radius: 8px; border: 1px solid #333333; }
[data-testid="stDataFrame"] { background-color: #1E1E1E; color: #EAEAEA; }
[data-testid="stNotification"] { border-radius: 10px; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- Title and Description ---
st.title("⚙ AI-Driven Adaptive Scheduler")
st.markdown("""
    Upload your production dataset and let the AI predict machine assignments
    for new tasks based on historical data.
""")
st.divider()

# --- Upload Dataset ---
st.header("1. Upload your dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # --- Select Features and Target ---
    st.header("2. Train AI Model")
    target_column = st.selectbox(
        "Select the target column to predict",
        df.columns,
        index=df.columns.get_loc("Machine_Available") if "Machine_Available" in df.columns else 0
    )
    feature_columns = st.multiselect(
        "Select feature columns",
        [col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column]
    )

    if st.button("Train Model"):
        if not feature_columns:
            st.error("Select at least one feature column!")
        else:
            X = pd.get_dummies(df[feature_columns])
            y = df[target_column]

            model = RandomForestClassifier()
            model.fit(X, y)

            # Save model & training data in session_state
            st.session_state["model"] = model
            st.session_state["X_columns"] = X.columns
            st.session_state["feature_columns"] = feature_columns
            st.session_state["df"] = df

            st.success("✅ AI Model trained successfully!")

# --- Predict New Task ---
if "model" in st.session_state:
    st.header("3. Predict Machine for a New Task")
    df = st.session_state["df"]
    feature_columns = st.session_state["feature_columns"]

    with st.form(key="predict_form"):
        new_task_data = {}
        for col in feature_columns:
            if df[col].dtype == 'object':
                new_task_data[col] = st.selectbox(col, df[col].unique())
            else:
                new_task_data[col] = st.number_input(col, min_value=0)

        submit_pred = st.form_submit_button("Predict Machine")
        if submit_pred:
            new_task_df = pd.DataFrame([new_task_data])
            new_task_encoded = pd.get_dummies(new_task_df)
            new_task_encoded = new_task_encoded.reindex(columns=st.session_state["X_columns"], fill_value=0)

            prediction = st.session_state["model"].predict(new_task_encoded)[0]
            st.success(f"✅ Recommended Machine: {prediction}")

st.divider()
st.markdown("---")
st.caption("© 2025 Adaptive Scheduling Team. All rights reserved.")
