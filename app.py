import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

st.title("Adaptive Scheduling: AI-driven Machine & Manpower Allocation")

# --- Step 1: Upload Dataset ---
uploaded_file = st.file_uploader("Upload your scheduling dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Define features and target
    target_col = "Machine_Available"  # what we want to predict
    feature_cols = [col for col in df.columns if col not in ["Job_ID", target_col]]
    X = df[feature_cols]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "scheduling_model.pkl")

    st.success("✅ Model trained successfully!")

    # --- Step 2: Dynamic Scheduling Simulation ---
    st.subheader("Real-time Task Scheduling")

    # Initialize availability trackers
    if "machine_loads" not in st.session_state:
        st.session_state.machine_loads = {m: 0 for m in sorted(df[target_col].unique())}
    if "manpower_available" not in st.session_state:
        st.session_state.manpower_available = 50  # assume 50 workers total

    # Task input
    task_type = st.selectbox("Task Type", label_encoders["Task_Type"].classes_)
    load_units = st.number_input("Load Units", min_value=10, max_value=500, value=100)
    manpower_req = st.number_input("Manpower Required", min_value=1, max_value=20, value=5)
    priority = st.selectbox("Priority", label_encoders["Priority"].classes_)
    est_time = st.number_input("Estimated Time (hrs)", min_value=1, max_value=24, value=6)
    deadline = st.number_input("Deadline (hrs)", min_value=1, max_value=48, value=12)

    if st.button("Allocate Task"):
        # Encode input
        input_data = pd.DataFrame([[
            label_encoders["Task_Type"].transform([task_type])[0],
            load_units,
            manpower_req,
            label_encoders["Priority"].transform([priority])[0],
            est_time,
            deadline
        ]], columns=feature_cols)

        # Predict machine probabilities
        proba = model.predict_proba(input_data)[0]
        machine_ranking = np.argsort(proba)[::-1]  # descending order

        assigned_machine = None
        for m in machine_ranking:
            # Check machine availability & manpower
            if st.session_state.manpower_available >= manpower_req:
                assigned_machine = m
                st.session_state.machine_loads[m] += est_time
                st.session_state.manpower_available -= manpower_req
                break

        if assigned_machine is not None:
            st.success(f"✅ Task allocated to Machine {assigned_machine} | "
                       f"Manpower Remaining: {st.session_state.manpower_available}")
        else:
            st.error("❌ No machine/manpower available for this task.")

    # Show current system state
    st.subheader("Current System Load")

# Convert keys & values to safe Python types
safe_machine_loads = {str(int(k)): int(v) for k, v in st.session_state.machine_loads.items()}

st.json(safe_machine_loads)  # safer for dicts
st.write("Available Manpower:", int(st.session_state.manpower_available))

