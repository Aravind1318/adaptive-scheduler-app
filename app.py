import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

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
st.title("⚙ AI-Driven Adaptive Scheduling")
st.markdown("""
    Welcome to the *Adaptive Scheduler*, an AI-driven system that dynamically allocates machines and manpower
    based on real-time production loads.
""")
st.divider()

# --- Input Sections ---
st.header("1. Enter Production Load Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Machines")
    num_machines = st.number_input("Number of available machines:", min_value=1, value=10)
    machine_load_str = st.text_area("Enter machine loads (comma-separated):", "")

with col2:
    st.subheader("Manpower")
    num_manpower = st.number_input("Number of available workers:", min_value=1, value=20)
    worker_skills_str = st.text_area("Enter worker skills (comma-separated):", 
                                     "A, B, B, C, A, A, B, C, A, C, B, A, C, B, B, A, C, B, A, A")

st.divider()

st.header("2. Run the AI Scheduling Algorithm")
if st.button("Generate Adaptive Schedule", type="primary"):

    # --- Input Parsing ---
    try:
        machine_loads = [int(load.strip()) for load in machine_load_str.split(',') if load.strip()]
        worker_skills = [skill.strip().upper() for skill in worker_skills_str.split(',') if skill.strip()]

        if len(machine_loads) > num_machines:
            st.warning(f"Number of machine loads ({len(machine_loads)}) exceeds the number of machines ({num_machines}). Only using the first {num_machines} values.")
            machine_loads = machine_loads[:num_machines]
        elif len(machine_loads) < num_machines:
            st.warning(f"Number of machine loads ({len(machine_loads)}) is less than the number of machines ({num_machines}). Remaining machines will be unassigned.")

        if len(worker_skills) > num_manpower:
            st.warning(f"Number of worker skills ({len(worker_skills)}) exceeds the number of workers ({num_manpower}). Only using the first {num_manpower} values.")
            worker_skills = worker_skills[:num_manpower]
        elif len(worker_skills) < num_manpower:
            st.warning(f"Number of worker skills ({len(worker_skills)}) is less than the number of workers ({num_manpower}). Remaining workers will be considered unassigned.")

    except (ValueError, IndexError):
        st.error("❌ Invalid input format. Machine loads must be numbers, skills must be A/B/C, separated by commas.")
        st.stop()

    st.success("✅ Schedule is being generated...")

    # --- Generate Simulated Dataset for AI Model ---
    all_workers = [f"W{i+1}" for i in range(len(worker_skills))]
    data = []
    for m_id, load in enumerate(machine_loads, start=1):
        for i, skill in enumerate(worker_skills):
            if skill == 'A':
                productivity = load + random.randint(5, 15)
            elif skill == 'B':
                productivity = load + random.randint(0, 10)
            else:
                productivity = load + random.randint(-5, 5)
            data.append({
                'Machine Load': load,
                'Worker Skill': skill,
                'Worker': all_workers[i],
                'Productivity': productivity
            })

    df = pd.DataFrame(data)

    # --- Train AI Model ---
    X = pd.get_dummies(df[['Machine Load', 'Worker Skill']])
    y = df['Worker']
    model = RandomForestClassifier()
    model.fit(X, y)

    # --- Predict Best Worker for Each Machine ---
    assigned_schedule = []
    used_workers = set()

    for load in machine_loads:
        predictions = []
        for i, skill in enumerate(worker_skills):
            if all_workers[i] in used_workers:
                continue
            x_input = pd.DataFrame([{
                'Machine Load': load,
                'Worker Skill_A': 1 if skill=='A' else 0,
                'Worker Skill_B': 1 if skill=='B' else 0,
                'Worker Skill_C': 1 if skill=='C' else 0
            }])
            pred = model.predict(x_input)[0]
            # Simulate predicted productivity from dataset
            productivity = df[(df['Machine Load']==load) & (df['Worker']==pred)]['Productivity'].values[0]
            predictions.append((pred, skill, productivity))
        if predictions:
            # Pick worker with highest predicted productivity
            best_worker, best_skill, best_prod = max(predictions, key=lambda x: x[2])
            machine_id = machine_loads.index(load)+1
            assigned_schedule.append({
                "Machine ID": machine_id,
                "Production Load": load,
                "Assigned Worker ID": best_worker,
                "Worker Skill": best_skill,
                "Predicted Productivity": best_prod
            })
            used_workers.add(best_worker)

    # --- Display Results ---
    st.markdown("### Generated AI Schedule")
    if assigned_schedule:
        df_schedule = pd.DataFrame(assigned_schedule)
        st.dataframe(df_schedule.set_index("Machine ID"), use_container_width=True)
    else:
        st.warning("No schedule could be generated. Check your input data.")

    st.markdown("---")
    st.info(f"**Summary:** {len(assigned_schedule)} machines were assigned. {len(worker_skills) - len(used_workers)} workers are unassigned.")

st.divider()
st.markdown("---")
st.caption("© 2025 Adaptive Scheduling Team. All rights reserved.")
