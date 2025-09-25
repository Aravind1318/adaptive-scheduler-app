import streamlit as st
import pandas as pd
import random
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Adaptive Scheduler",
    page_icon="⚙",
    layout="wide"
)

# --- Custom Background & Theme ---
page_bg = """
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #121212; 
    color: #EAEAEA;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E1E1E; 
    color: #EAEAEA;
}

/* Header */
[data-testid="stHeader"] {
    background-color: #0A84FF; 
    color: white;
}

/* Headings */
[data-testid="stMarkdownContainer"] h1, 
[data-testid="stMarkdownContainer"] h2, 
[data-testid="stMarkdownContainer"] h3, 
[data-testid="stMarkdownContainer"] h4 {
    color: #0A84FF; 
    font-weight: bold;
}

/* Form Labels (fix grey text issue) */
label, .stTextInput label, .stNumberInput label, .stTextArea label {
    color: #EAEAEA !important;
    font-weight: 500;
}

/* Input Fields */
.stTextInput input, .stNumberInput input, textarea {
    background-color: #1E1E1E !important;
    color: #EAEAEA !important;
    border-radius: 8px;
    border: 1px solid #333333;
}

/* DataFrame Tables */
[data-testid="stDataFrame"] {
    background-color: #1E1E1E; 
    color: #EAEAEA;
}

/* Info / Warning / Success Boxes */
[data-testid="stNotification"] {
    border-radius: 10px;
}
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
    worker_skills_str = st.text_area("Enter worker skills (comma-separated):", "A, B, B, C, A, A, B, C, A, C, B, A, C, B, B, A, C, B, A, A")

st.divider()

st.header("2. Run the Scheduling Algorithm")
if st.button("Generate Adaptive Schedule", type="primary"):
    # --- Input Parsing and Validation ---
    try:
        machine_loads = [int(load.strip()) for load in machine_load_str.split(',') if load.strip()]
        worker_skills = [skill.strip().upper() for skill in worker_skills_str.split(',') if skill.strip()]

        if len(machine_loads) > num_machines:
            st.warning(f"Number of machine loads ({len(machine_loads)}) exceeds the number of machines ({num_machines}). Only using the first {num_machines} values.")
            machine_loads = machine_loads[:num_machines]
        elif len(machine_loads) < num_machines:
            st.warning(f"Number of machine loads ({len(machine_loads)}) is less than the number of machines ({num_machines}). The remaining machines will be unassigned.")

        if len(worker_skills) > num_manpower:
            st.warning(f"Number of worker skills ({len(worker_skills)}) exceeds the number of workers ({num_manpower}). Only using the first {num_manpower} values.")
            worker_skills = worker_skills[:num_manpower]
        elif len(worker_skills) < num_manpower:
            st.warning(f"Number of worker skills ({len(worker_skills)}) is less than the number of workers ({num_manpower}). The remaining workers will be considered unassigned.")

    except (ValueError, IndexError):
        st.error("❌ **Invalid input format.** Please ensure machine loads are numbers and skills are single characters, both separated by commas.")
        st.stop()

    st.success("✅ Schedule is being generated...")
    st.snow()

    # --- Scheduling Algorithm ---
    workers_by_skill = {'A': [], 'B': [], 'C': []}
    for i, skill in enumerate(worker_skills):
        workers_by_skill[skill].append(f"W{i+1}")

    machines = [{"id": i+1, "load": load} for i, load in enumerate(machine_loads)]
    machines.sort(key=lambda x: x["load"], reverse=True)
    
    assigned_schedule = []
    unassigned_workers = workers_by_skill['A'] + workers_by_skill['B'] + workers_by_skill['C']
    
    for machine in machines:
        assigned_worker = None
        if workers_by_skill['A']:
            assigned_worker = workers_by_skill['A'].pop(0)
        elif workers_by_skill['B']:
            assigned_worker = workers_by_skill['B'].pop(0)
        elif workers_by_skill['C']:
            assigned_worker = workers_by_skill['C'].pop(0)
            
        if assigned_worker:
            worker_id = re.sub(r'[^0-9]', '', assigned_worker)
            assigned_schedule.append({
                "Machine ID": machine["id"],
                "Production Load": machine["load"],
                "Assigned Worker ID": int(worker_id),
                "Worker Skill": worker_skills[int(worker_id)-1]
            })
            unassigned_workers.remove(assigned_worker)
            
    # --- Display Results ---
    st.markdown("### Generated Schedule")
    
    if assigned_schedule:
        df = pd.DataFrame(assigned_schedule)
        st.dataframe(df.set_index("Machine ID"), use_container_width=True)
    else:
        st.warning("No schedule could be generated. Check your input data.")
        
    st.markdown("---")
    st.info(f"**Summary:** {len(assigned_schedule)} machines were assigned. {len(unassigned_workers)} workers are unassigned.")
    
st.divider()

st.markdown("---")
st.caption("© 2025 Adaptive Scheduling Team. All rights reserved.")
