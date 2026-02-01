import streamlit as st

# Define pages
hw1 = st.Page("HW/Homework1.py", title="Homework 1", icon=":material/description:", default=True)
hw2 = st.Page("HW/HW2.py", title="Homework 2", icon=":material/description:")

# Create navigation
pg = st.navigation([hw1, hw2])

# Configure page
st.set_page_config(page_title="HW Manager", page_icon=":material/school:")

# Run the selected page
pg.run()