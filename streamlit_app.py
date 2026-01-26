import streamlit as st

# Define pages
hw1 = st.Page("Homework1.py", title="Homework 1", icon=":material/description:", default=True)

# Future homeworks can be added here
# hw2 = st.Page("Homework2.py", title="Homework 2", icon=":material/extension:")

# Create navigation
pg = st.navigation([hw1])

# Configure page
st.set_page_config(page_title="IST 488 Homework", page_icon=":material/school:")

# Run the selected page
pg.run()