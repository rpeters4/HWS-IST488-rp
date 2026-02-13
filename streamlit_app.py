import streamlit as st

# Define pages
# Define pages
hw1 = st.Page("HW/Homework1.py", title="Homework 1", icon=":material/description:")
hw2 = st.Page("HW/HW2.py", title="Homework 2", icon=":material/description:")
hw3 = st.Page("HW/HW3.py", title="Homework 3", icon=":material/description:")
hw4 = st.Page("HW/HW4.py", title="Homework 4", icon=":material/description:", default=True)

# Create navigation
pg = st.navigation([hw1, hw2, hw3, hw4])

# Configure page
st.set_page_config(page_title="HW Manager", page_icon=":material/school:")

# Run the selected pagea
pg.run()