# Main entry point of the Streamlit application

import streamlit as st
import sys
from pathlib import Path

# Import all pages from the pages module 
from src.pages import home_page, about_page, contact_page, map_page, optimize_page

def main():
    st.set_page_config(
        page_title="Delivery Route Optimization",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.sidebar.title("Navigation")
    
    # Sidebar navigation
    pages = {
        "Home": home_page,
        "Map": map_page,
        "Optimizer": optimize_page,  # Add the new page
        "About": about_page,
        "Contact": contact_page
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Render the selected page
    pages[selection]()

if __name__ == "__main__":
    main()
