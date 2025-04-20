# Main entry point of the Streamlit application

import streamlit as st
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.components import *
from src.pages import *

def main():
    st.title("Streamlit App Template")
    st.sidebar.title("Navigation")
    
    # Sidebar navigation
    pages = {
        "Home": home_page,
        "About": about_page,
        "Contact": contact_page
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    # Render the selected page
    pages[selection]()

if __name__ == "__main__":
    main()