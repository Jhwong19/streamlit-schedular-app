import streamlit as st

def contact_page():
    """
    Render the contact page
    """
    st.title("Contact")
    
    st.write("""
    ### Get in Touch
    
    For questions, feedback, or suggestions about this application, please feel free to reach out.
    
    **Email**: jinghui.me@gmail.com
    
    ### Repository
    
    This project is open-source. Find the code on GitHub:
    [streamlit-schedular-app](https://github.com/yourusername/streamlit-schedular-app)
    
    ### License
    
    This project is licensed under the MIT License. See the LICENSE file for more details.
    """)

# Make sure the function can be executed standalone
if __name__ == "__main__":
    contact_page()