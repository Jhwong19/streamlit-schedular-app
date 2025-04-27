import streamlit as st

def about_page():
    """
    Render the about page
    """
    st.title("About This Project")
    
    st.write("""
    ## Project Overview
    
    This project is a **Delivery Route Optimization** tool built using Streamlit. It aims to optimize delivery
    routes for a fleet of vehicles while considering constraints such as delivery time windows, vehicle capacity,
    and traffic conditions.
             
    """)

    # Project overview from about page   
    st.write("""
    This project is a **Delivery Route Optimization** tool that provides an interactive web interface
    for solving complex logistics challenges. It uses advanced algorithms to determine the most efficient
    delivery routes while balancing various constraints and business priorities.
    """)
    
    # Key features in columns
    st.subheader("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Route Optimization
        - Solves the **Vehicle Routing Problem (VRP)** to determine efficient routes
        - Incorporates constraints like time windows and vehicle capacity
        - Prioritizes deliveries based on importance and urgency
        
        #### Map Visualization
        - Displays optimized routes on an interactive map
        - Highlights delivery stops and depot locations
        - Provides detailed route information and statistics
        """)
        
    with col2:
        st.markdown("""
        #### Calendar View
        - Calendar-based schedule for deliveries
        - Shows delivery timeline and workload distribution
        - Helps manage delivery schedules efficiently
        
        #### Interactive Dashboard
        - Real-time delivery status monitoring
        - Data filtering and visualization options
        - Customizable optimization parameters
        """)
    
    # Tools and technologies in an expander
    with st.expander("Tools and Technologies"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### Core Technologies
            - **Python** - Main programming language
            - **Streamlit** - Interactive web interface
            - **Google OR-Tools** - Optimization engine
            """)
            
        with col2:
            st.markdown("""
            #### Data Visualization
            - **Folium** - Interactive maps
            - **Plotly** - Charts and timelines
            - **Pandas** - Data processing
            """)
            
        with col3:
            st.markdown("""
            #### Routing Services
            - **OSRM** - Road distances calculation
            - **TimeMatrix** - Travel time estimation
            - **Geocoding** - Location services
            """)
    
    # Navigation guidance
    st.header("Getting Started")
    st.write("""
    Use the sidebar navigation to explore the application:
    
    - **Map**: Visualize delivery locations and vehicle depots
    - **Optimizer**: Create optimized delivery routes
    - **About**: Learn more about this application
    - **Contact**: Get in touch with the team
    """)

# Make sure the function can be executed standalone
if __name__ == "__main__":
    about_page()