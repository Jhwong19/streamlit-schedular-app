import streamlit as st
import pandas as pd
import os
from pathlib import Path

def home_page():
    """
    Render the combined home and about page
    """
    st.title("Delivery Route Optimization")
    
    st.write("""
    Welcome to the Delivery Route Optimization application! This tool helps logistics teams
    optimize delivery routes for a fleet of vehicles while considering constraints such as delivery time windows,
    vehicle capacity, and traffic conditions.
    
    Use the navigation sidebar to explore different features of this application.
    """)
    
    # Quick stats from data at the top
    try:
        # Get data paths
        root_dir = Path(__file__).resolve().parent.parent.parent
        delivery_path = os.path.join(root_dir, 'data', 'delivery-data', 'delivery_data.csv')
        vehicle_path = os.path.join(root_dir, 'data', 'vehicle-data', 'vehicle_data.csv')
        
        if os.path.exists(delivery_path) and os.path.exists(vehicle_path):
            # Load data for stats
            delivery_data = pd.read_csv(delivery_path)
            vehicle_data = pd.read_csv(vehicle_path)
            
            # Display stats
            st.subheader("Current Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Deliveries", len(delivery_data))
            with col2:
                st.metric("Total Vehicles", len(vehicle_data))
            with col3:
                pending = delivery_data[delivery_data['status'] == 'Pending'] if 'status' in delivery_data.columns else []
                st.metric("Pending Deliveries", len(pending))
                
            # Add more detailed stats in an expander
            with st.expander("View More Statistics"):
                # Status breakdown
                if 'status' in delivery_data.columns:
                    st.write("#### Delivery Status Breakdown")
                    status_counts = delivery_data['status'].value_counts().reset_index()
                    status_counts.columns = ['Status', 'Count']
                    status_chart = st.bar_chart(status_counts.set_index('Status'))
                
                # Priority breakdown
                if 'priority' in delivery_data.columns:
                    st.write("#### Delivery Priority Breakdown")
                    priority_counts = delivery_data['priority'].value_counts().reset_index()
                    priority_counts.columns = ['Priority', 'Count']
                    priority_chart = st.bar_chart(priority_counts.set_index('Priority'))
        else:
            st.info("Please generate data first to see statistics")
            st.code("python src/utils/generate_all_data.py")
    except Exception as e:
        st.info("Generate data first to see statistics")
        st.code("python src/utils/generate_all_data.py")
    
    # Add the image
    img_path = Path(__file__).resolve().parent.parent.parent / "img" / "delivery-route-network.jpg"
    if os.path.exists(img_path):
        st.image(str(img_path), caption="Delivery Route Network")


# Make sure the function can be executed standalone
if __name__ == "__main__":
    st.set_page_config(page_title="Home - Delivery Route Optimization", page_icon="🚚", layout="wide")
    home_page()