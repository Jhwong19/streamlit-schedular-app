import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import os
import plotly.figure_factory as ff
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

def map_page():
    """
    Render the map visualization page with delivery and depot locations.
    Can be called from app.py to display within the main application.
    """
    st.title("Delivery Route Map")
    st.write("""
    This page visualizes the delivery locations and vehicle depots on an interactive map.
    Use the filters in the sidebar to customize the view.
    """)

    # Create filters in sidebar
    with st.sidebar:
        st.header("Map Filters")
        
        # Show/hide options
        show_deliveries = st.checkbox("Show Deliveries", value=True)
        show_depots = st.checkbox("Show Depots", value=True)
        
        # Show/hide data table
        show_data_table = st.checkbox("Show Data Table", value=False)
        
        # Choose visualization tabs
        show_calendar = st.checkbox("Show Calendar View", value=True)

    # Try to load data
    try:
        # Get data paths
        root_dir = Path(__file__).resolve().parent.parent.parent  # Go up to project root level
        delivery_path = os.path.join(root_dir, 'data', 'delivery-data', 'delivery_data.csv')  # Fixed directory name with underscore
        vehicle_path = os.path.join(root_dir, 'data', 'vehicle-data', 'vehicle_data.csv')    # Fixed directory name with underscore
        
        # Check if files exist
        if not os.path.exists(delivery_path):
            # Try with hyphen instead of underscore
            delivery_path = os.path.join(root_dir, 'data', 'delivery-data', 'delivery_data.csv')
            if not os.path.exists(delivery_path):
                st.warning(f"Delivery data file not found at: {delivery_path}")
                st.info("Please generate data first with: python src/utils/generate_all_data.py")
                return
            
        if not os.path.exists(vehicle_path):
            # Try with hyphen instead of underscore
            vehicle_path = os.path.join(root_dir, 'data', 'vehicle-data', 'vehicle_data.csv')
            if not os.path.exists(vehicle_path):
                st.warning(f"Vehicle data file not found at: {vehicle_path}")
                st.info("Please generate data first with: python src/utils/generate_all_data.py")
                return
            
        # Load data
        delivery_data = pd.read_csv(delivery_path)
        vehicle_data = pd.read_csv(vehicle_path)
        
        # Ensure delivery_date is properly formatted as datetime
        if 'delivery_date' in delivery_data.columns:
            delivery_data['delivery_date'] = pd.to_datetime(delivery_data['delivery_date'])
        
        # Add more filters if data is available - CONVERT TO MULTI-SELECT
        if 'priority' in delivery_data.columns:
            with st.sidebar:
                all_priorities = sorted(delivery_data['priority'].unique().tolist())
                selected_priorities = st.multiselect(
                    "Filter by Priority",
                    options=all_priorities,
                    default=all_priorities  # Default to all selected
                )
                
                if selected_priorities:
                    delivery_data = delivery_data[delivery_data['priority'].isin(selected_priorities)]
        
        if 'status' in delivery_data.columns:
            with st.sidebar:
                all_statuses = sorted(delivery_data['status'].unique().tolist())
                selected_statuses = st.multiselect(
                    "Filter by Status",
                    options=all_statuses,
                    default=all_statuses  # Default to all selected
                )
                
                if selected_statuses:
                    delivery_data = delivery_data[delivery_data['status'].isin(selected_statuses)]
        
        if 'delivery_date' in delivery_data.columns:
            with st.sidebar:
                min_date = delivery_data['delivery_date'].min().date()
                max_date = delivery_data['delivery_date'].max().date()
                
                # Calculate a default end date that doesn't exceed max_date
                default_end_date = min(min_date + timedelta(days=7), max_date)
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, default_end_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (delivery_data['delivery_date'].dt.date >= start_date) & (delivery_data['delivery_date'].dt.date <= end_date)
                    delivery_data = delivery_data[mask]
                    
        # MOVED STATISTICS TO THE TOP
        st.subheader("Delivery Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Deliveries Shown", len(delivery_data))
            
        with col2:
            if 'weight_kg' in delivery_data.columns:
                total_weight = delivery_data['weight_kg'].sum()
                st.metric("Total Weight", f"{total_weight:.2f} kg")
                
        with col3:
            if 'status' in delivery_data.columns:
                pending = len(delivery_data[delivery_data['status'] == 'Pending'])
                st.metric("Pending Deliveries", pending)
        
        # Status count columns - dynamic based on available statuses
        if 'status' in delivery_data.columns:
            status_counts = delivery_data['status'].value_counts()
            # Create a varying number of columns based on unique statuses
            status_cols = st.columns(len(status_counts))
            
            for i, (status, count) in enumerate(status_counts.items()):
                with status_cols[i]:
                    # Choose color based on status
                    delta_color = "normal"
                    if status == "Delivered":
                        delta_color = "off" 
                    elif status == "In Transit":
                        delta_color = "normal"
                    elif status == "Pending":
                        delta_color = "inverse"  # Red
                    
                    # Calculate percentage
                    percentage = round((count / len(delivery_data)) * 100, 1)
                    st.metric(
                        f"{status}", 
                        count,
                        f"{percentage}% of total",
                        delta_color=delta_color
                    )
        
        # Create map
        singapore_coords = [1.3521, 103.8198]  # Center of Singapore
        m = folium.Map(location=singapore_coords, zoom_start=12)
        
        # Add delivery markers
        if show_deliveries:
            for _, row in delivery_data.iterrows():
                # Create popup content
                popup_content = f"<b>ID:</b> {row['delivery_id']}<br>"
                
                if 'customer_name' in row:
                    popup_content += f"<b>Customer:</b> {row['customer_name']}<br>"
                
                if 'address' in row:
                    popup_content += f"<b>Address:</b> {row['address']}<br>"
                    
                if 'time_window' in row:
                    popup_content += f"<b>Time Window:</b> {row['time_window']}<br>"
                    
                if 'priority' in row:
                    popup_content += f"<b>Priority:</b> {row['priority']}<br>"
                    
                if 'delivery_date' in row:
                    popup_content += f"<b>Date:</b> {row['delivery_date'].strftime('%b %d, %Y')}<br>"
                
                if 'status' in row:
                    popup_content += f"<b>Status:</b> {row['status']}<br>"
            
                # Choose marker color based on priority
                color = 'blue'
                if 'priority' in row:
                    if row['priority'] == 'High':
                        color = 'red'
                    elif row['priority'] == 'Medium':
                        color = 'orange'
            
                # Add marker to map
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Delivery {row['delivery_id']}",
                    icon=folium.Icon(color=color)
                ).add_to(m)
        
        # Add depot markers
        if show_depots:
            for _, row in vehicle_data.iterrows():
                # Create popup content
                popup_content = f"<b>Vehicle ID:</b> {row['vehicle_id']}<br>"
                
                if 'vehicle_type' in row:
                    popup_content += f"<b>Type:</b> {row['vehicle_type']}<br>"
                    
                if 'driver_name' in row:
                    popup_content += f"<b>Driver:</b> {row['driver_name']}<br>"
            
                # Add marker to map
                folium.Marker(
                    [row['depot_latitude'], row['depot_longitude']],
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Depot: {row['vehicle_id']}",
                    icon=folium.Icon(color='green', icon='home', prefix='fa')
                ).add_to(m)
        
        # Display the map
        folium_static(m, width=800, height=500)
        
        # Display calendar visualization if selected
        if show_calendar and 'delivery_date' in delivery_data.columns and 'time_window' in delivery_data.columns:
            st.subheader("Delivery Schedule Calendar")
            
            # Process data for calendar view
            calendar_data = delivery_data.copy()
            
            # Extract start and end times from time_window
            calendar_data[['start_time', 'end_time']] = calendar_data['time_window'].str.split('-', expand=True)
            
            # Create start and end datetime for each delivery
            calendar_data['Start'] = pd.to_datetime(
                calendar_data['delivery_date'].dt.strftime('%Y-%m-%d') + ' ' + calendar_data['start_time']
            )
            calendar_data['Finish'] = pd.to_datetime(
                calendar_data['delivery_date'].dt.strftime('%Y-%m-%d') + ' ' + calendar_data['end_time']
            )
            
            # Create task column for Gantt chart
            calendar_data['Task'] = calendar_data['delivery_id'] + ': ' + calendar_data['customer_name']
            
            # Create color mapping for priority
            if 'priority' in calendar_data.columns:
                color_map = {'High': 'rgb(255, 0, 0)', 'Medium': 'rgb(255, 165, 0)', 'Low': 'rgb(0, 0, 255)'}
                calendar_data['Color'] = calendar_data['priority'].map(color_map)
            else:
                calendar_data['Color'] = 'rgb(0, 0, 255)'  # Default blue
            
            # Group tasks by date for better visualization
            date_groups = calendar_data.groupby(calendar_data['delivery_date'].dt.date)
            
            # Create tabs for each date - UPDATED FORMAT TO MMM DD, YYYY
            date_tabs = st.tabs([date.strftime('%b %d, %Y') for date in sorted(date_groups.groups.keys())])
            
            for i, (date, tab) in enumerate(zip(sorted(date_groups.groups.keys()), date_tabs)):
                with tab:
                    # Filter data for this date
                    day_data = calendar_data[calendar_data['delivery_date'].dt.date == date]
                    
                    if len(day_data) > 0:
                        # Create figure
                        fig = px.timeline(
                            day_data, 
                            x_start="Start", 
                            x_end="Finish", 
                            y="Task",
                            color="priority" if 'priority' in day_data.columns else None,
                            color_discrete_map={"High": "red", "Medium": "orange", "Low": "blue"},
                            hover_data=["customer_name", "address", "weight_kg", "status"]
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Deliveries scheduled for {date.strftime('%b %d, %Y')}",
                            xaxis_title="Time of Day",
                            yaxis_title="Delivery",
                            height=max(300, 50 * len(day_data)),
                            yaxis={'categoryorder':'category ascending'}
                        )
                        
                        # Display figure
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Deliveries", len(day_data))
                        with col2:
                            if 'weight_kg' in day_data.columns:
                                st.metric("Total Weight", f"{day_data['weight_kg'].sum():.2f} kg")
                        with col3:
                            if 'priority' in day_data.columns and 'High' in day_data['priority'].values:
                                st.metric("High Priority", len(day_data[day_data['priority'] == 'High']))
                        
                        # NEW - Add delivery status breakdown for this day
                        if 'status' in day_data.columns:
                            st.write("##### Deliveries by Status")
                            status_counts = day_data['status'].value_counts()
                            status_cols = st.columns(min(4, len(status_counts)))
                            
                            for i, (status, count) in enumerate(status_counts.items()):
                                col_idx = i % len(status_cols)
                                with status_cols[col_idx]:
                                    st.metric(status, count)
                    else:
                        st.info(f"No deliveries scheduled for {date.strftime('%b %d, %Y')}")
        
        # Display raw data table if selected
        if show_data_table:
            st.subheader("Delivery Data")
            
            # Create a copy for display
            display_df = delivery_data.copy()
            
            # Convert delivery_date back to string for display
            if 'delivery_date' in display_df.columns:
                display_df['delivery_date'] = display_df['delivery_date'].dt.strftime('%b %d, %Y')
            
            # Compute which deliveries are urgent (next 7 days)
            if 'delivery_date' in delivery_data.columns:
                today = datetime.now().date()
                next_week = today + timedelta(days=7)
                
                # Function to highlight rows based on delivery status and urgency
                def highlight_rows(row):
                    delivery_date = pd.to_datetime(row['delivery_date']).date() if 'delivery_date' in row else None
                    
                    # Check status first - highlight delivered rows in green
                    if 'status' in row and row['status'] == 'Delivered':
                        return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
                    # Then check for urgent high-priority deliveries - highlight in red
                    elif delivery_date and delivery_date <= next_week and delivery_date >= today and row['priority'] == 'High':
                        return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
                    else:
                        return [''] * len(row)
                
                # Display styled dataframe
                st.dataframe(display_df.style.apply(highlight_rows, axis=1))
            else:
                st.dataframe(display_df)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please generate the data first by running: python src/utils/generate_all_data.py")
        st.write("Error details:", e)  # Detailed error for debugging

# Make the function executable when file is run directly
if __name__ == "__main__":
    # This is for debugging/testing the function independently
    st.set_page_config(page_title="Map View - Delivery Route Optimization", page_icon="🗺️", layout="wide")
    map_page()