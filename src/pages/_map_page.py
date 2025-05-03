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


    # Add help section with expander
    with st.expander("📚 How to Use the Map Pageß"):
        st.markdown("""
        ## Step-by-Step Guide to the Map Page
        
        The Map page provides an interactive visualization of all delivery locations and vehicle depots. It helps you understand delivery distribution, monitor delivery status, and plan logistics operations.
        
        ### 1. Map Navigation
        
        - **Pan**: Click and drag to move around the map
        - **Zoom**: Use the scroll wheel or the +/- buttons in the top-left corner
        - **View Details**: Click on any marker to see detailed information about that delivery or depot
        
        ### 2. Using Map Filters (Sidebar)
        
        - **Show/Hide Elements**:
          - Toggle "Show Deliveries" to display or hide delivery markers
          - Toggle "Show Depots" to display or hide vehicle depot markers
          - Enable "Show Data Table" to view raw delivery data below the map
          - Enable "Show Calendar View" to see delivery schedules organized by date
        
        - **Filter by Attributes**:
          - Use "Filter by Priority" to show only deliveries of selected priority levels (High, Medium, Low)
          - Use "Filter by Status" to show only deliveries with selected statuses (Pending, In Transit, Delivered)
        
        - **Date Filtering**:
          - Use the "Date Range" selector to focus on deliveries within specific dates
          - This affects both the map display and the calendar view
        
        ### 3. Understanding the Map Markers
        
        - **Delivery Markers**:
          - Red markers: High priority deliveries
          - Orange markers: Medium priority deliveries
          - Blue markers: Low priority deliveries
        
        - **Depot Markers**:
          - Green house icons: Vehicle depot locations
        
        ### 4. Using the Calendar View
        
        - Select specific dates from the dropdown to view scheduled deliveries
        - Each tab shows deliveries for one selected date
        - Timeline bars are color-coded by priority (red=High, orange=Medium, blue=Low)
        - Hover over timeline bars to see detailed delivery information
        - Check the summary metrics below each calendar for quick insights
        
        ### 5. Reading the Delivery Statistics
        
        - The top section shows key metrics about displayed deliveries:
          - Total number of deliveries shown
          - Total weight of all displayed deliveries
          - Number of pending deliveries
          - Breakdown of deliveries by status
        
        ### 6. Data Table Features
        
        When "Show Data Table" is enabled:
        - Green highlighted rows: Completed deliveries
        - Red highlighted rows: Urgent high-priority deliveries due within the next week
        - Sort any column by clicking the column header
        - Search across all fields using the search box
        
        This map view helps you visualize your delivery operations geographically while the calendar provides a time-based perspective of your delivery schedule.
        """)
    
    # Initialize session state variables for filters
    if 'map_filters' not in st.session_state:
        st.session_state.map_filters = {
            'selected_dates': ["All"],
            'priority_filter': [],
            'status_filter': [],
            'date_range': [None, None],
            'show_calendar': True,
            'show_map': True,
            'show_data_table': False,
            'cluster_markers': True
        }
    
    # Create filters in sidebar
    with st.sidebar:
        st.header("Map Filters")
        
        # Show/hide options - use session state values as defaults
        show_deliveries = st.checkbox(
            "Show Deliveries", 
            value=st.session_state.map_filters.get('show_deliveries', True),
            key="show_deliveries_checkbox"
        )
        st.session_state.map_filters['show_deliveries'] = show_deliveries
        
        show_depots = st.checkbox(
            "Show Depots", 
            value=st.session_state.map_filters.get('show_depots', True),
            key="show_depots_checkbox"
        )
        st.session_state.map_filters['show_depots'] = show_depots
        
        # Show/hide data table
        show_data_table = st.checkbox(
            "Show Data Table", 
            value=st.session_state.map_filters.get('show_data_table', False),
            key="show_data_table_checkbox"
        )
        st.session_state.map_filters['show_data_table'] = show_data_table
        
        # Choose visualization tabs
        show_calendar = st.checkbox(
            "Show Calendar View", 
            value=st.session_state.map_filters.get('show_calendar', True),
            key="show_calendar_checkbox"
        )
        st.session_state.map_filters['show_calendar'] = show_calendar

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
                    default=st.session_state.map_filters.get('priority_filter', all_priorities),
                    key="priority_multiselect"
                )
                st.session_state.map_filters['priority_filter'] = selected_priorities
                
                if selected_priorities:
                    delivery_data = delivery_data[delivery_data['priority'].isin(selected_priorities)]
        
        if 'status' in delivery_data.columns:
            with st.sidebar:
                all_statuses = sorted(delivery_data['status'].unique().tolist())
                selected_statuses = st.multiselect(
                    "Filter by Status",
                    options=all_statuses,
                    default=st.session_state.map_filters.get('status_filter', all_statuses),
                    key="status_multiselect"
                )
                st.session_state.map_filters['status_filter'] = selected_statuses
                
                if selected_statuses:
                    delivery_data = delivery_data[delivery_data['status'].isin(selected_statuses)]
        
        if 'delivery_date' in delivery_data.columns:
            with st.sidebar:
                # Get the min/max dates from the ORIGINAL unfiltered data
                # Load original data to get proper date range
                original_data = pd.read_csv(delivery_path)
                if 'delivery_date' in original_data.columns:
                    original_data['delivery_date'] = pd.to_datetime(original_data['delivery_date'])
                    
                min_date = original_data['delivery_date'].min().date()
                max_date = original_data['delivery_date'].max().date()
                
                # Get saved values from session state
                saved_start_date = st.session_state.map_filters.get('date_range', [None, None])[0]
                saved_end_date = st.session_state.map_filters.get('date_range', [None, None])[1]
                
                # Validate saved dates - ensure they're within allowed range
                if saved_start_date and saved_start_date < min_date:
                    saved_start_date = min_date
                if saved_end_date and saved_end_date > max_date:
                    saved_end_date = max_date
                    
                # Set default values with proper validation
                default_start_date = saved_start_date if saved_start_date else min_date
                default_end_date = saved_end_date if saved_end_date else min(min_date + timedelta(days=7), max_date)
                
                # Add date range picker
                try:
                    date_range = st.date_input(
                        "Date Range",
                        value=(default_start_date, default_end_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="date_range_input"
                    )
                    
                    # Update session state with new date range
                    if len(date_range) == 2:
                        st.session_state.map_filters['date_range'] = list(date_range)
                        start_date, end_date = date_range
                        mask = (delivery_data['delivery_date'].dt.date >= start_date) & (delivery_data['delivery_date'].dt.date <= end_date)
                        delivery_data = delivery_data[mask]
                except Exception as e:
                    # If there's any error with the date range, reset it
                    st.error(f"Error with date range: {e}")
                    st.session_state.map_filters['date_range'] = [min_date, max_date]
                    date_range = (min_date, max_date)
                    mask = (delivery_data['delivery_date'].dt.date >= min_date) & (delivery_data['delivery_date'].dt.date <= max_date)
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
            
            # Get all available dates and add ƒmulti-select filter
            all_dates = sorted(calendar_data['delivery_date'].dt.date.unique())

            # Format dates for display in the dropdown
            date_options = {date.strftime('%b %d, %Y'): date for date in all_dates}

            # Get default selection from session state
            default_selections = st.session_state.map_filters.get('calendar_selected_dates', [])

            # Validate default selections - only keep dates that exist in current options
            valid_default_selections = [date_str for date_str in default_selections if date_str in date_options.keys()]

            # If no valid selections remain, default to first date (if available)
            if not valid_default_selections and date_options:
                valid_default_selections = [list(date_options.keys())[0]]

            # Add multiselect for date filtering with validated defaults
            selected_date_strings = st.multiselect(
                "Select dates to display",
                options=list(date_options.keys()),
                default=valid_default_selections,
                key="calendar_date_selector"
            )

            # Save selections to session state
            st.session_state.map_filters['calendar_selected_dates'] = selected_date_strings

            # Convert selected strings back to date objects
            selected_dates = [date_options[date_str] for date_str in selected_date_strings]
            
            if not selected_dates:
                st.info("Please select at least one date to view the delivery schedule.")
            else:
                # Filter calendar data to only include selected dates
                filtered_calendar = calendar_data[calendar_data['delivery_date'].dt.date.isin(selected_dates)]
                
                # Group tasks by date for better visualization
                date_groups = filtered_calendar.groupby(filtered_calendar['delivery_date'].dt.date)
                
                # Create tabs only for the selected dates
                date_tabs = st.tabs([date.strftime('%b %d, %Y') for date in selected_dates])
                
                for i, (date, tab) in enumerate(zip(selected_dates, date_tabs)):
                    with tab:
                        # Filter data for this date
                        day_data = filtered_calendar[filtered_calendar['delivery_date'].dt.date == date]
                        
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