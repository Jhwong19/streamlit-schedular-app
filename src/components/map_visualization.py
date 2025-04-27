def load_data():
    """
    Load delivery and vehicle data from CSV files
    
    Returns:
        tuple: (delivery_data, vehicle_data)
    """
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    
    # Define data paths
    delivery_data_path = os.path.join(root_dir, 'data', 'delivery-data', 'delivery_data.csv')
    vehicle_data_path = os.path.join(root_dir, 'data', 'vehicle-data', 'vehicle_data.csv')
    
    # Load data
    try:
        delivery_data = pd.read_csv(delivery_data_path)
        vehicle_data = pd.read_csv(vehicle_data_path)
        return delivery_data, vehicle_data
    except FileNotFoundError as e:
        st.error(f"Could not load data: {e}")
        st.info("Please generate the data first by running: python src/utils/generate_all_data.py")
        return None, None

def create_delivery_map(delivery_data=None, vehicle_data=None, show_deliveries=True, show_depots=True, 
                       date_filter=None, status_filter=None, priority_filter=None):
    """
    Create a Folium map with markers for deliveries and vehicle depots
    
    Parameters:
        delivery_data (pd.DataFrame): Delivery data
        vehicle_data (pd.DataFrame): Vehicle data
        show_deliveries (bool): Whether to show delivery markers
        show_depots (bool): Whether to show depot markers
        date_filter (str): Filter deliveries by date
        status_filter (str): Filter deliveries by status
        priority_filter (str): Filter deliveries by priority
    
    Returns:
        folium.Map: Folium map with markers
    """
    # If data not provided, load it
    if delivery_data is None or vehicle_data is None:
        delivery_data, vehicle_data = load_data()
        if delivery_data is None or vehicle_data is None:
            return None
    
    # Apply filters to delivery data
    if date_filter is not None:
        delivery_data = delivery_data[delivery_data['delivery_date'] == date_filter]
        
    if status_filter is not None:
        delivery_data = delivery_data[delivery_data['status'] == status_filter]
        
    if priority_filter is not None:
        delivery_data = delivery_data[delivery_data['priority'] == priority_filter]
    
    # Create map centered around Singapore
    singapore_coords = [1.3521, 103.8198]  # Center of Singapore
    m = folium.Map(location=singapore_coords, zoom_start=12)
    
    # Add delivery markers
    if show_deliveries and not delivery_data.empty:
        for _, row in delivery_data.iterrows():
            # Create popup content with delivery information
            popup_content = f"""
                <b>Delivery ID:</b> {row['delivery_id']}<br>
                <b>Customer:</b> {row['customer_name']}<br>
                <b>Address:</b> {row['address']}<br>
                <b>Time Window:</b> {row['time_window']}<br>
                <b>Status:</b> {row['status']}<br>
                <b>Priority:</b> {row['priority']}<br>
                <b>Weight:</b> {row['weight_kg']} kg<br>
                <b>Volume:</b> {row['volume_m3']} m³
            """
            
            # Set marker color based on priority
            color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'blue'}
            color = color_map.get(row['priority'], 'blue')
            
            # Add marker to map
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Delivery {row['delivery_id']}: {row['customer_name']}",
                icon=folium.Icon(color=color, icon="package", prefix="fa")
            ).add_to(m)
    
    # Add depot markers
    if show_depots and not vehicle_data.empty:
        for _, row in vehicle_data.iterrows():
            # Create popup content with vehicle information
            popup_content = f"""
                <b>Vehicle ID:</b> {row['vehicle_id']}<br>
                <b>Type:</b> {row['vehicle_type']}<br>
                <b>Driver:</b> {row['driver_name']}<br>
                <b>Status:</b> {row['status']}<br>
                <b>Capacity:</b> {row['max_weight_kg']} kg / {row['max_volume_m3']} m³<br>
                <b>Working Hours:</b> {row['start_time']} - {row['end_time']}
            """
            
            # Add marker to map
            folium.Marker(
                location=[row['depot_latitude'], row['depot_longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Depot: {row['vehicle_id']}",
                icon=folium.Icon(color="green", icon="truck", prefix="fa")
            ).add_to(m)
    
    return m

def display_map_component():
    """
    Display the map visualization component in Streamlit
    """
    st.subheader("Delivery and Depot Locations")
    
    # Load data
    delivery_data, vehicle_data = load_data()
    if delivery_data is None or vehicle_data is None:
        return
    
    # Create sidebar filters
    with st.sidebar:
        st.subheader("Map Filters")
        
        # Show/hide options
        show_deliveries = st.checkbox("Show Deliveries", value=True)
        show_depots = st.checkbox("Show Depots", value=True)
        
        # Delivery date filter
        dates = sorted(delivery_data['delivery_date'].unique())
        selected_date = st.selectbox(
            "Filter by Date",
            options=["All"] + list(dates),
            index=0
        )
        date_filter = None if selected_date == "All" else selected_date
        
        # Delivery status filter
        statuses = sorted(delivery_data['status'].unique())
        selected_status = st.selectbox(
            "Filter by Status",
            options=["All"] + list(statuses),
            index=0
        )
        status_filter = None if selected_status == "All" else selected_status
        
        # Delivery priority filter
        priorities = sorted(delivery_data['priority'].unique())
        selected_priority = st.selectbox(
            "Filter by Priority",
            options=["All"] + list(priorities),
            index=0
        )
        priority_filter = None if selected_priority == "All" else selected_priority
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    # Apply filters for stats calculation
    filtered_delivery_data = delivery_data
    if date_filter:
        filtered_delivery_data = filtered_delivery_data[filtered_delivery_data['delivery_date'] == date_filter]
    if status_filter:
        filtered_delivery_data = filtered_delivery_data[filtered_delivery_data['status'] == status_filter]
    if priority_filter:
        filtered_delivery_data = filtered_delivery_data[filtered_delivery_data['priority'] == priority_filter]
    
    with col1:
        st.metric("Total Deliveries", filtered_delivery_data.shape[0])
    
    with col2:
        st.metric("Total Weight", f"{filtered_delivery_data['weight_kg'].sum():.2f} kg")
    
    with col3:
        st.metric("Available Vehicles", vehicle_data[vehicle_data['status'] == 'Available'].shape[0])
    
    # Create and display the map
    delivery_map = create_delivery_map(
        delivery_data=delivery_data,
        vehicle_data=vehicle_data,
        show_deliveries=show_deliveries,
        show_depots=show_depots,
        date_filter=date_filter,
        status_filter=status_filter,
        priority_filter=priority_filter
    )
    
    if delivery_map:
        folium_static(delivery_map, width=800, height=600)
    else:
        st.error("Could not create map. Please check that data is available.")

if __name__ == "__main__":
    # Run the map visualization component
    display_map_component()