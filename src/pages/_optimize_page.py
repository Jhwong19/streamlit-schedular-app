import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import os
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random
import time
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium.plugins
from folium.features import DivIcon
import requests
import plotly.express as px

def optimize_page():
    """
    Render the optimization page with controls for route optimization
    """
    st.title("Delivery Route Optimization")
    
    # Load data
    data = load_all_data()
    if not data:
        return
    
    delivery_data, vehicle_data, distance_matrix, time_matrix, locations = data
    
    # Optimization parameters
    st.sidebar.header("Optimization Parameters")
    
    # Date selection for deliveries
    if 'delivery_date' in delivery_data.columns:
        available_dates = sorted(delivery_data['delivery_date'].unique())
        date_options = ["All"] + list(available_dates)
        selected_dates = st.sidebar.multiselect(
            "Select Delivery Dates",
            options=date_options,
            default=["All"]  # Default to "All"
        )
        
        # Handle filtering based on selection
        if "All" not in selected_dates:
            if selected_dates:  # If specific dates were selected
                delivery_data = delivery_data[delivery_data['delivery_date'].isin(selected_dates)]
            elif available_dates:  # No dates selected, show warning
                st.sidebar.warning("No dates selected. Please select at least one delivery date.")
                return
        # If "All" is selected, keep all dates - no filtering needed
    
    # Priority weighting
    priority_weight = st.sidebar.slider(
        "Priority Importance",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="Higher values give more importance to high-priority deliveries"
    )
    
    # Time window importance
    time_window_weight = st.sidebar.slider(
        "Time Window Importance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Higher values enforce stricter adherence to delivery time windows"
    )
    
    # Distance vs load balancing
    balance_weight = st.sidebar.slider(
        "Load Balancing vs Distance",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        help="Higher values prioritize even distribution of deliveries across vehicles over total distance"
    )
    
    # Max vehicles to use
    available_vehicles = vehicle_data[vehicle_data['status'] == 'Available']
    max_vehicles = st.sidebar.slider(
        "Maximum Vehicles to Use",
        min_value=1,
        max_value=len(available_vehicles),
        value=min(5, len(available_vehicles))
    )
    
    # Main optimization section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Delivery Route Optimizer")
        st.write(f"Optimizing routes for {len(delivery_data)} deliveries using up to {max_vehicles} vehicles")
        
        # Statistics
        st.write("#### Delivery Statistics")
        st.write(f"Total Deliveries: {len(delivery_data)}")
        
        if 'priority' in delivery_data.columns:
            priority_counts = delivery_data['priority'].value_counts()
        
        if 'weight_kg' in delivery_data.columns:
            total_weight = delivery_data['weight_kg'].sum()
            st.write(f"Total Weight: {total_weight:.2f} kg")
    
    with col2:
        st.write("#### Vehicle Availability")
        st.write(f"Available Vehicles: {len(available_vehicles)}")
        
        # Show vehicle capacity
        if 'max_weight_kg' in vehicle_data.columns:
            total_capacity = available_vehicles['max_weight_kg'].sum()
            st.write(f"Total Capacity: {total_capacity:.2f} kg")
            
            # Check if we have enough capacity
            if 'weight_kg' in delivery_data.columns:
                if total_capacity < total_weight:
                    st.warning("⚠️ Insufficient vehicle capacity for all deliveries")
                else:
                    st.success("✅ Sufficient vehicle capacity")
    
    # Run optimization button
    if st.button("Generate Optimal Routes"):
        with st.spinner("Calculating optimal routes..."):
            start_time = time.time()
            
            # Prepare data for optimization
            optimization_result = run_optimization(
                delivery_data=delivery_data,
                vehicle_data=available_vehicles.iloc[:max_vehicles],
                distance_matrix=distance_matrix,
                time_matrix=time_matrix,
                locations=locations,
                priority_weight=priority_weight,
                time_window_weight=time_window_weight,
                balance_weight=balance_weight
            )
            
            end_time = time.time()
            st.success(f"Optimization completed in {end_time - start_time:.2f} seconds")
            
            # Display results
            display_optimization_results(
                optimization_result=optimization_result,
                delivery_data=delivery_data,
                vehicle_data=available_vehicles.iloc[:max_vehicles],
                distance_matrix=distance_matrix,
                time_matrix=time_matrix,
                locations=locations
            )

def load_all_data():
    """
    Load all necessary data for optimization
    
    Returns:
        tuple of (delivery_data, vehicle_data, distance_matrix, time_matrix, locations)
    """
    # Get data paths
    root_dir = Path(__file__).resolve().parent.parent.parent
    delivery_path = os.path.join(root_dir, 'data', 'delivery-data', 'delivery_data.csv')
    vehicle_path = os.path.join(root_dir, 'data', 'vehicle-data', 'vehicle_data.csv')
    distance_matrix_path = os.path.join(root_dir, 'data', 'time-matrix', 'distance_matrix.csv')
    time_matrix_path = os.path.join(root_dir, 'data', 'time-matrix', 'base_time_matrix.csv')
    locations_path = os.path.join(root_dir, 'data', 'time-matrix', 'locations.csv')
    
    # Check if files exist
    missing_files = []
    for path, name in [
        (delivery_path, "delivery data"),
        (vehicle_path, "vehicle data"),
        (distance_matrix_path, "distance matrix"),
        (time_matrix_path, "time matrix"),
        (locations_path, "locations data")
    ]:
        if not os.path.exists(path):
            missing_files.append(name)
    
    if missing_files:
        st.error(f"Missing required data: {', '.join(missing_files)}")
        st.info("Please generate all data first by running: python src/utils/generate_all_data.py")
        return None
    
    # Load data
    delivery_data = pd.read_csv(delivery_path)
    vehicle_data = pd.read_csv(vehicle_path)
    distance_matrix = pd.read_csv(distance_matrix_path, index_col=0)
    time_matrix = pd.read_csv(time_matrix_path, index_col=0)
    locations = pd.read_csv(locations_path)
    
    return delivery_data, vehicle_data, distance_matrix, time_matrix, locations

def run_optimization(delivery_data, vehicle_data, distance_matrix, time_matrix, locations, 
                    priority_weight, time_window_weight, balance_weight):
    """
    Run the route optimization algorithm using Google OR-Tools
    
    Parameters:
        delivery_data (pd.DataFrame): DataFrame containing delivery information
        vehicle_data (pd.DataFrame): DataFrame containing vehicle information
        distance_matrix (pd.DataFrame): Distance matrix between locations
        time_matrix (pd.DataFrame): Time matrix between locations
        locations (pd.DataFrame): DataFrame with location details
        priority_weight (float): Weight for delivery priority in optimization
        time_window_weight (float): Weight for time window adherence
        balance_weight (float): Weight for balancing load across vehicles
        
    Returns:
        dict: Optimization results
    """
    st.write("Setting up optimization model with OR-Tools...")
    
    # Extract required data for optimization
    num_vehicles = len(vehicle_data)
    num_deliveries = len(delivery_data)
    
    # Create a list of all locations (depots + delivery points)
    all_locations = []
    delivery_locations = []
    depot_locations = []
    vehicle_capacities = []
    
    # First, add depot locations (one per vehicle)
    for i, (_, vehicle) in enumerate(vehicle_data.iterrows()):
        depot_loc = {
            'id': vehicle['vehicle_id'],
            'type': 'depot',
            'index': i,  # Important for mapping to OR-Tools indices
            'latitude': vehicle['depot_latitude'],
            'longitude': vehicle['depot_longitude'],
            'vehicle_index': i
        }
        depot_locations.append(depot_loc)
        all_locations.append(depot_loc)
        
        # Add vehicle capacity
        if 'max_weight_kg' in vehicle:
            vehicle_capacities.append(int(vehicle['max_weight_kg'] * 100))  # Convert to integers (OR-Tools works better with integers)
        else:
            vehicle_capacities.append(1000)  # Default capacity of 10kg (1000 in scaled units)
    
    # Then add delivery locations
    for i, (_, delivery) in enumerate(delivery_data.iterrows()):
        # Determine priority factor (will be used in the objective function)
        priority_factor = 1.0
        if 'priority' in delivery:
            if delivery['priority'] == 'High':
                priority_factor = 0.5  # Higher priority = lower cost
            elif delivery['priority'] == 'Low':
                priority_factor = 2.0  # Lower priority = higher cost
        
        # Calculate delivery demand (weight)
        demand = int(delivery.get('weight_kg', 1.0) * 100)  # Convert to integers
        
        delivery_loc = {
            'id': delivery['delivery_id'],
            'type': 'delivery',
            'index': num_vehicles + i,  # Important for mapping to OR-Tools indices
            'latitude': delivery['latitude'],
            'longitude': delivery['longitude'],
            'priority': delivery.get('priority', 'Medium'),
            'priority_factor': priority_factor,
            'weight_kg': delivery.get('weight_kg', 1.0),
            'demand': demand,
            'time_window': delivery.get('time_window', '09:00-17:00'),
            'customer_name': delivery.get('customer_name', 'Unknown')
        }
        delivery_locations.append(delivery_loc)
        all_locations.append(delivery_loc)
    
    # Create distance and time matrices for OR-Tools
    # OR-Tools expects a flat list of costs
    dist_matrix = np.zeros((len(all_locations), len(all_locations)))
    time_matrix_mins = np.zeros((len(all_locations), len(all_locations)))
    
    # Use the provided distance_matrix if it's the right size, otherwise compute distances
    if isinstance(distance_matrix, pd.DataFrame) and len(distance_matrix) == len(all_locations):
        # Convert dataframe to numpy array
        dist_matrix = distance_matrix.values
        time_matrix_mins = time_matrix.values
    else:
        # Compute simple Euclidean distances (this is a fallback)
        for i in range(len(all_locations)):
            for j in range(len(all_locations)):
                if i == j:
                    continue
                
                # Approximate distance in km (very rough)
                lat1, lon1 = all_locations[i]['latitude'], all_locations[i]['longitude']
                lat2, lon2 = all_locations[j]['latitude'], all_locations[j]['longitude']
                
                # Simple Euclidean distance (for demo purposes)
                dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111  # Convert to km
                dist_matrix[i, j] = dist
                time_matrix_mins[i, j] = dist * 2  # Rough estimate: 30km/h -> 2 mins per km
    
    # Prepare demand array (0 for depots, actual demand for deliveries)
    demands = [0] * num_vehicles + [d['demand'] for d in delivery_locations]
    
    # OR-Tools setup
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(all_locations),  # Number of nodes (depots + deliveries)
        num_vehicles,        # Number of vehicles
        list(range(num_vehicles)),  # Vehicle start nodes (depot indices)
        list(range(num_vehicles))   # Vehicle end nodes (back to depots)
    )
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Define distance callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node, to_node] * 1000)  # Convert to integers (OR-Tools works better with integers)
    
    # Define time callback
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix_mins[from_node, to_node] * 60)  # Convert minutes to seconds (integers)
    
    # Define demand callback
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands array
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]
    
    # Register callbacks
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    # Set the cost function (distance by default)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity dimension
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,                   # null capacity slack
        vehicle_capacities,  # vehicle maximum capacities
        True,                # start cumul to zero
        'Capacity'
    )
    
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    for i in range(num_vehicles):
        capacity_dimension.SetCumulVarSoftUpperBound(
            routing.End(i), vehicle_capacities[i], 
            int(1000 * (1 - balance_weight))  # Soft penalty for using more capacity
        )
    
    # Add time dimension
    routing.AddDimension(
        time_callback_index,
        30 * 60,       # Allow waiting time of 30 mins
        24 * 60 * 60,  # Maximum time per vehicle (24 hours in seconds)
        False,         # Don't force start cumul to zero
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    # Add time window constraints if time window information is available
    if time_window_weight > 0.01:
        for delivery_idx, delivery in enumerate(delivery_locations):
            if 'time_window' in delivery and delivery['time_window']:
                try:
                    start_time_str, end_time_str = delivery['time_window'].split('-')
                    start_hour, start_min = map(int, start_time_str.split(':'))
                    end_hour, end_min = map(int, end_time_str.split(':'))
                    
                    # Convert to seconds since midnight
                    start_time_sec = (start_hour * 60 + start_min) * 60
                    end_time_sec = (end_hour * 60 + end_min) * 60
                    
                    # Add the time window constraint
                    index = manager.NodeToIndex(num_vehicles + delivery_idx)
                    time_dimension.CumulVar(index).SetRange(start_time_sec, end_time_sec)
                    
                    # Set a penalty for violating time windows based on time_window_weight
                    # Higher weight = higher penalty
                    time_dimension.SetCumulVarSoftUpperBound(
                        index, end_time_sec, 
                        int(100000 * time_window_weight)
                    )
                    time_dimension.SetCumulVarSoftLowerBound(
                        index, start_time_sec, 
                        int(100000 * time_window_weight)
                    )
                except:
                    # Skip if time window format is invalid
                    pass
    
    # Add priority adjustments to the cost function if priority_weight > 0
    if priority_weight > 0.01:
        for delivery_idx, delivery in enumerate(delivery_locations):
            # Apply custom evaluator for delivery nodes
            node_idx = num_vehicles + delivery_idx
            for vehicle_idx in range(num_vehicles):
                # Adjust the cost based on priority factor
                penalty = int(1000 * (1 - priority_weight * (1 - delivery['priority_factor'])))
                routing.AddDisjunction([manager.NodeToIndex(node_idx)], penalty)
    
    # Set parameters for the solver
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # Use path cheapest arc as the first solution strategy
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # Set time limit for optimization
    search_parameters.time_limit.seconds = 5  # Limit to 5 seconds for demo purposes
    
    # Add local search metaheuristics for better solutions
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    
    # Log the search progress
    st.write("Running optimization solver...")
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Extract solution
    optimized_routes = {}
    route_stats = {}
    
    if solution:
        st.write("Solution found!")
        
        for vehicle_idx in range(num_vehicles):
            route = []
            vehicle_id = vehicle_data.iloc[vehicle_idx]['vehicle_id']
            
            # Get the vehicle information
            vehicle_info = {
                'id': vehicle_id,
                'type': vehicle_data.iloc[vehicle_idx].get('vehicle_type', 'Standard'),
                'capacity': vehicle_data.iloc[vehicle_idx].get('max_weight_kg', 1000),
                'depot_latitude': vehicle_data.iloc[vehicle_idx]['depot_latitude'],
                'depot_longitude': vehicle_data.iloc[vehicle_idx]['depot_longitude']
            }
            
            # Get the route for this vehicle
            index = routing.Start(vehicle_idx)
            total_distance = 0
            total_time = 0
            total_load = 0
            
            while not routing.IsEnd(index):
                # Get the node index in the original data
                node_idx = manager.IndexToNode(index)
                
                # Skip depot nodes (they're already at the start)
                if node_idx >= num_vehicles:
                    # This is a delivery node - get the corresponding delivery
                    delivery_idx = node_idx - num_vehicles
                    delivery = delivery_locations[delivery_idx]
                    
                    # Add to route
                    route.append(delivery)
                    
                    # Add to total load
                    total_load += delivery['demand'] / 100  # Convert back to original units
                
                # Move to the next node
                previous_idx = index
                index = solution.Value(routing.NextVar(index))
                
                # Add distance and time from previous to current
                if not routing.IsEnd(index):
                    total_distance += dist_matrix[manager.IndexToNode(previous_idx), manager.IndexToNode(index)]
                    total_time += time_matrix_mins[manager.IndexToNode(previous_idx), manager.IndexToNode(index)]
            
            # Store the route if it's not empty
            if route:
                optimized_routes[vehicle_id] = route
                
                # Store route statistics
                route_stats[vehicle_id] = {
                    'vehicle_type': vehicle_info['type'],
                    'capacity_kg': vehicle_info['capacity'],
                    'deliveries': len(route),
                    'total_distance_km': round(total_distance, 2),
                    'estimated_time_mins': round(total_time),
                    'total_load_kg': round(total_load, 2)
                }
    else:
        st.error("No solution found. Try adjusting the parameters.")
        optimized_routes = {}
        route_stats = {}
    
    return {
        'routes': optimized_routes,
        'stats': route_stats,
        'parameters': {
            'priority_weight': priority_weight,
            'time_window_weight': time_window_weight,
            'balance_weight': balance_weight
        }
    }

def display_optimization_results(optimization_result, delivery_data, vehicle_data, 
                               distance_matrix, time_matrix, locations):
    """
    Display the optimization results
    
    Parameters:
        optimization_result (dict): Result from the optimization algorithm
        delivery_data (pd.DataFrame): Delivery information
        vehicle_data (pd.DataFrame): Vehicle information
        distance_matrix (pd.DataFrame): Distance matrix between locations
        time_matrix (pd.DataFrame): Time matrix between locations
        locations (pd.DataFrame): Location details
    """
    routes = optimization_result['routes']
    
    # Display summary statistics
    st.subheader("Optimization Results")

    # Calculate overall statistics
    total_deliveries = sum(len(route) for route in routes.values())
    active_vehicles = sum(1 for route in routes.values() if len(route) > 0)

    # Calculate additional metrics
    total_distance = sum(stats.get('total_distance_km', 0) for stats in optimization_result.get('stats', {}).values())
    total_time_mins = sum(stats.get('estimated_time_mins', 0) for stats in optimization_result.get('stats', {}).values())

    # Calculate time window compliance
    on_time_deliveries = 0
    total_route_deliveries = 0

    # Count deliveries within time window
    for vehicle_id, route in routes.items():
        stats = optimization_result.get('stats', {}).get(vehicle_id, {})
        
        # Only process if we have stats for this vehicle
        if stats and 'time_window_compliance' in stats:
            on_time_deliveries += stats['time_window_compliance']
        else:
            # Try to estimate based on delivery details
            for delivery in route:
                if 'time_window' in delivery and 'estimated_arrival' in delivery:
                    # Format is typically "HH:MM-HH:MM"
                    try:
                        time_window = delivery['time_window']
                        start_time_str, end_time_str = time_window.split('-')
                        
                        # Convert to minutes for comparison
                        start_mins = int(start_time_str.split(':')[0]) * 60 + int(start_time_str.split(':')[1])
                        end_mins = int(end_time_str.split(':')[0]) * 60 + int(end_time_str.split(':')[1])
                        arrival_mins = delivery.get('estimated_arrival', 0)
                        
                        if start_mins <= arrival_mins <= end_mins:
                            on_time_deliveries += 1
                    except:
                        pass
        
        total_route_deliveries += len(route)

    # Ensure we have a valid number for on-time percentage
    time_window_percent = 0
    if total_route_deliveries > 0:
        time_window_percent = (on_time_deliveries / total_route_deliveries) * 100
        
    # Display metrics in a nicer layout with columns
    st.write("### Overall Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Deliveries Assigned", f"{total_deliveries}/{len(delivery_data)}")
        st.metric("Vehicles Used", f"{active_vehicles}/{len(vehicle_data)}")

    with col2:
        st.metric("Total Distance", f"{total_distance:.1f} km")
        st.metric("Total Time", f"{int(total_time_mins//60)}h {int(total_time_mins%60)}m")

    with col3:
        st.metric("Time Window Compliance", f"{time_window_percent:.0f}%")
        
        # Calculate route efficiency (meters per delivery)
        if total_deliveries > 0:
            efficiency = (total_distance * 1000) / total_deliveries
            st.metric("Avg Distance per Delivery", f"{efficiency:.0f} m")

    # Add a visualization of time distribution
    st.write("### Time & Distance Distribution by Vehicle")
    time_data = {vehicle_id: stats.get('estimated_time_mins', 0) 
                 for vehicle_id, stats in optimization_result.get('stats', {}).items()
                 if len(routes.get(vehicle_id, [])) > 0}

    if time_data:
        # Create bar charts for time and distance
        time_df = pd.DataFrame({
            'Vehicle': list(time_data.keys()),
            'Time (mins)': list(time_data.values())
        })
        
        distance_data = {vehicle_id: stats.get('total_distance_km', 0) 
                         for vehicle_id, stats in optimization_result.get('stats', {}).items()
                         if len(routes.get(vehicle_id, [])) > 0}
        
        distance_df = pd.DataFrame({
            'Vehicle': list(distance_data.keys()),
            'Distance (km)': list(distance_data.values())
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(time_df.set_index('Vehicle'))
        with col2:
            st.bar_chart(distance_df.set_index('Vehicle'))
    
    # Display the map with all routes
    st.subheader("Route Map with Road Navigation")

    # Add info about the route visualization
    st.info("""
    The map shows delivery routes that follow road networks from the depot to each stop in sequence, and back to the depot.
    Numbered circles indicate the stop sequence, and arrows show travel direction.
    """)
    
    # Create a map centered on Singapore
    singapore_coords = [1.3521, 103.8198]
    m = folium.Map(location=singapore_coords, zoom_start=12)
    
    # Add each vehicle's route to the map
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 
        'darkred', 'lightblue', 'darkgreen', 'cadetblue', 'darkpurple'
    ]
    
    # Count total route segments for progress bar
    total_segments = sum(len(route) + 1 for route in routes.values() if route)  # +1 for return to depot
    route_progress = st.progress(0)
    progress_container = st.empty()
    progress_container.text("Calculating routes: 0%")
    
    # Counter for processed segments
    processed_segments = 0
    
    for i, (vehicle_id, route) in enumerate(routes.items()):
        if not route:
            continue
        
        # Get vehicle info
        vehicle_info = vehicle_data[vehicle_data['vehicle_id'] == vehicle_id].iloc[0]
        
        # Use color cycling if we have more vehicles than colors
        color = colors[i % len(colors)]
        
        # Add depot marker
        depot_lat, depot_lon = vehicle_info['depot_latitude'], vehicle_info['depot_longitude']
        
        # Create depot popup content
        depot_popup = f"""
            <b>Depot:</b> {vehicle_id}<br>
            <b>Vehicle Type:</b> {vehicle_info['vehicle_type']}<br>
            <b>Driver:</b> {vehicle_info.get('driver_name', 'Unknown')}<br>
        """
        
        # Add depot marker with START label
        folium.Marker(
            [depot_lat, depot_lon],
            popup=folium.Popup(depot_popup, max_width=300),
            tooltip=f"Depot: {vehicle_id} (START/END)",
            icon=folium.Icon(color=color, icon='home', prefix='fa')
        ).add_to(m)
        
        # Create route points for complete journey
        waypoints = [(depot_lat, depot_lon)]  # Start at depot
        
        # Add all delivery locations as waypoints
        for delivery in route:
            waypoints.append((delivery['latitude'], delivery['longitude']))
        
        # Close the loop back to depot
        waypoints.append((depot_lat, depot_lon))
        
        # Add delivery point markers with sequenced numbering
        for j, delivery in enumerate(route):
            lat, lon = delivery['latitude'], delivery['longitude']
            
            # Create popup content
            popup_content = f"""
                <b>Stop {j+1}:</b> {delivery['id']}<br>
                <b>Customer:</b> {delivery.get('customer_name', 'Unknown')}<br>
            """
            
            if 'priority' in delivery:
                popup_content += f"<b>Priority:</b> {delivery['priority']}<br>"
                
            if 'weight_kg' in delivery:
                popup_content += f"<b>Weight:</b> {delivery['weight_kg']:.2f} kg<br>"
                
            if 'time_window' in delivery:
                popup_content += f"<b>Time Window:</b> {delivery['time_window']}<br>"
            
            # Add circle markers and other delivery visualizations
            folium.Circle(
                location=[lat, lon],
                radius=50,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"Stop {j+1}: {delivery['id']}"
            ).add_to(m)
            
            # Add text label with stop number
            folium.map.Marker(
                [lat, lon],
                icon=DivIcon(
                    icon_size=(20, 20),
                    icon_anchor=(10, 10),
                    html=f'<div style="font-size: 12pt; color: #444444; font-weight: bold; text-align: center;">{j+1}</div>',
                )
            ).add_to(m)
            
            # Add regular marker with popup
            folium.Marker(
                [lat + 0.0003, lon],  # slight offset to not overlap with the circle
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Delivery {delivery['id']}",
                icon=folium.Icon(color=color, icon='box', prefix='fa')
            ).add_to(m)
        
        # Create road-based routes between each waypoint with progress tracking
        for k in range(len(waypoints) - 1):
            # Get start and end points of this segment
            start_point = waypoints[k]
            end_point = waypoints[k+1]
            
            # Update progress text
            segment_desc = "depot" if k == 0 else f"stop {k}"
            next_desc = f"stop {k+1}" if k < len(waypoints) - 2 else "depot"
            progress_text = f"Calculating route for Vehicle {vehicle_id}: {segment_desc} → {next_desc}"
            
            # Use spinner context with progress updates
            with st.spinner(progress_text):
                # Get a road-like route between these points
                road_route = get_road_route(start_point, end_point)
                
                # Add the route line (non-animated)
                folium.PolyLine(
                    road_route,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    tooltip=f"Route {vehicle_id}: {segment_desc} → {next_desc}"
                ).add_to(m)
                
                # Add direction arrow
                idx = int(len(road_route) * 0.7)
                if idx < len(road_route) - 1:
                    p1 = road_route[idx]
                    p2 = road_route[idx + 1]
                    
                    # Calculate direction angle
                    dy = p2[0] - p1[0]
                    dx = p2[1] - p1[1]
                    angle = (90 - np.degrees(np.arctan2(dy, dx))) % 360
                    
                    # Add arrow marker
                    folium.RegularPolygonMarker(
                        location=p1,
                        number_of_sides=3,
                        radius=8,
                        rotation=angle,
                        color=color,
                        fill_color=color,
                        fill_opacity=0.8
                    ).add_to(m)
            
            # Update progress after each segment
            processed_segments += 1
            progress_percentage = int((processed_segments / total_segments) * 100)
            route_progress.progress(processed_segments / total_segments)
            progress_container.text(f"Calculating routes: {progress_percentage}%")
    
    # Clear progress display when done
    progress_container.empty()
    route_progress.empty()
    st.success("All routes calculated successfully!")
    
    # Display the map
    folium_static(m, width=800, height=600)
    
    # -----------------------------------------------------
    # Delivery Schedule Calendar Section
    # -----------------------------------------------------
    st.subheader("Delivery Schedule Calendar")
    st.write("This calendar shows the planned delivery schedule, with on-time deliveries in green and late deliveries in red.")

    # Process data for calendar view
    if routes:
        calendar_data = []
        
        for vehicle_id, route in routes.items():
            for delivery in route:
                # Get vehicle info
                vehicle_info = vehicle_data[vehicle_data['vehicle_id'] == vehicle_id].iloc[0]
                vehicle_type = vehicle_info.get('vehicle_type', 'Standard')
                driver_name = vehicle_info.get('driver_name', 'Unknown')
                
                # Extract delivery data
                delivery_id = delivery['id']
                customer_name = delivery.get('customer_name', 'Unknown')
                priority = delivery.get('priority', 'Medium')
                time_window = delivery.get('time_window', '09:00-17:00')
                weight = delivery.get('weight_kg', 0)
                
                # Extract start and end times from time_window
                start_time_str, end_time_str = time_window.split('-')
                
                # Get delivery date from original data
                delivery_row = delivery_data[delivery_data['delivery_id'] == delivery_id]
                delivery_date = delivery_row['delivery_date'].iloc[0] if not delivery_row.empty and 'delivery_date' in delivery_row else datetime.now().date()
                
                # Create start and end datetime for the delivery
                try:
                    # Convert to pandas datetime
                    if isinstance(delivery_date, pd.Timestamp):
                        date_str = delivery_date.strftime('%Y-%m-%d')
                    elif isinstance(delivery_date, str):
                        date_str = pd.to_datetime(delivery_date).strftime('%Y-%m-%d')
                    else:
                        date_str = delivery_date.strftime('%Y-%m-%d')
                    
                    start_datetime = pd.to_datetime(f"{date_str} {start_time_str}")
                    end_datetime = pd.to_datetime(f"{date_str} {end_time_str}")
                    
                    # Check if this is on time (based on the estimated arrival from the route)
                    estimated_arrival_mins = delivery.get('estimated_arrival', 0)
                    
                    # Convert time_window to minutes for comparison
                    start_mins = int(start_time_str.split(':')[0]) * 60 + int(start_time_str.split(':')[1])
                    end_mins = int(end_time_str.split(':')[0]) * 60 + int(end_time_str.split(':')[1])
                    
                    # Determine if delivery is on time
                    on_time = start_mins <= estimated_arrival_mins <= end_mins
                    
                    calendar_data.append({
                        'delivery_id': delivery_id,
                        'customer_name': customer_name,
                        'vehicle_id': vehicle_id,
                        'driver_name': driver_name,
                        'vehicle_type': vehicle_type,
                        'priority': priority,
                        'time_window': time_window,
                        'estimated_arrival_mins': estimated_arrival_mins,
                        'estimated_arrival_time': f"{estimated_arrival_mins//60:02d}:{estimated_arrival_mins%60:02d}",
                        'weight_kg': weight,
                        'Start': start_datetime,
                        'Finish': end_datetime,
                        'Task': f"{delivery_id}: {customer_name}",
                        'on_time': on_time,
                        'delivery_date': pd.to_datetime(date_str)
                    })
                except Exception as e:
                    st.warning(f"Could not process time window for delivery {delivery_id}: {str(e)}")
        
        if calendar_data:
            # Convert to DataFrame
            cal_df = pd.DataFrame(calendar_data)
            
            # Create color mapping for on-time status
            cal_df['Color'] = cal_df['on_time'].map({True: 'rgb(0, 200, 0)', False: 'rgb(255, 0, 0)'})
            
            # Group tasks by date for visualization
            date_groups = cal_df.groupby(cal_df['delivery_date'].dt.date)
            
            # Create tabs for each date
            date_tabs = st.tabs([date.strftime('%b %d, %Y') for date in sorted(date_groups.groups.keys())])
            
            for i, (date, tab) in enumerate(zip(sorted(date_groups.groups.keys()), date_tabs)):
                with tab:
                    # Filter data for this date
                    day_data = cal_df[cal_df['delivery_date'].dt.date == date]
                    
                    if len(day_data) > 0:
                        # Create figure
                        fig = px.timeline(
                            day_data, 
                            x_start="Start", 
                            x_end="Finish", 
                            y="Task",
                            color="on_time",
                            color_discrete_map={True: "green", False: "red"},
                            hover_data=["customer_name", "vehicle_id", "driver_name", "priority", "time_window", 
                                       "estimated_arrival_time", "weight_kg"]
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
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Deliveries", len(day_data))
                        with col2:
                            st.metric("On-Time Deliveries", len(day_data[day_data['on_time']]))
                        with col3:
                            st.metric("Late Deliveries", len(day_data[~day_data['on_time']]))
                        with col4:
                            if 'weight_kg' in day_data.columns:
                                st.metric("Total Weight", f"{day_data['weight_kg'].sum():.2f} kg")
                        
                        # Add breakdown of deliveries by priority
                        if 'priority' in day_data.columns:
                            st.write("##### Deliveries by Priority")
                            priority_counts = day_data['priority'].value_counts()
                            priority_cols = st.columns(min(4, len(priority_counts)))
                            
                            for i, (priority, count) in enumerate(priority_counts.items()):
                                col_idx = i % len(priority_cols)
                                with priority_cols[col_idx]:
                                    st.metric(priority, count)
                    else:
                        st.info(f"No deliveries scheduled for {date.strftime('%b %d, %Y')}")
        else:
            st.info("No delivery calendar data available. Please generate routes first.")

    # -----------------------------------------------------
    # Vehicle Schedule Calendar Section
    # -----------------------------------------------------
    st.subheader("Vehicle Schedule Calendar")
    st.write("This calendar shows the schedule for each vehicle, helping visualize their workload throughout the day.")

    # Process data for vehicle calendar view (same data, different grouping)
    if routes and calendar_data:
        # Convert list to DataFrame if needed
        if not isinstance(calendar_data, pd.DataFrame):
            vehicle_cal_df = pd.DataFrame(calendar_data)
        else:
            vehicle_cal_df = calendar_data.copy()
        
        # Create a new Task column based on vehicle ID
        vehicle_cal_df['Vehicle Task'] = vehicle_cal_df['vehicle_id'] + ': ' + vehicle_cal_df['driver_name']
        
        # Group by date
        date_groups = vehicle_cal_df.groupby(vehicle_cal_df['delivery_date'].dt.date)
        
        # Create tabs for each date
        date_tabs = st.tabs([date.strftime('%b %d, %Y') for date in sorted(date_groups.groups.keys())])
        
        for i, (date, tab) in enumerate(zip(sorted(date_groups.groups.keys()), date_tabs)):
            with tab:
                # Filter data for this date
                day_data = vehicle_cal_df[vehicle_cal_df['delivery_date'].dt.date == date]
                
                if len(day_data) > 0:
                    # Create figure grouped by vehicle
                    fig = px.timeline(
                        day_data, 
                        x_start="Start", 
                        x_end="Finish", 
                        y="Vehicle Task",
                        color="on_time",
                        color_discrete_map={True: "green", False: "red"},
                        hover_data=["delivery_id", "customer_name", "priority", "time_window", 
                                   "estimated_arrival_time", "weight_kg"]
                    )
                    
                    # Add labels for each delivery to the bars
                    for idx, row in day_data.iterrows():
                        fig.add_annotation(
                            x=(row['Start'] + (row['Finish'] - row['Start'])/2),
                            y=row['Vehicle Task'],
                            text=f"#{row['delivery_id']}",
                            showarrow=False,
                            font=dict(size=10, color="black")
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Vehicle schedule for {date.strftime('%b %d, %Y')}",
                        xaxis_title="Time of Day",
                        yaxis_title="Vehicle",
                        height=max(300, 70 * day_data['Vehicle Task'].nunique()),
                        yaxis={'categoryorder':'category ascending'}
                    )
                    
                    # Display figure
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info(f"No vehicle schedules for {date.strftime('%b %d, %Y')}")
    else:
        st.info("No vehicle schedule data available. Please generate routes first.")
    
    # Display detailed route information
    st.subheader("Route Details")
    
    # Create tabs for each route
    if not routes:
        st.warning("No routes were created. Try adjusting the optimization parameters.")
    else:
        tabs = st.tabs([f"Vehicle {vid}" for vid in routes.keys() if routes[vid]])
        
        for i, (vehicle_id, route) in enumerate([(vid, r) for vid, r in routes.items() if r]):
            with tabs[i]:
                if not route:
                    st.info(f"No deliveries assigned to vehicle {vehicle_id}")
                    continue
                    
                # Get vehicle info
                vehicle_info = vehicle_data[vehicle_data['vehicle_id'] == vehicle_id].iloc[0]
                
                # Display vehicle details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Vehicle", vehicle_id)
                    
                with col2:
                    st.metric("Driver", vehicle_info.get('driver_name', 'Unknown'))
                    
                with col3:
                    st.metric("Total Deliveries", len(route))
                
                # Display statistics from route_stats if available
                stats = optimization_result.get('stats', {}).get(vehicle_id, {})
                col1, col2 = st.columns(2)
                
                with col1:
                    distance = stats.get('total_distance_km', 'N/A')
                    st.metric("Total Distance", f"{distance} km" if isinstance(distance, (int, float)) else distance)
                    
                with col2:
                    time_mins = stats.get('estimated_time_mins', 'N/A')
                    st.metric("Estimated Time", f"{time_mins} mins" if isinstance(time_mins, (int, float)) else time_mins)
                
                # Display the route as a table
                st.write("#### Delivery Sequence")
                
                # Create route data for table
                route_data = []
                for j, delivery in enumerate(route):
                    delivery_id = delivery['id']  # FIXED: Use 'id' instead of 'delivery_id'
                    
                    # Find more details from the original data
                    matching_rows = delivery_data[delivery_data['delivery_id'] == delivery_id]
                    if not matching_rows.empty:
                        delivery_details = matching_rows.iloc[0]
                        
                        route_data.append({
                            'Stop': j + 1,
                            'Delivery ID': delivery_id,
                            'Customer': delivery_details.get('customer_name', delivery.get('customer_name', 'Unknown')),
                            'Priority': delivery_details.get('priority', delivery.get('priority', 'Medium')),
                            'Time Window': delivery_details.get('time_window', delivery.get('time_window', '09:00-17:00')),
                            'Weight (kg)': round(delivery_details.get('weight_kg', delivery.get('weight_kg', 0)), 2)
                        })
                    else:
                        # If not found in original data, use the data from the delivery object
                        route_data.append({
                            'Stop': j + 1,
                            'Delivery ID': delivery_id,
                            'Customer': delivery.get('customer_name', 'Unknown'),
                            'Priority': delivery.get('priority', 'Medium'),
                            'Time Window': delivery.get('time_window', '09:00-17:00'),
                            'Weight (kg)': round(delivery.get('weight_kg', 0), 2)
                        })
                
                # Convert to DataFrame for display
                route_df = pd.DataFrame(route_data)
                st.dataframe(route_df)

def create_distance_matrix(locations):
    """
    Create a simple Euclidean distance matrix between locations
    
    In a real implementation, this would be replaced by actual road distances
    
    Parameters:
        locations (list): List of location dictionaries with lat and lon
        
    Returns:
        numpy.ndarray: Distance matrix
    """
    n = len(locations)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Approximate distance in km (very rough)
            lat1, lon1 = locations[i]['latitude'], locations[i]['longitude']
            lat2, lon2 = locations[j]['latitude'], locations[j]['longitude']
            
            # Simple Euclidean distance (for demo purposes)
            # In reality, we'd use actual road distances
            dist = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111
            matrix[i, j] = dist
    
    return matrix

def get_road_route(start_point, end_point):
    """
    Get a route that follows actual roads between two points using OpenStreetMap's routing service.
    
    Args:
        start_point: (lat, lon) tuple of start location
        end_point: (lat, lon) tuple of end location
        
    Returns:
        list: List of (lat, lon) points representing the actual road route
    """
    try:
        # OSRM expects coordinates in lon,lat format
        start_lat, start_lon = start_point
        end_lat, end_lon = end_point
        
        # Build the API URL for OSRM (OpenStreetMap Routing Machine)
        url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true"
        }
        
        # Replace direct text output with spinner
        with st.spinner(f"Getting route from ({start_lat:.4f}, {start_lon:.4f}) to ({end_lat:.4f}, {end_lon:.4f})..."):
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if a route was found
                if data['code'] == 'Ok' and len(data['routes']) > 0:
                    # Extract the geometry (list of coordinates) from the response
                    geometry = data['routes'][0]['geometry']['coordinates']
                    
                    # OSRM returns points as [lon, lat], but we need [lat, lon]
                    route_points = [(lon, lat) for lat, lon in geometry]
                    return route_points
        
        # If we get here, something went wrong with the API call
        st.warning(f"Could not get road route: {response.status_code} - {response.text if response.status_code != 200 else 'No routes found'}")
        
    except Exception as e:
        st.warning(f"Error getting road route: {str(e)}")
    
    # Fallback to our approximation method if the API call fails
    with st.spinner("Generating approximate route..."):
        # Create a more sophisticated approximation with higher density of points
        start_lat, start_lon = start_point
        end_lat, end_lon = end_point
        
        # Calculate the direct distance
        direct_dist = ((start_lat - end_lat)**2 + (start_lon - end_lon)**2)**0.5
        
        # Generate more points for longer distances
        num_points = max(10, int(direct_dist * 10000))  # Scale based on distance
        
        # Create a path with small random deviations to look like a road
        route_points = []
        
        # Starting point
        route_points.append((start_lat, start_lon))
        
        # Calculate major waypoints - like going through major roads
        # Find a midpoint that's slightly off the direct line
        mid_lat = (start_lat + end_lat) / 2
        mid_lon = (start_lon + end_lon) / 2
        
        # Add some perpendicular deviation to simulate taking streets
        # Get perpendicular direction
        dx = end_lat - start_lat
        dy = end_lon - start_lon
        
        # Perpendicular direction
        perpendicular_x = -dy
        perpendicular_y = dx
        
        # Normalize and scale
        magnitude = (perpendicular_x**2 + perpendicular_y**2)**0.5
        if magnitude > 0:
            perpendicular_x /= magnitude
            perpendicular_y /= magnitude
        
        # Scale the perpendicular offset based on distance
        offset_scale = direct_dist * 0.2  # 20% of direct distance
        
        # Apply offset to midpoint
        mid_lat += perpendicular_x * offset_scale * random.choice([-1, 1])
        mid_lon += perpendicular_y * offset_scale * random.choice([-1, 1])
        
        # Generate a smooth path from start to midpoint
        for i in range(1, num_points // 2):
            t = i / (num_points // 2)
            # Quadratic Bezier curve parameters
            u = 1 - t
            lat = u**2 * start_lat + 2 * u * t * mid_lat + t**2 * mid_lat
            lon = u**2 * start_lon + 2 * u * t * mid_lon + t**2 * mid_lon
            
            # Add small random noise to make it look like following streets
            noise_scale = 0.0002 * direct_dist
            lat += random.uniform(-noise_scale, noise_scale)
            lon += random.uniform(-noise_scale, noise_scale)
            
            route_points.append((lat, lon))
        
        # Generate a smooth path from midpoint to end
        for i in range(num_points // 2, num_points):
            t = (i - num_points // 2) / (num_points // 2)
            # Quadratic Bezier curve parameters
            u = 1 - t
            lat = u**2 * mid_lat + 2 * u * t * mid_lat + t**2 * end_lat
            lon = u**2 * mid_lon + 2 * u * t * mid_lon + t**2 * end_lon
            
            # Add small random noise to make it look like following streets
            noise_scale = 0.0002 * direct_dist
            lat += random.uniform(-noise_scale, noise_scale)
            lon += random.uniform(-noise_scale, noise_scale)
            
            route_points.append((lat, lon))
        
        # Ending point
        route_points.append((end_lat, end_lon))
    
    return route_points

# Add this condition to make the function importable
if __name__ == "__main__":
    st.set_page_config(
        page_title="Route Optimizer - Delivery Route Optimization",
        page_icon="🛣️",
        layout="wide"
    )
    optimize_page()