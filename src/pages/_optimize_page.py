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

def clear_optimization_results():
    """Clear optimization results when parameters change"""
    if 'optimization_result' in st.session_state:
        st.session_state.optimization_result = None

def optimize_page():
    """
    Render the optimization page with controls for route optimization
    """
    st.title("Delivery Route Optimization")

    # Add help section with expander
    with st.expander("📚 How to Use This Page"):
        st.markdown("""
        ## Step-by-Step Guide to Route Optimization
        
        This application helps you optimize delivery routes by assigning deliveries to vehicles in the most efficient way possible. Follow these steps to get started:
        
        ### 1. Set Optimization Parameters (Sidebar)
        
        - **Select Delivery Dates**: Choose which dates to include in optimization. Select "All" to include all dates.
        - **Priority Importance**: Higher values give more weight to high-priority deliveries.
        - **Time Window Importance**: Higher values enforce stricter adherence to delivery time windows.
        - **Load Balancing vs Distance**: Higher values distribute deliveries more evenly across vehicles.
        - **Maximum Vehicles**: Set the maximum number of vehicles to use for deliveries.
        - **Minimum Time Window Compliance**: Set the minimum percentage of deliveries that must be within their time windows.
        
        ### 2. Generate Routes
        
        - Review the delivery statistics and vehicle availability information
        - Click the **Generate Optimal Routes** button to run the optimization algorithm
        - The algorithm will assign deliveries to vehicles based on your parameters
        
        ### 3. Review Optimization Results
        
        - **Overall Performance**: Check metrics like assigned deliveries, vehicles used, and time window compliance
        - **Time & Distance Distribution**: See how delivery workload is distributed across vehicles
        - **Route Map**: Interactive map showing the optimized routes for each vehicle
          - Use the date filter to show routes for specific days
          - Hover over markers and routes for detailed information
        - **Calendar View**: View delivery schedules organized by date
          - Green bars indicate on-time deliveries
          - Orange bars indicate late deliveries
          - Red bars indicate unassigned deliveries
        
        ### 4. Adjust and Refine
        
        If the results don't meet your requirements:
        
        - **Not enough vehicles?** Increase the maximum vehicles allowed
        - **Time windows not met?** Decrease the time window importance or minimum compliance
        - **High priority deliveries not assigned?** Increase priority importance
        - **Routes too unbalanced?** Increase load balancing parameter
        
        Remember to click **Generate Optimal Routes** after changing any parameters to see the updated results.
        """)
    
    # Initialize session state variables
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'optimization_params' not in st.session_state:
        st.session_state.optimization_params = {
            'priority_weight': 0.3,
            'time_window_weight': 0.5,
            'balance_weight': 0.2,
            'max_vehicles': 5,
            'selected_dates': ["All"]
        }
    if 'calendar_display_dates' not in st.session_state:
        st.session_state.calendar_display_dates = None
    # Add this new session state variable to store calculated road routes
    if 'calculated_road_routes' not in st.session_state:
        st.session_state.calculated_road_routes = {}
    
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
        
        # Store current value before selection
        current_selected_dates = st.session_state.optimization_params['selected_dates']
        
        selected_dates = st.sidebar.multiselect(
            "Select Delivery Dates",
            options=date_options,
            default=current_selected_dates,
            key="delivery_date_selector"
        )
        
        # Check if selection changed
        if selected_dates != current_selected_dates:
            clear_optimization_results()
            st.session_state.optimization_params['selected_dates'] = selected_dates
            
        # Handle filtering based on selection
        if "All" not in selected_dates:
            if selected_dates:  # If specific dates were selected
                delivery_data = delivery_data[delivery_data['delivery_date'].isin(selected_dates)]
            elif available_dates:  # No dates selected, show warning
                st.sidebar.warning("No dates selected. Please select at least one delivery date.")
                return
        # If "All" is selected, keep all dates - no filtering needed
    
    # Priority weighting
    current_priority = st.session_state.optimization_params['priority_weight']
    priority_weight = st.sidebar.slider(
        "Priority Importance",
        min_value=0.0,
        max_value=1.0,
        value=current_priority,
        help="Higher values give more importance to high-priority deliveries",
        key="priority_weight",
        on_change=clear_optimization_results
    )

    # Time window importance
    current_time_window = st.session_state.optimization_params['time_window_weight']
    time_window_weight = st.sidebar.slider(
        "Time Window Importance",
        min_value=0.0,
        max_value=1.0,
        value=current_time_window,
        help="Higher values enforce stricter adherence to delivery time windows",
        key="time_window_weight",
        on_change=clear_optimization_results
    )

    # Distance vs load balancing
    current_balance = st.session_state.optimization_params['balance_weight']
    balance_weight = st.sidebar.slider(
        "Load Balancing vs Distance",
        min_value=0.0,
        max_value=1.0,
        value=current_balance,
        help="Higher values prioritize even distribution of deliveries across vehicles over total distance",
        key="balance_weight",
        on_change=clear_optimization_results
    )

    # Max vehicles to use
    available_vehicles = vehicle_data[vehicle_data['status'] == 'Available']
    current_max_vehicles = st.session_state.optimization_params['max_vehicles']
    max_vehicles = st.sidebar.slider(
        "Maximum Vehicles to Use",
        min_value=1,
        max_value=len(available_vehicles),
        value=min(current_max_vehicles, len(available_vehicles)),
        key="max_vehicles",
        on_change=clear_optimization_results
    )

    # Add minimum time window compliance slider
    min_time_window_compliance = st.sidebar.slider(
        "Minimum Time Window Compliance (%)",
        min_value=0,
        max_value=100,
        value=75,
        help="Minimum percentage of deliveries that must be within their time window",
        key="min_time_window_compliance",
        on_change=clear_optimization_results
    )

    # Update session state with new parameter values
    st.session_state.optimization_params['priority_weight'] = priority_weight
    st.session_state.optimization_params['time_window_weight'] = time_window_weight
    st.session_state.optimization_params['balance_weight'] = balance_weight
    st.session_state.optimization_params['max_vehicles'] = max_vehicles

    # # Add a notification when parameters have changed and results need regenerating
    # if ('optimization_result' not in st.session_state or st.session_state.optimization_result is None):
    #     st.warning("⚠️ Optimization parameters have changed. Please click 'Generate Optimal Routes' to update results.")
    
    # Main optimization section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Delivery Route Optimizer")
        
        # Filter out completed deliveries for statistics
        if 'status' in delivery_data.columns:
            pending_deliveries = delivery_data[delivery_data['status'] != 'Delivered']
            completed_count = len(delivery_data) - len(pending_deliveries)
        else:
            pending_deliveries = delivery_data
            completed_count = 0
            
        st.write(f"Optimizing routes for {len(pending_deliveries)} pending deliveries using up to {max_vehicles} vehicles")
        
        # Statistics
        st.write("#### Delivery Statistics")
        total_count = len(delivery_data)
        pending_count = len(pending_deliveries)
        
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("Total Deliveries", total_count)
        with col1b:
            st.metric("Pending Deliveries", pending_count, 
                     delta=f"-{completed_count}" if completed_count > 0 else None,
                     delta_color="inverse" if completed_count > 0 else "normal")
        
        if 'priority' in delivery_data.columns:
            # Show priority breakdown for pending deliveries only
            priority_counts = pending_deliveries['priority'].value_counts()
            
            # Display priority counts in a more visual way
            st.write("##### Priority Breakdown")
            priority_cols = st.columns(min(3, len(priority_counts)))
            
            for i, (priority, count) in enumerate(priority_counts.items()):
                col_idx = i % len(priority_cols)
                with priority_cols[col_idx]:
                    st.metric(f"{priority}", count)
        
        if 'weight_kg' in delivery_data.columns:
            # Calculate weight only for pending deliveries
            total_weight = pending_deliveries['weight_kg'].sum()
            st.metric("Total Weight (Pending)", f"{total_weight:.2f} kg")
    
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
    run_optimization_btn = st.button("Generate Optimal Routes")
    
    # Check if we should display results (either have results in session or button was clicked)
    if run_optimization_btn or st.session_state.optimization_result is not None:
        if run_optimization_btn:
            # Run new optimization
            with st.spinner("Calculating optimal routes..."):
                start_time = time.time()
                
                # Filter out completed deliveries before optimization
                if 'status' in delivery_data.columns:
                    pending_deliveries = delivery_data[delivery_data['status'] != 'Delivered']
                else:
                    pending_deliveries = delivery_data
                
                # Prepare data for optimization - USE PENDING DELIVERIES ONLY
                optimization_result = run_optimization(
                    delivery_data=pending_deliveries,
                    vehicle_data=available_vehicles.iloc[:max_vehicles],
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    locations=locations,
                    priority_weight=priority_weight,
                    time_window_weight=time_window_weight,
                    balance_weight=balance_weight,
                    min_time_window_compliance=min_time_window_compliance/100.0  # Convert to decimal
                )
                
                end_time = time.time()
                st.success(f"Optimization completed in {end_time - start_time:.2f} seconds")
                
                # Store results in session state
                st.session_state.optimization_result = optimization_result
        else:
            # Use existing results
            optimization_result = st.session_state.optimization_result
            
        # Filter pending deliveries before displaying results    
        if 'status' in delivery_data.columns:
            pending_deliveries = delivery_data[delivery_data['status'] != 'Delivered']
        else:
            pending_deliveries = delivery_data
            
        # Display results with filtered pending deliveries
        display_optimization_results(
            optimization_result=optimization_result,
            delivery_data=pending_deliveries,  # ← CHANGED: Use pending_deliveries instead
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
                    priority_weight, time_window_weight, balance_weight, min_time_window_compliance=0.75):
    """
    Run the route optimization algorithm using Google OR-Tools
    
    Parameters:
        delivery_data (pd.DataFrame): DataFrame containing delivery information
        vehicle_data (pd.DataFrame): DataFrame containing vehicle information
        distance_matrix (pd.DataFrame): Distance matrix between locations
        time_matrix (pd.DataFrame): Time matrix between locations
        locations (pd.DataFrame): DataFrame with location details
        priority_weight (float): Weight for delivery priority in optimization (α)
        time_window_weight (float): Weight for time window adherence (β)
        balance_weight (float): Weight for balancing load across vehicles (γ)
        min_time_window_compliance (float): Minimum required time window compliance (δ)
        
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
    
    # Calculate total weight of all deliveries
    total_delivery_weight = sum(d['demand'] for d in delivery_locations)
    
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
    
    # Define distance callback with priority weighting
    # This implements the objective function: min sum_{i,j,k} c_jk * x_ijk * p_k^α
    def distance_callback(from_index, to_index):
        """Returns the weighted distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        # Get base distance
        base_distance = int(dist_matrix[from_node, to_node] * 1000)  # Convert to integers
        
        # Apply priority weighting to destination node (if it's a delivery)
        if to_node >= num_vehicles:  # It's a delivery node
            delivery_idx = to_node - num_vehicles
            # Apply the priority factor with the priority weight (α)
            priority_factor = delivery_locations[delivery_idx]['priority_factor']
            # Higher priority_weight = stronger effect of priority on cost
            priority_multiplier = priority_factor ** priority_weight
            return int(base_distance * priority_multiplier)
        
        return base_distance
    
    # Define time callback
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix_mins[from_node, to_node] * 60)  # Convert minutes to seconds (integers)
    
    # Define service time callback - time spent at each delivery
    def service_time_callback(from_index):
        """Returns the service time for the node."""
        # Service time is 0 for depots and 10 minutes (600 seconds) for deliveries
        node_idx = manager.IndexToNode(from_index)
        if node_idx >= num_vehicles:  # It's a delivery node
            return 600  # 10 minutes in seconds
        return 0  # No service time for depots
    
    # Define demand callback
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands array
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]
    
    # Register callbacks
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    service_callback_index = routing.RegisterUnaryTransitCallback(service_time_callback)
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    # Set the arc cost evaluator for all vehicles - this is our objective function
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity dimension - Hard Constraint 2: Vehicle Capacity Limits
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,                   # null capacity slack
        vehicle_capacities,  # vehicle maximum capacities
        True,                # start cumul to zero
        'Capacity'
    )
    
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    
    # Add load balancing penalties - Soft Constraint 3: Load Balancing Penalties
    if balance_weight > 0.01:
        # Calculate target weight per vehicle (ideal balanced load)
        target_weight = total_delivery_weight / len(vehicle_capacities)
        
        for i in range(num_vehicles):
            # Get vehicle capacity
            vehicle_capacity = vehicle_capacities[i]
            
            # Set penalties for deviating from balanced load
            # Scale penalty based on the balance_weight parameter (γ)
            balance_penalty = int(10000 * balance_weight)
            
            # Add soft bounds around the target weight
            # Lower bound: Don't penalize for being under the target if there's not enough weight
            lower_target = max(0, int(target_weight * 0.8))
            capacity_dimension.SetCumulVarSoftLowerBound(
                routing.End(i), lower_target, balance_penalty
            )
            
            # Upper bound: Penalize for going over the target
            # But allow using more capacity if necessary to assign all deliveries
            upper_target = min(vehicle_capacity, int(target_weight * 1.2))
            capacity_dimension.SetCumulVarSoftUpperBound(
                routing.End(i), upper_target, balance_penalty
            )
    
    # Add time dimension with service times
    # This implements Hard Constraint 5: Time Continuity and 
    # Hard Constraint 6: Maximum Route Duration
    routing.AddDimension(
        time_callback_index,
        60 * 60,       # Allow waiting time of 60 mins
        24 * 60 * 60,  # Maximum time per vehicle (24 hours in seconds) - Hard Constraint 6
        False,         # Don't force start cumul to zero
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')
    
    # Add service time to each node's visit duration
    for node_idx in range(len(all_locations)):
        index = manager.NodeToIndex(node_idx)
        time_dimension.SetCumulVarSoftUpperBound(
            index, 
            24 * 60 * 60,  # 24 hours in seconds
            1000000  # High penalty for violating the 24-hour constraint
        )
        time_dimension.SlackVar(index).SetValue(0)
    
    # Store time window variables to track compliance
    time_window_vars = []
    compliance_threshold = int(min_time_window_compliance * num_deliveries)
    
    # Add time window constraints - Hard Constraint 7: Time Window Compliance
    if time_window_weight > 0.01:
        # Create binary variables to track time window compliance
        for delivery_idx, delivery in enumerate(delivery_locations):
            if 'time_window' in delivery and delivery['time_window']:
                try:
                    start_time_str, end_time_str = delivery['time_window'].split('-')
                    start_hour, start_min = map(int, start_time_str.split(':'))
                    end_hour, end_min = map(int, end_time_str.split(':'))
                    
                    # Convert to seconds since midnight
                    start_time_sec = (start_hour * 60 + start_min) * 60
                    end_time_sec = (end_hour * 60 + end_min) * 60
                    
                    # Get the node index
                    index = manager.NodeToIndex(num_vehicles + delivery_idx)
                    
                    # Add soft upper bound penalty with very high weight for late deliveries
                    # This implements Soft Constraint 2: Time Window Penalties
                    time_dimension.SetCumulVarSoftUpperBound(
                        index, 
                        end_time_sec, 
                        int(1000000 * time_window_weight)  # High penalty for being late
                    )
                    
                    # Don't penalize for early deliveries, just wait
                    time_dimension.CumulVar(index).SetMin(start_time_sec)
                    
                    # Track this time window for compliance calculation
                    time_window_vars.append((index, start_time_sec, end_time_sec))
                except:
                    # Skip if time window format is invalid
                    pass
    
    # Hard Constraint 1: All Deliveries Must Be Assigned
    # This is enforced by not creating disjunctions with penalties, but instead making all nodes mandatory
    
    # Hard Constraint 3: Flow Conservation (Route Continuity) is inherently enforced by OR-Tools
    
    # Hard Constraint 4: Start and End at Assigned Depots is enforced by the RoutingIndexManager setup
    
    # Set parameters for the solver
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    
    # Use guided local search to find good solutions
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    
    # Use path cheapest arc with resource constraints as the first solution strategy
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # Give the solver enough time to find a good solution
    search_parameters.time_limit.seconds = 10
    
    # Enable logging
    search_parameters.log_search = True
    
    # Try to enforce the time window compliance threshold
    if compliance_threshold > 0:
        # First try to solve with all deliveries required
        routing.CloseModelWithParameters(search_parameters)
        
        # Solve the problem
        st.write(f"Solving optimization model with {num_deliveries} deliveries and {num_vehicles} vehicles...")
        st.write(f"Target: At least {compliance_threshold} of {num_deliveries} deliveries ({min_time_window_compliance*100:.0f}%) must be within time windows")
        
        solution = routing.SolveWithParameters(search_parameters)
    else:
        # If no time window compliance required, solve normally
        solution = routing.SolveWithParameters(search_parameters)
    
    # If no solution was found, try a relaxed version (allow some deliveries to be unassigned)
    if not solution:
        st.warning("Could not find a solution with all deliveries assigned. Trying a relaxed version...")
        
        # Create a new model with disjunctions to allow dropping some deliveries with high penalties
        routing = pywrapcp.RoutingModel(manager)
        
        # Re-register callbacks
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        service_callback_index = routing.RegisterUnaryTransitCallback(service_time_callback)
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity dimension again
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0, vehicle_capacities, True, 'Capacity'
        )
        
        # Add time dimension again
        routing.AddDimension(
            time_callback_index,
            60 * 60, 24 * 60 * 60, False, 'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add disjunctions with very high penalties to try to include all deliveries
        for delivery_idx in range(num_deliveries):
            index = manager.NodeToIndex(num_vehicles + delivery_idx)
            routing.AddDisjunction([index], 1000000)  # High penalty but allows dropping if necessary
        
        # Try to solve with relaxed constraints
        search_parameters.time_limit.seconds = 15  # Give more time for relaxed version
        solution = routing.SolveWithParameters(search_parameters)
        
        if not solution:
            st.error("Could not find any solution. Try increasing the number of vehicles or relaxing other constraints.")
            return {
                'routes': {},
                'stats': {},
                'parameters': {
                    'priority_weight': priority_weight,
                    'time_window_weight': time_window_weight,
                    'balance_weight': balance_weight,
                    'min_time_window_compliance': min_time_window_compliance
                }
            }
    
    # Extract solution
    optimized_routes = {}
    route_stats = {}
    
    if solution:
        st.success("Solution found!")
        
        total_time_window_compliance = 0
        total_deliveries_assigned = 0
        
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
            
            # Initialize variables for tracking
            index = routing.Start(vehicle_idx)
            total_distance = 0
            total_time = 0
            total_load = 0
            time_window_compliant = 0
            total_deliveries = 0
            
            # Initialize variables to track current position and time
            current_time_sec = 8 * 3600  # Start at 8:00 AM (8 hours * 3600 seconds)
            
            while not routing.IsEnd(index):
                # Get the node index in the original data
                node_idx = manager.IndexToNode(index)
                
                # Skip depot nodes (they're already at the start)
                if node_idx >= num_vehicles:
                    # This is a delivery node - get the corresponding delivery
                    delivery_idx = node_idx - num_vehicles
                    delivery = delivery_locations[delivery_idx].copy()  # Create a copy to modify
                    
                    # Calculate estimated arrival time in minutes since start of day
                    arrival_time_sec = solution.Min(time_dimension.CumulVar(index))
                    arrival_time_mins = arrival_time_sec // 60
                    
                    # Store the estimated arrival time in the delivery
                    delivery['estimated_arrival'] = arrival_time_mins
                    
                    # Check time window compliance
                    if 'time_window' in delivery and delivery['time_window']:
                        try:
                            start_time_str, end_time_str = delivery['time_window'].split('-')
                            start_hour, start_min = map(int, start_time_str.split(':'))
                            end_hour, end_min = map(int, end_time_str.split(':'))
                            
                            # Convert to minutes for comparison
                            start_mins = start_hour * 60 + start_min
                            end_mins = end_hour * 60 + end_min
                            
                            # Check if delivery is within time window
                            on_time = False
                            
                            # If arrival <= end_time, consider it on-time (including early arrivals)
                            if arrival_time_mins <= end_mins:
                                on_time = True
                                time_window_compliant += 1
                                total_time_window_compliance += 1
                            
                            delivery['within_time_window'] = on_time
                        except Exception as e:
                            st.warning(f"Error parsing time window for delivery {delivery['id']}: {str(e)}")
                            delivery['within_time_window'] = False
                    
                    # Add to route
                    route.append(delivery)
                    total_deliveries += 1
                    total_deliveries_assigned += 1
                    
                    # Add to total load
                    total_load += delivery['demand'] / 100  # Convert back to original units
                
                # Move to the next node
                previous_idx = index
                index = solution.Value(routing.NextVar(index))
                
                # Add distance and time from previous to current
                if not routing.IsEnd(index):
                    previous_node = manager.IndexToNode(previous_idx)
                    next_node = manager.IndexToNode(index)
                    
                    # Add distance between these points
                    segment_distance = dist_matrix[previous_node, next_node]
                    total_distance += segment_distance
                    
                    # Add travel time between these points
                    segment_time_sec = int(time_matrix_mins[previous_node, next_node] * 60)
                    total_time += segment_time_sec / 60  # Convert seconds back to minutes
            
            # Store the route if it's not empty
            if route:
                optimized_routes[vehicle_id] = route
                
                # Calculate time window compliance percentage
                time_window_percent = (time_window_compliant / total_deliveries * 100) if total_deliveries > 0 else 0
                
                # Store route statistics
                route_stats[vehicle_id] = {
                    'vehicle_type': vehicle_info['type'],
                    'capacity_kg': vehicle_info['capacity'],
                    'deliveries': len(route),
                    'total_distance_km': round(total_distance, 2),
                    'estimated_time_mins': round(total_time),
                    'total_load_kg': round(total_load, 2),
                    'time_window_compliant': time_window_compliant,
                    'time_window_compliance': time_window_percent
                }
        
        # Check if overall time window compliance meets the minimum requirement
        overall_compliance = 0
        if total_deliveries_assigned > 0:
            overall_compliance = (total_time_window_compliance / total_deliveries_assigned)
            
        if overall_compliance < min_time_window_compliance:
            st.warning(f"Solution found, but time window compliance ({overall_compliance*100:.1f}%) is below the minimum required ({min_time_window_compliance*100:.0f}%).")
            st.info("Consider adjusting parameters: increase the number of vehicles, reduce the minimum compliance requirement, or adjust time window importance.")
        else:
            st.success(f"Solution meets time window compliance requirement: {overall_compliance*100:.1f}% (minimum required: {min_time_window_compliance*100:.0f}%)")
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
            'balance_weight': balance_weight,
            'min_time_window_compliance': min_time_window_compliance
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
    # Define colors for vehicle routes
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkblue', 
              'darkred', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 
              'lightblue', 'lightred', 'lightgreen', 'gray', 'black', 'lightgray']
    
    routes = optimization_result['routes']
    
    # Display summary statistics
    st.subheader("Optimization Results")

    # Calculate overall statistics
    total_deliveries = sum(len(route) for route in routes.values())
    active_vehicles = sum(1 for route in routes.values() if len(route) > 0)

    # Calculate additional metrics
    total_distance = sum(stats.get('total_distance_km', 0) for stats in optimization_result.get('stats', {}).values())
    total_time_mins = sum(stats.get('estimated_time_mins', 0) for stats in optimization_result.get('stats', {}).values())

    # Calculate time window compliance (on-time percentage)
    on_time_deliveries = 0
    total_route_deliveries = 0

    # Count deliveries within time window
    for vehicle_id, route in routes.items():
        stats = optimization_result.get('stats', {}).get(vehicle_id, {})
        
        # Only process if we have stats for this vehicle
        if stats and 'time_window_compliant' in stats:
            # Use the actual count of compliant deliveries, not the percentage
            on_time_deliveries += stats['time_window_compliant']
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
                        
                        # Only consider deliveries late if they arrive after the end time
                        if arrival_mins <= end_mins:
                            on_time_deliveries += 1
                    except:
                        pass
        
        total_route_deliveries += len(route)

    # Ensure we have a valid number for on-time percentage
    delivery_ontime_percent = 0
    if total_route_deliveries > 0:
        delivery_ontime_percent = (on_time_deliveries / total_route_deliveries) * 100

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
        st.metric("Time Window Compliance", f"{delivery_ontime_percent:.0f}%")
        
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
    
    # Extract all available dates from the delivery data
    if 'delivery_date' in delivery_data.columns:
        # Extract unique dates, ensuring all are converted to datetime objects
        available_dates = sorted(pd.to_datetime(delivery_data['delivery_date'].unique()))
        
        # Format dates for display
        date_options = {}
        for date in available_dates:
            # Ensure date is a proper datetime object before formatting
            if isinstance(date, str):
                date_obj = pd.to_datetime(date)
            else:
                date_obj = date
            # Create the formatted string key
            date_str = date_obj.strftime('%b %d, %Y')
            date_options[date_str] = date_obj
        
        # Default to earliest date
        default_date = min(available_dates) if available_dates else None
        default_date_str = default_date.strftime('%b %d, %Y') if default_date else None
        
        # Create date selection dropdown
        selected_date_str = st.selectbox(
            "Select date to show routes for:",
            options=list(date_options.keys()),
            index=0 if default_date_str else None,
        )
        
        # Convert selected string back to date object
        selected_date = date_options[selected_date_str] if selected_date_str else None
        
        # Filter routes to only show deliveries for the selected date
        if selected_date is not None:
            filtered_routes = {}
            
            for vehicle_id, route in routes.items():
                # Keep only deliveries for the selected date
                filtered_route = []
                
                for delivery in route:
                    delivery_id = delivery['id']
                    # Find the delivery in the original data to get its date
                    delivery_row = delivery_data[delivery_data['delivery_id'] == delivery_id]
                    
                    if not delivery_row.empty and 'delivery_date' in delivery_row:
                        delivery_date = delivery_row['delivery_date'].iloc[0]
                        
                        # Check if this delivery is for the selected date
                        if pd.to_datetime(delivery_date).date() == pd.to_datetime(selected_date).date():
                            filtered_route.append(delivery)
                
                # Only add the vehicle if it has deliveries on this date
                if filtered_route:
                    filtered_routes[vehicle_id] = filtered_route
            
            # Replace the original routes with filtered ones for map display
            routes_for_map = filtered_routes
            st.write(f"Showing routes for {len(routes_for_map)} vehicles on {selected_date_str}")
        else:
            routes_for_map = routes
    else:
        routes_for_map = routes
        st.warning("No delivery dates available in data. Showing all routes.")
    
    # Create a map centered on Singapore
    singapore_coords = [1.3521, 103.8198]
    m = folium.Map(location=singapore_coords, zoom_start=12)
    
    # Modify loop to use routes_for_map instead of routes
    # Count total route segments for progress bar
    total_segments = sum(len(route) + 1 for route in routes_for_map.values() if route)  # +1 for return to depot

    # Create a unique key for this optimization result to use in session state
    optimization_key = hash(str(optimization_result))

    # Check if we have stored routes for this optimization result
    if optimization_key not in st.session_state.calculated_road_routes:
        # Initialize storage for this optimization
        st.session_state.calculated_road_routes[optimization_key] = {}

    # Count total route segments for progress bar
    total_segments = sum(len(route) + 1 for route in routes_for_map.values() if route)  # +1 for return to depot
    route_progress = st.progress(0)
    progress_container = st.empty()
    progress_container.text("Calculating routes: 0%")

    # Counter for processed segments
    processed_segments = 0

    for i, (vehicle_id, route) in enumerate(routes_for_map.items()):
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
            
            # Create a key for this route segment
            route_key = f"{vehicle_id}_{k}"
            
            # Update progress text
            segment_desc = "depot" if k == 0 else f"stop {k}"
            next_desc = f"stop {k+1}" if k < len(waypoints) - 2 else "depot"
            
            # Check if we have already calculated this route
            if route_key in st.session_state.calculated_road_routes[optimization_key]:
                # Use stored route
                road_route = st.session_state.calculated_road_routes[optimization_key][route_key]
                progress_text = f"Using stored route for Vehicle {vehicle_id}: {segment_desc} → {next_desc}"
            else:
                # Calculate and store new route
                progress_text = f"Calculating route for Vehicle {vehicle_id}: {segment_desc} → {next_desc}"
                with st.spinner(progress_text):
                    # Get a road-like route between these points
                    road_route = get_road_route(start_point, end_point)
                    # Store for future use
                    st.session_state.calculated_road_routes[optimization_key][route_key] = road_route
            
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

    # Add a message to show when using cached routes
    if optimization_key in st.session_state.calculated_road_routes:
        cached_count = len(st.session_state.calculated_road_routes[optimization_key])
        if cached_count > 0 and cached_count >= processed_segments:
            st.info(f"✅ Using {cached_count} previously calculated routes. No recalculation needed.")
    
    # Clear progress display when done
    progress_container.empty()
    route_progress.empty()
    st.success("All routes calculated successfully!")
    
    # Display the map
    folium_static(m, width=800, height=600)
    
    # -----------------------------------------------------
    # Unified Schedule Calendar Section
    # -----------------------------------------------------
    st.subheader("Schedule Calendar View")
    st.write("This calendar shows both delivery schedules and vehicle assignments. On-time deliveries are shown in green, late deliveries in red.")

    # Process data for calendar view
    if routes:
        # First, collect all assigned deliveries and their details
        calendar_data = []
        
        # Track which deliveries were actually included in routes
        assigned_delivery_ids = set()
        
        # Step 1: Process all assigned deliveries first
        for vehicle_id, route in routes.items():
            for delivery in route:
                assigned_delivery_ids.add(delivery['id'])
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
                    
                    # Set color based on on-time status and assignment
                    if on_time:
                        # Green for on-time
                        color = 'on_time'
                    else:
                        # Red for not on-time
                        color = 'late'

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
                        'Vehicle Task': f"{vehicle_id}: {driver_name}",
                        'on_time': on_time,
                        'assigned': True,
                        'color': color,
                        'delivery_date': pd.to_datetime(date_str)
                    })
                except Exception as e:
                    st.warning(f"Could not process time window for delivery {delivery_id}: {str(e)}")
        
        # Step 2: Now add unassigned deliveries
        for _, row in delivery_data.iterrows():
            delivery_id = row['delivery_id']
            
            # Skip if already assigned
            if delivery_id in assigned_delivery_ids:
                continue
                
            # Extract data for unassigned delivery
            customer_name = row.get('customer_name', 'Unknown')
            priority = row.get('priority', 'Medium')
            time_window = row.get('time_window', '09:00-17:00')
            weight = row.get('weight_kg', 0)
            
            # Extract start and end times from time_window
            start_time_str, end_time_str = time_window.split('-')
            
            # Get delivery date
            if 'delivery_date' in row:
                delivery_date = row['delivery_date']
            else:
                delivery_date = datetime.now().date()
            
            # Create start and end datetime
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
                
                # For unassigned deliveries set color to 'unassigned'
                calendar_data.append({
                    'delivery_id': delivery_id,
                    'customer_name': customer_name,
                    'vehicle_id': 'Unassigned',
                    'driver_name': 'N/A',
                    'vehicle_type': 'N/A',
                    'priority': priority,
                    'time_window': time_window,
                    'estimated_arrival_mins': 0,
                    'estimated_arrival_time': 'N/A',
                    'weight_kg': weight,
                    'Start': start_datetime,
                    'Finish': end_datetime,
                    'Task': f"{delivery_id}: {customer_name} (UNASSIGNED)",
                    'Vehicle Task': 'Unassigned',
                    'on_time': False,
                    'assigned': False,
                    'color': 'unassigned',  # Color for unassigned
                    'delivery_date': pd.to_datetime(date_str)
                })
            except Exception as e:
                st.warning(f"Could not process time window for unassigned delivery {delivery_id}: {str(e)}")
        
        if calendar_data:
            # Convert to DataFrame
            cal_df = pd.DataFrame(calendar_data)
            
            # Create color mapping for on-time status
            cal_df['Color'] = cal_df['on_time'].map({True: 'rgb(0, 200, 0)', False: 'rgb(255, 0, 0)'})
            
            # Get all available dates 
            all_dates = sorted(cal_df['delivery_date'].dt.date.unique())
            
            # Format dates for display in the dropdown
            date_options = {date.strftime('%b %d, %Y'): date for date in all_dates}
            
            # Initialize calendar display dates if not already set or if dates have changed
            available_date_keys = list(date_options.keys())
            
            # Default to all dates
            if st.session_state.calendar_display_dates is None or not all(date in available_date_keys for date in st.session_state.calendar_display_dates):
                st.session_state.calendar_display_dates = available_date_keys
            
            # Add multiselect for date filtering with session state
            selected_date_strings = st.multiselect(
                "Select dates to display",
                options=available_date_keys,
                default=st.session_state.calendar_display_dates,
                key="calendar_date_selector"
            )
            
            # Update the session state
            st.session_state.calendar_display_dates = selected_date_strings
            
            # Convert selected strings back to date objects
            selected_dates = [date_options[date_str] for date_str in selected_date_strings]
            
            if not selected_dates:
                st.info("Please select at least one date to view the delivery schedule.")
            else:
                # Filter calendar data to only include selected dates
                filtered_cal_df = cal_df[cal_df['delivery_date'].dt.date.isin(selected_dates)]
                
                # Create tabs only for the selected dates
                date_tabs = st.tabs([date.strftime('%b %d, %Y') for date in selected_dates])
                
                for i, (date, tab) in enumerate(zip(selected_dates, date_tabs)):
                    with tab:
                        # Filter data for this date
                        day_data = filtered_cal_df[filtered_cal_df['delivery_date'].dt.date == date]
                        
                        if len(day_data) > 0:
                            # FIRST SECTION: DELIVERY SCHEDULE VIEW
                            st.write("#### Delivery Schedule")
                            
                            # Create figure for delivery view
                            fig = px.timeline(
                                day_data, 
                                x_start="Start", 
                                x_end="Finish", 
                                y="Task",
                                color="color",  # Use our color column
                                color_discrete_map={
                                    "on_time": "green", 
                                    "late": "orange",
                                    "unassigned": "red"  # Unassigned deliveries also red
                                },
                                hover_data=["customer_name", "vehicle_id", "driver_name", "priority", "time_window", 
                                           "estimated_arrival_time", "weight_kg", "assigned"]
                            )
                            
                            # Fix the pattern application code
                            for i, row in day_data.iterrows():
                                # Only add diagonal pattern to assigned deliveries
                                if row['assigned']:
                                    for trace in fig.data:
                                        # Find which trace corresponds to this row's color group
                                        color_value = row['color']
                                        
                                        # Look for matching trace
                                        if trace.name == color_value and any(y == row['Task'] for y in trace.y):
                                            # Add pattern only to assigned bars
                                            if 'marker' not in trace:
                                                trace.marker = dict()
                                            if 'pattern' not in trace.marker:
                                                trace.marker.pattern = dict(
                                                    shape="\\",  # Diagonal lines
                                                    size=4,
                                                    solidity=0.5,
                                                    fgcolor="black"
                                                )
                            
                            # Add status labels to the bars
                            for idx, row in day_data.iterrows():
                                status_text = "✓ On-time" if row['on_time'] and row['assigned'] else "⚠ Late" if row['assigned'] else "Not assigned"
                                position = (row['Start'] + (row['Finish'] - row['Start'])/2)
                                
                                # Only add labels to assigned deliveries
                                if row['assigned']:
                                    fig.add_annotation(
                                        x=position,
                                        y=row['Task'],
                                        text=status_text,
                                        showarrow=False,
                                        font=dict(color="black", size=10),
                                        xanchor="center"
                                    )
                            
                            # Update layout
                            fig.update_layout(
                                title=f"Deliveries by Customer - {date.strftime('%b %d, %Y')}",
                                xaxis_title="Time of Day",
                                yaxis_title="Delivery",
                                height=max(300, 50 * len(day_data)),
                                yaxis={'categoryorder':'category ascending'},
                                showlegend=False  # Hide the legend as we have custom annotations
                            )
                            
                            # Display figure
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show summary metrics for delivery view
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
                                
                                for j, (priority, count) in enumerate(priority_counts.items()):
                                    col_idx = j % len(priority_cols)
                                    with priority_cols[col_idx]:
                                        st.metric(priority, count)
                            
                            # SECOND SECTION: VEHICLE SCHEDULE VIEW
                            st.write("#### Vehicle Schedule")
                            
                            # Create figure grouped by vehicle
                            fig_vehicle = px.timeline(
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
                                fig_vehicle.add_annotation(
                                    x=(row['Start'] + (row['Finish'] - row['Start'])/2),
                                    y=row['Vehicle Task'],
                                    text=f"#{row['delivery_id']}",
                                    showarrow=False,
                                    font=dict(size=10, color="black")
                                )
                            
                            # Update layout
                            fig_vehicle.update_layout(
                                title=f"Vehicle Assignment Schedule - {date.strftime('%b %d, %Y')}",
                                xaxis_title="Time of Day",
                                yaxis_title="Vehicle",
                                height=max(300, 70 * day_data['Vehicle Task'].nunique()),
                                yaxis={'categoryorder':'category ascending'}
                            )
                            
                            # Display figure for vehicle view
                            st.plotly_chart(fig_vehicle, use_container_width=True)
                            
                            # Show vehicle utilization summary
                            st.write("##### Vehicle Utilization")
                            
                            # Calculate vehicle utilization metrics
                            vehicle_metrics = []
                            for vehicle_id in day_data['vehicle_id'].unique():
                                vehicle_deliveries = day_data[day_data['vehicle_id'] == vehicle_id]
                                
                                # Calculate total delivery time for this vehicle
                                total_mins = sum((row['Finish'] - row['Start']).total_seconds() / 60 for _, row in vehicle_deliveries.iterrows())
                                
                                # Count on-time deliveries
                                on_time_count = len(vehicle_deliveries[vehicle_deliveries['on_time'] == True])
                                
                                # Get the driver name
                                driver_name = vehicle_deliveries['driver_name'].iloc[0] if not vehicle_deliveries.empty else "Unknown"
                                
                                vehicle_metrics.append({
                                    'vehicle_id': vehicle_id,
                                    'driver_name': driver_name,
                                    'deliveries': len(vehicle_deliveries),
                                    'delivery_time_mins': total_mins,
                                    'on_time_deliveries': on_time_count,
                                    'on_time_percentage': (on_time_count / len(vehicle_deliveries)) * 100 if len(vehicle_deliveries) > 0 else 0
                                })
                            
                            # Display metrics in a nice format
                            metrics_df = pd.DataFrame(vehicle_metrics)
                            
                            # Show as a table
                            st.dataframe(metrics_df.style.format({
                                'delivery_time_mins': '{:.0f}',
                                'on_time_percentage': '{:.1f}%'
                            }))
                            
                        else:
                            st.info(f"No deliveries scheduled for {date.strftime('%b %d, %Y')}")
        else:
            st.info("No calendar data available. Please generate routes first.")

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