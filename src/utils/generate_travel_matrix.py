import pandas as pd
import numpy as np
import os
import time
import requests
from math import radians, sin, cos, sqrt, atan2
import random

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points in kilometers.
    The Haversine distance is the great-circle distance between two points on a sphere.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Coordinates of the first point in decimal degrees
    lat2, lon2 : float
        Coordinates of the second point in decimal degrees
        
    Returns:
    --------
    float
        Distance between the two points in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # Radius of Earth in kilometers
    
    return distance

def get_road_distance_with_retry(origin, destination, max_retries=3, initial_backoff=1):
    """
    Get road distance between two points with retry logic
    
    Parameters:
    -----------
    origin : dict
        Origin location with 'latitude' and 'longitude' keys
    destination : dict
        Destination location with 'latitude' and 'longitude' keys
    max_retries : int
        Maximum number of retry attempts
    initial_backoff : int
        Initial backoff time in seconds
    
    Returns:
    --------
    tuple of (float, float)
        Distance in km and duration in minutes
    """
    # URLs for different public OSRM instances to distribute load
    osrm_urls = [
        "http://router.project-osrm.org",
        "https://routing.openstreetmap.de",
        # Add more public OSRM servers if available
    ]
    
    retry_count = 0
    backoff = initial_backoff
    
    while retry_count < max_retries:
        try:
            # Use a random OSRM server from the list to distribute load
            base_url = random.choice(osrm_urls)
            url = f"{base_url}/route/v1/driving/{origin['longitude']},{origin['latitude']};{destination['longitude']},{destination['latitude']}?overview=false"
            
            # Add a timeout to prevent hanging connections
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data.get('code') == 'Ok':
                # Extract distance and duration
                distance = data['routes'][0]['distance'] / 1000  # meters to km
                duration = data['routes'][0]['duration'] / 60    # seconds to minutes
                return round(distance, 2), round(duration, 2)
            else:
                print(f"API returned error: {data.get('message', 'Unknown error')}")
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retry {retry_count+1}/{max_retries}")
        
        # Exponential backoff with jitter to prevent thundering herd
        jitter = random.uniform(0, 0.5 * backoff)
        sleep_time = backoff + jitter
        time.sleep(sleep_time)
        backoff *= 2  # Exponential backoff
        retry_count += 1
    
    # Fallback to haversine after all retries failed
    print(f"All retries failed for route from ({origin['latitude']},{origin['longitude']}) to ({destination['latitude']},{destination['longitude']}). Using haversine distance.")
    distance = haversine_distance(
        origin['latitude'], origin['longitude'],
        destination['latitude'], destination['longitude']
    )
    distance = distance * 1.3  # Road factor
    time_mins = (distance / 40) * 60  # 40 km/h
    
    return round(distance, 2), round(time_mins, 2)

def get_road_distance(origins, destinations, use_osrm=True):
    """
    Calculate actual road distances and travel times between multiple origins and destinations
    using the OSRM (Open Source Routing Machine) API.
    
    Parameters:
    -----------
    origins : list of dict
        List of origin locations with 'latitude' and 'longitude' keys
    destinations : list of dict
        List of destination locations with 'latitude' and 'longitude' keys
    use_osrm : bool, default=True
        Whether to use OSRM API or fall back to haversine distance
        
    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray)
        Arrays containing distances (in km) and durations (in minutes) between each origin-destination pair
    """
    n_origins = len(origins)
    n_destinations = len(destinations)
    distance_matrix = np.zeros((n_origins, n_destinations))
    duration_matrix = np.zeros((n_origins, n_destinations))
    
    # If OSRM is not requested, fall back to haversine distance
    if not use_osrm:
        print("Using haversine distance as fallback.")
        for i, origin in enumerate(origins):
            for j, dest in enumerate(destinations):
                distance = haversine_distance(
                    origin['latitude'], origin['longitude'],
                    dest['latitude'], dest['longitude']
                )
                # Adjust for road networks (roads are typically not straight lines)
                distance = distance * 1.3  # Apply a factor to approximate road distance
                time_mins = (distance / 40) * 60  # Assuming average speed of 40 km/h
                
                distance_matrix[i, j] = round(distance, 2)
                duration_matrix[i, j] = round(time_mins, 2)
        return distance_matrix, duration_matrix
    
    # Process in batches to prevent overwhelming the API
    print(f"Processing {n_origins} origins and {n_destinations} destinations in batches...")
    total_requests = n_origins * n_destinations
    completed = 0
    
    try:
        # Try OSRM's table service for small datasets first (more efficient)
        if n_origins + n_destinations <= 50:
            print("Trying OSRM table API for efficient matrix calculation...")
            try:
                # Code for table API would go here, but we'll skip for now as it's more complex
                # and the batch approach is more reliable for handling errors
                raise NotImplementedError("Table API not implemented, falling back to individual routes")
            except Exception as e:
                print(f"Table API failed: {e}. Using individual routes instead.")
                # Continue with individual route requests below
        
        # Process with individual route requests
        for i, origin in enumerate(origins):
            for j, dest in enumerate(destinations):
                # Skip if origin and destination are the same point
                if i == j:
                    distance_matrix[i, j] = 0
                    duration_matrix[i, j] = 0
                    completed += 1
                    continue
                
                # Get distance with retry logic
                distance, duration = get_road_distance_with_retry(origin, dest)
                distance_matrix[i, j] = distance
                duration_matrix[i, j] = duration
                
                # Show progress
                completed += 1
                if completed % 10 == 0:
                    print(f"Progress: {completed}/{total_requests} routes calculated ({(completed/total_requests)*100:.1f}%)")
                
                # Add randomized delay to prevent overwhelming the API
                time.sleep(random.uniform(0.1, 0.5))
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Saving partial results...")
    
    return distance_matrix, duration_matrix

def generate_travel_matrix(use_osrm=True):
    """
    Generate travel time and distance matrices between all locations in the delivery problem.
    
    Parameters:
    -----------
    use_osrm : bool, default=True
        Whether to use OSRM API for real road distances instead of haversine
        
    Returns:
    --------
    tuple of (pd.DataFrame, pd.DataFrame, dict)
        Distance matrix, base time matrix, and hourly time matrices
    """
    # Create data directories if they don't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    time_matrix_dir = os.path.join(data_dir, 'time-matrix')
    delivery_data_dir = os.path.join(data_dir, 'delivery-data')    
    vehicle_data_dir = os.path.join(data_dir, 'vehicle-data')
    
    # Ensure all directories exist
    for directory in [time_matrix_dir, delivery_data_dir, vehicle_data_dir]:
        os.makedirs(directory, exist_ok=True)
     
    # Read delivery and vehicle data
    try:
        delivery_data = pd.read_csv(os.path.join(delivery_data_dir, 'delivery_data.csv'))
        vehicle_data = pd.read_csv(os.path.join(vehicle_data_dir, 'vehicle_data.csv'))
    except FileNotFoundError:
        print("Error: Please generate delivery and vehicle data first!")
        return
    
    # Extract locations
    delivery_locations = delivery_data[['delivery_id', 'latitude', 'longitude']].values
    depot_locations = vehicle_data[['vehicle_id', 'depot_latitude', 'depot_longitude']].values
    
    # Average speed for time calculation (km/h)
    avg_speed = vehicle_data['avg_speed_kmh'].mean()
    
    # Traffic factor matrix (to simulate traffic conditions at different times)
    hours_in_day = 24
    traffic_factors = np.ones((hours_in_day, 1))
    
    # Simulate morning rush hour (8-10 AM)
    traffic_factors[8:10] = 1.5
    
    # Simulate evening rush hour (5-7 PM)
    traffic_factors[17:19] = 1.8
    
    # Late night (less traffic)
    traffic_factors[22:] = 0.8
    traffic_factors[:5] = 0.7
    
    # Create a combined list of all locations (depots + delivery points)
    all_locations = []
    
    # Add depot locations
    for row in depot_locations:
        all_locations.append({
            'id': row[0],  # vehicle_id as location id
            'type': 'depot',
            'latitude': row[1],
            'longitude': row[2]
        })
    
    # Add delivery locations
    for row in delivery_locations:
        all_locations.append({
            'id': row[0],  # delivery_id as location id
            'type': 'delivery',
            'latitude': row[1],
            'longitude': row[2]
        })
    
    print(f"Calculating distances between {len(all_locations)} locations...")
    
    # Save the locations file early so we have this data even if the process is interrupted
    location_df = pd.DataFrame(all_locations)
    location_df.to_csv(os.path.join(time_matrix_dir, 'locations.csv'), index=False)
    
    # Calculate distances and times using OSRM with improved error handling
    if use_osrm:
        print("Using OSRM API for road distances...")
        distance_matrix, base_time_matrix = get_road_distance(all_locations, all_locations, use_osrm=True)
    else:
        print("Using haversine distance with road factor adjustment...")
        distance_matrix, base_time_matrix = get_road_distance(all_locations, all_locations, use_osrm=False)
    
    # Create DataFrames for the matrices
    location_ids = [loc['id'] for loc in all_locations]
    
    distance_df = pd.DataFrame(distance_matrix, index=location_ids, columns=location_ids)
    time_df = pd.DataFrame(base_time_matrix, index=location_ids, columns=location_ids)
    
    # Save distance and base time matrices early in case later steps fail
    distance_df.to_csv(os.path.join(time_matrix_dir, 'distance_matrix.csv'))
    time_df.to_csv(os.path.join(time_matrix_dir, 'base_time_matrix.csv'))
    print("Basic distance and time matrices saved successfully.")
    
    # Create time matrices for different hours of the day
    hourly_time_matrices = {}
    for hour in range(24):
        traffic_factor = traffic_factors[hour][0]
        hourly_time = base_time_matrix * traffic_factor
        hourly_time_matrices[f"{hour:02d}:00"] = pd.DataFrame(hourly_time, index=location_ids, columns=location_ids)
    
    # Save a sample of time matrices (e.g., rush hour and normal time)
    try:
        hourly_time_matrices['08:00'].to_csv(os.path.join(time_matrix_dir, 'morning_rush_time_matrix.csv'))
        hourly_time_matrices['18:00'].to_csv(os.path.join(time_matrix_dir, 'evening_rush_time_matrix.csv'))
        hourly_time_matrices['12:00'].to_csv(os.path.join(time_matrix_dir, 'midday_time_matrix.csv'))
        hourly_time_matrices['00:00'].to_csv(os.path.join(time_matrix_dir, 'night_time_matrix.csv'))
        print("Time matrices for different hours saved successfully.")
    except Exception as e:
        print(f"Error saving hourly time matrices: {e}")
        print("Continuing with basic matrices only.")
    
    print("Travel matrices generation complete.")
    return distance_df, time_df, hourly_time_matrices

if __name__ == "__main__":
    # For development, allow falling back to haversine if needed
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate travel matrices for delivery route optimization")
    parser.add_argument("--use-osrm", action="store_true", help="Use OSRM API for real road distances")
    parser.add_argument("--use-haversine", action="store_true", help="Use haversine distance only (faster)")
    
    args = parser.parse_args()
    
    if args.use_haversine:
        generate_travel_matrix(use_osrm=False)
    else:
        # Default to OSRM unless explicitly disabled
        generate_travel_matrix(use_osrm=True) 