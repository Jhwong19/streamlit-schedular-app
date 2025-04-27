import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(43)

def generate_vehicle_data(n_vehicles=10):
    """
    Generate synthetic vehicle data for a delivery fleet optimization problem.
    
    This function creates a realistic delivery fleet with various vehicle types,
    capacities, and operational parameters to be used in route optimization.
    
    Parameters:
    -----------
    n_vehicles : int, default=10
        Number of vehicles to generate in the fleet
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the generated vehicle data
    """
    
    # Vehicle IDs
    vehicle_ids = [f'VEH{str(i).zfill(3)}' for i in range(1, n_vehicles + 1)]
    
    # Vehicle types
    vehicle_types = []
    for _ in range(n_vehicles):
        vehicle_type = random.choices(['Standard', 'Large', 'Refrigerated'], 
                                     weights=[0.7, 0.2, 0.1])[0]
        vehicle_types.append(vehicle_type)
    
    # Vehicle capacities based on type
    max_weights = []
    max_volumes = []
    for v_type in vehicle_types:
        if v_type == 'Standard':
            max_weights.append(random.uniform(800, 1200))
            max_volumes.append(random.uniform(8, 12))
        elif v_type == 'Large':
            max_weights.append(random.uniform(1500, 2500))
            max_volumes.append(random.uniform(15, 25))
        else:  # Refrigerated
            max_weights.append(random.uniform(600, 1000))
            max_volumes.append(random.uniform(6, 10))
    
    # Realistic depot/warehouse locations in Singapore industrial areas
    # [name, latitude, longitude]
    warehouse_locations = [
        ["Tuas Logistics Hub", 1.3187, 103.6390],
        ["Jurong Industrial Estate", 1.3233, 103.6994],
        ["Loyang Industrial Park", 1.3602, 103.9761],
        ["Changi Logistics Centre", 1.3497, 103.9742],
        ["Keppel Distripark", 1.2706, 103.8219],
        ["Pandan Logistics Hub", 1.3187, 103.7509],
        ["Alexandra Distripark", 1.2744, 103.8012],
        ["Kallang Way Industrial", 1.3315, 103.8731],
        ["Defu Industrial Park", 1.3610, 103.8891],
        ["Woodlands Industrial", 1.4428, 103.7875]
    ]
    
    # Assign warehouses to vehicles (multiple vehicles can be from same warehouse)
    # Either assign sequentially to ensure all warehouses are used at least once (if n_vehicles >= len(warehouse_locations)),
    # or randomly select from the list
    depot_names = []
    depot_lats = []
    depot_lons = []
    
    if n_vehicles <= len(warehouse_locations):
        # Use first n_vehicles warehouses (one vehicle per warehouse)
        selected_warehouses = warehouse_locations[:n_vehicles]
    else:
        # Ensure every warehouse is used at least once
        selected_warehouses = warehouse_locations.copy()
        # Then add random ones for remaining vehicles
        remaining = n_vehicles - len(warehouse_locations)
        selected_warehouses.extend([random.choice(warehouse_locations) for _ in range(remaining)])
    
    # Shuffle to avoid sequential assignment
    random.shuffle(selected_warehouses)
    
    # Extract depot information
    for warehouse in selected_warehouses:
        depot_names.append(warehouse[0])
        depot_lats.append(warehouse[1])
        depot_lons.append(warehouse[2])
    
    # Add small variation for vehicles from the same warehouse (within warehouse compound)
    # This makes each vehicle's position slightly different, simulating different loading bays
    for i in range(len(depot_lats)):
        # Much smaller variation - within warehouse compound (approximately 50-100m variation)
        depot_lats[i] += random.uniform(-0.0005, 0.0005)
        depot_lons[i] += random.uniform(-0.0005, 0.0005)
    
    # Driver names
    first_names = ['Ahmad', 'Raj', 'Michael', 'Wei', 'Siti', 'Kumar', 'Chong', 'David', 'Suresh', 'Ali']
    last_names = ['Tan', 'Singh', 'Lee', 'Wong', 'Kumar', 'Abdullah', 'Zhang', 'Lim', 'Raj', 'Teo']
    driver_names = []
    
    for i in range(n_vehicles):
        if i < len(first_names):
            driver_names.append(f"{first_names[i]} {random.choice(last_names)}")
        else:
            driver_names.append(f"{random.choice(first_names)} {random.choice(last_names)}")
    
    # Vehicle availability
    start_times = [f"{random.randint(7, 10):02d}:00" for _ in range(n_vehicles)]
    end_times = [f"{random.randint(17, 21):02d}:00" for _ in range(n_vehicles)]
    
    # Max working hours
    max_working_hours = [random.randint(8, 10) for _ in range(n_vehicles)]
    
    # Average speed (km/h)
    avg_speeds = [random.uniform(30, 50) for _ in range(n_vehicles)]
    
    # Cost per km
    cost_per_km = [random.uniform(0.5, 1.5) for _ in range(n_vehicles)]
    
    # Vehicle status
    statuses = np.random.choice(['Available', 'In Service', 'Maintenance'], n_vehicles, p=[0.7, 0.2, 0.1])
    
    # License plates (Singapore format)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    license_plates = []
    for _ in range(n_vehicles):
        letter_part = ''.join(random.choices(letters, k=3))
        number_part = random.randint(1000, 9999)
        license_plates.append(f"S{letter_part}{number_part}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'vehicle_id': vehicle_ids,
        'vehicle_type': vehicle_types,
        'license_plate': license_plates,
        'driver_name': driver_names,
        'max_weight_kg': np.array(max_weights).round(2),
        'max_volume_m3': np.array(max_volumes).round(2),
        'depot_name': depot_names,
        'depot_latitude': np.array(depot_lats).round(6),
        'depot_longitude': np.array(depot_lons).round(6),
        'start_time': start_times,
        'end_time': end_times,
        'max_working_hours': max_working_hours,
        'avg_speed_kmh': np.array(avg_speeds).round(2),
        'cost_per_km': np.array(cost_per_km).round(2),
        'status': statuses
    })
    
    # Ensure the directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'vehicle-data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(data_dir, 'vehicle_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Vehicle data generated and saved to {output_path}")
    return df

if __name__ == "__main__":
    # Generate vehicle data
    vehicle_data = generate_vehicle_data(10)
    print("Sample of vehicle data:")
    print(vehicle_data.head())