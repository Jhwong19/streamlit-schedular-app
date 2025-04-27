import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def create_data_directory():
    """
    Ensure data directories exist for all generated files.
    
    This function creates the necessary directory structure to store
    delivery data, vehicle data, and travel time matrices.
    
    Returns:
    --------
    tuple of (str, str, str)
        Paths to time matrix directory, vehicle data directory, and delivery data directory    
    """
    vehicle_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'vehicle_data')
    os.makedirs(vehicle_data_dir, exist_ok=True)

    delivery_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'delivery_data')
    os.makedirs(delivery_data_dir, exist_ok=True)

    time_matrix_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'time_matrix')
    os.makedirs(time_matrix_data_dir, exist_ok=True)
    return time_matrix_data_dir, vehicle_data_dir, delivery_data_dir

def main():
    """
    Run all data generation scripts for the delivery route optimization project.
    
    This function orchestrates the creation of all synthetic datasets needed for
    the route optimization problem, including delivery data, vehicle data, and
    travel time/distance matrices.
    
    Generated Files:
    --------------
    1. Delivery Data:
       - Contains information about delivery locations, time windows, packages, etc.
       - Used to define the delivery stops in the routing problem.
       
    2. Vehicle Data:
       - Contains information about the delivery fleet, capacity, depots, etc.
       - Used to define the available resources for delivery routes.
       
    3. Travel Matrices:
       - Contains distance and time information between all locations.
       - Used by the optimization algorithm to calculate route costs.
       
    Usage:
    ------
    These generated datasets form the foundation of the delivery route optimization
    application. Together they define:
    - Where deliveries need to be made (delivery data)
    - What resources are available for deliveries (vehicle data)
    - How long it takes to travel between locations (travel matrices)
    
    The route optimization algorithm uses these inputs to determine the most
    efficient assignment of deliveries to vehicles and the optimal sequence of
    stops for each vehicle.    
    """
    print("Starting data generation process...")
    
    time_matrix_data_dir, vehicle_data_dir, delivery_data_dir = create_data_directory()
    print(f"Time Matrix Data will be saved to: {time_matrix_data_dir}")
    print(f"Delivery Data will be saved to: {delivery_data_dir}")
    print(f"Vehicle Data will be saved to: {vehicle_data_dir}")

    # Import and run delivery data generation
    print("\n1. Generating delivery data...")
    from src.utils.generate_delivery_data import generate_delivery_data
    delivery_data = generate_delivery_data(50, use_geocoding=True)
    
    # Import and run vehicle data generation
    print("\n2. Generating vehicle data...")
    from src.utils.generate_vehicle_data import generate_vehicle_data
    vehicle_data = generate_vehicle_data(10)
    
    # Import and run travel matrix generation
    print("\n3. Generating travel matrices...")
    from src.utils.generate_travel_matrix import generate_travel_matrix
    generate_travel_matrix()
    
    print("\nAll data generation complete! Files saved to data directory.")

if __name__ == "__main__":
    main()