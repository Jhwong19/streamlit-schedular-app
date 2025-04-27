import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import time
import requests
from geopy.geocoders import Nominatim

# Set random seed for reproducibility
np.random.seed(42)

def generate_delivery_data(n_deliveries=50, use_geocoding=False):
    """
    Generate synthetic delivery data with realistic Singapore addresses
    """
    # Define real Singapore neighborhoods and their actual streets
    # Format: [neighborhood_name, [list_of_real_streets], postal_code_prefix]
    sg_neighborhoods = [
        ['Ang Mo Kio', ['Ang Mo Kio Avenue 1', 'Ang Mo Kio Avenue 3', 'Ang Mo Kio Avenue 4', 'Ang Mo Kio Avenue 10'], '56'],
        ['Bedok', ['Bedok North Avenue 1', 'Bedok North Road', 'Bedok Reservoir Road', 'New Upper Changi Road'], '46'],
        ['Bishan', ['Bishan Street 11', 'Bishan Street 12', 'Bishan Street 13', 'Bishan Street 22'], '57'],
        ['Bukit Merah', ['Jalan Bukit Merah', 'Henderson Road', 'Tiong Bahru Road', 'Redhill Close'], '15'],
        ['Bukit Batok', ['Bukit Batok East Avenue 6', 'Bukit Batok West Avenue 8', 'Bukit Batok Street 21'], '65'],
        ['Clementi', ['Clementi Avenue 1', 'Clementi Avenue 4', 'Clementi Road', 'Commonwealth Avenue West'], '12'],
        ['Geylang', ['Geylang East Avenue 1', 'Geylang Road', 'Guillemard Road', 'Sims Avenue'], '38'],
        ['Hougang', ['Hougang Avenue 1', 'Hougang Avenue 7', 'Hougang Street 91', 'Upper Serangoon Road'], '53'],
        ['Jurong East', ['Jurong East Street 13', 'Jurong East Avenue 1', 'Jurong Gateway Road'], '60'],
        ['Jurong West', ['Jurong West Street 41', 'Jurong West Street 52', 'Jurong West Street 93'], '64'],
        ['Kallang', ['Kallang Avenue', 'Geylang Bahru', 'Boon Keng Road', 'Upper Boon Keng Road'], '33'],
        ['Punggol', ['Punggol Central', 'Punggol Field', 'Punggol Road', 'Punggol Way'], '82'],
        ['Queenstown', ['Commonwealth Avenue', 'Commonwealth Drive', 'Mei Chin Road', 'Stirling Road'], '14'],
        ['Sengkang', ['Sengkang East Way', 'Sengkang West Way', 'Compassvale Road', 'Fernvale Road'], '54'],
        ['Serangoon', ['Serangoon Avenue 2', 'Serangoon Avenue 3', 'Serangoon North Avenue 1'], '55'],
        ['Tampines', ['Tampines Street 11', 'Tampines Street 21', 'Tampines Avenue 1', 'Tampines Avenue 4'], '52'],
        ['Toa Payoh', ['Toa Payoh Lorong 1', 'Toa Payoh Lorong 2', 'Toa Payoh Lorong 4', 'Toa Payoh Central'], '31'],
        ['Woodlands', ['Woodlands Avenue 1', 'Woodlands Drive 16', 'Woodlands Drive 72', 'Woodlands Circle'], '73'],
        ['Yishun', ['Yishun Avenue 1', 'Yishun Avenue 4', 'Yishun Ring Road', 'Yishun Street 22'], '76']
    ]
    
    # Bounding boxes for neighborhoods (for fallback coordinates)
    # Format: [name, min_lat, max_lat, min_lon, max_lon]
    neighborhood_bounds = {
        'Ang Mo Kio': [1.360000, 1.380000, 103.830000, 103.860000],
        'Bedok': [1.320000, 1.335000, 103.920000, 103.950000],
        'Bishan': [1.345000, 1.360000, 103.830000, 103.855000],
        'Bukit Merah': [1.270000, 1.290000, 103.800000, 103.830000],
        'Bukit Batok': [1.340000, 1.360000, 103.740000, 103.770000],
        'Clementi': [1.310000, 1.325000, 103.750000, 103.780000],
        'Geylang': [1.310000, 1.325000, 103.880000, 103.900000],
        'Hougang': [1.370000, 1.385000, 103.880000, 103.900000],
        'Jurong East': [1.330000, 1.345000, 103.730000, 103.750000],
        'Jurong West': [1.340000, 1.360000, 103.690000, 103.720000],
        'Kallang': [1.300000, 1.320000, 103.850000, 103.880000],
        'Punggol': [1.390000, 1.410000, 103.900000, 103.920000],
        'Queenstown': [1.290000, 1.310000, 103.780000, 103.805000],
        'Sengkang': [1.380000, 1.395000, 103.870000, 103.900000],
        'Serangoon': [1.345000, 1.360000, 103.865000, 103.885000],
        'Tampines': [1.345000, 1.365000, 103.930000, 103.960000],
        'Toa Payoh': [1.326000, 1.341000, 103.840000, 103.865000],
        'Woodlands': [1.430000, 1.450000, 103.770000, 103.800000],
        'Yishun': [1.410000, 1.430000, 103.820000, 103.850000]
    }
    
    # Generate delivery IDs
    delivery_ids = [f'DEL{str(i).zfill(4)}' for i in range(1, n_deliveries + 1)]
    
    # Generate customer names (fictional)
    first_names = ['Tan', 'Lim', 'Lee', 'Ng', 'Wong', 'Chan', 'Goh', 'Ong', 'Teo', 'Koh', 
                   'Chua', 'Loh', 'Yeo', 'Sim', 'Ho', 'Ang', 'Tay', 'Yap', 'Leong', 'Foo']
    last_names = ['Wei', 'Ming', 'Hui', 'Ling', 'Yong', 'Jun', 'Hong', 'Xin', 'Yi', 'Jie',
                  'Cheng', 'Kai', 'Zhi', 'Tian', 'Yu', 'En', 'Yang', 'Hao', 'Chong', 'Zheng']
    customer_names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_deliveries)]
    
    addresses = []
    postal_codes = []
    latitudes = []
    longitudes = []
    neighborhood_names = []
    
    # Initialize geocoder if using geocoding
    if use_geocoding:
        geolocator = Nominatim(user_agent="delivery_app")
    
    # Generate realistic addresses
    for i in range(n_deliveries):
        # Randomly select a neighborhood
        neighborhood_data = random.choice(sg_neighborhoods)
        neighborhood = neighborhood_data[0]
        streets = neighborhood_data[1]
        postal_prefix = neighborhood_data[2]
        
        # Randomly select a street in that neighborhood
        street = random.choice(streets)
        
        # Generate block number (realistic for HDB)
        block = random.randint(100, 600)
        
        # Generate unit number
        unit_floor = random.randint(2, 20)
        unit_number = random.randint(1, 150)
        
        # Generate postal code (with realistic prefix)
        postal_suffix = str(random.randint(0, 999)).zfill(3)
        postal_code = postal_prefix + postal_suffix
        
        # Create two formats of address - one for display, one for geocoding
        display_address = f"Block {block}, #{unit_floor:02d}-{unit_number:03d}, {street}, Singapore {postal_code}"
        geocode_address = f"{block} {street}, Singapore {postal_code}"  # Simpler format for geocoding
        
        # Default coordinates from neighborhood bounding box (fallback)
        bounds = neighborhood_bounds[neighborhood]
        default_lat = round(random.uniform(bounds[0], bounds[1]), 6)
        default_lon = round(random.uniform(bounds[2], bounds[3]), 6)
        
        # Use geocoding API if requested
        if use_geocoding:
            try:
                location = geolocator.geocode(geocode_address)
                
                if location:
                    lat = location.latitude
                    lon = location.longitude
                    print(f"✓ Successfully geocoded: {geocode_address} → ({lat}, {lon})")
                else:
                    # First fallback: try with just street and postal code
                    simpler_address = f"{street}, Singapore {postal_code}"
                    location = geolocator.geocode(simpler_address)
                    
                    if location:
                        lat = location.latitude
                        lon = location.longitude
                        print(f"✓ Fallback geocoded: {simpler_address} → ({lat}, {lon})")
                    else:
                        # Second fallback: just use the neighborhood center
                        lat = default_lat
                        lon = default_lon
                        print(f"✗ Could not geocode: {geocode_address}, using neighborhood coordinates")
                
                # Add delay to avoid being rate limited
                time.sleep(1)
                
            except Exception as e:
                print(f"✗ Geocoding error for {geocode_address}: {str(e)}")
                lat = default_lat
                lon = default_lon
        else:
            # Without geocoding, use the default coordinates
            lat = default_lat
            lon = default_lon
            
        addresses.append(display_address)
        postal_codes.append(postal_code)
        latitudes.append(lat)
        longitudes.append(lon)
        neighborhood_names.append(neighborhood)
    
    # Generate delivery dates (within the next 7 days)
    base_date = datetime.now().date()
    delivery_dates = [base_date + timedelta(days=random.randint(1, 7)) for _ in range(n_deliveries)]
    
    # Generate time windows (between 9 AM and 5 PM)
    time_windows = []
    for _ in range(n_deliveries):
        start_hour = random.randint(9, 16)
        window_length = random.choice([1, 2, 3])  # 1, 2, or 3 hour windows
        end_hour = min(start_hour + window_length, 18)
        
        start_time = f"{start_hour:02d}:00"
        end_time = f"{end_hour:02d}:00"
        time_windows.append(f"{start_time}-{end_time}")
    
    # Generate package details
    weights = np.random.uniform(0.5, 20.0, n_deliveries)  # in kg
    volumes = np.random.uniform(0.01, 0.5, n_deliveries)  # in m³
    
    # Priority levels
    priorities = np.random.choice(['High', 'Medium', 'Low'], n_deliveries, 
                                 p=[0.2, 0.5, 0.3])  # 20% High, 50% Medium, 30% Low
    
    # Required vehicle type
    vehicle_types = np.random.choice(['Standard', 'Large', 'Refrigerated'], n_deliveries,
                                   p=[0.7, 0.2, 0.1])
    
    # Status
    statuses = np.random.choice(['Pending', 'Assigned', 'In Transit', 'Delivered'], n_deliveries,
                              p=[0.6, 0.2, 0.15, 0.05])
    
    # Additional notes
    notes = []
    special_instructions = [
        'Call customer before delivery', 
        'Fragile items', 
        'Leave at door',
        'Signature required',
        'No delivery on weekends',
        None
    ]
    
    for _ in range(n_deliveries):
        if random.random() < 0.7:  # 70% chance of having a note
            notes.append(random.choice(special_instructions))
        else:
            notes.append(None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'delivery_id': delivery_ids,
        'customer_name': customer_names,
        'address': addresses,
        'postal_code': postal_codes,
        'neighborhood': neighborhood_names,
        'latitude': latitudes,
        'longitude': longitudes,
        'delivery_date': delivery_dates,
        'time_window': time_windows,
        'weight_kg': weights.round(2),
        'volume_m3': volumes.round(3),
        'priority': priorities,
        'vehicle_type': vehicle_types,
        'status': statuses,
        'special_instructions': notes
    })
    
    # Ensure the directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'delivery-data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(data_dir, 'delivery_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Delivery data generated and saved to {output_path}")
    return df

if __name__ == "__main__":
    # Set to True if you want to use real geocoding (slower but more accurate)
    USE_GEOCODING = True
    delivery_data = generate_delivery_data(50, use_geocoding=USE_GEOCODING)
    print("Sample of delivery data:")
    print(delivery_data.head())

