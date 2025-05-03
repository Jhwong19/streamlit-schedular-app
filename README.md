# Project: Delivery Route Optimization

![Delivery Route Network](img/delivery-route-network.jpg)

This project is a **Delivery Route Optimization** tool built using Streamlit. It aims to optimize delivery routes for a fleet of vehicles while considering constraints such as delivery time windows, vehicle capacity, and traffic conditions.


### Key Features
1. **Route Optimization**:
   - Solve the **Vehicle Routing Problem (VRP)** to determine the most efficient routes for a fleet of vehicles.
   - Incorporate constraints like:
     - Delivery time windows.
     - Vehicle capacity.
     - Traffic conditions.

2. **Map Visualization**:
   - Display optimized routes on an interactive map using **Folium**.
   - Highlight delivery stops, start and end points, and route distances.

3. **Calendar View**:
   - Provide a calendar-based schedule for deliveries.
   - Allow users to view and manage delivery schedules for specific days or weeks.

4. **Real-Time Updates**:
   - Enable real-time updates for route changes due to unexpected events (e.g., traffic congestion, vehicle breakdowns).
   - Re-optimize routes dynamically and update the map and calendar views.

### Tools and Technologies
- **Python**: Core programming language for optimization and application logic.
- **Google OR-Tools**: Solve the Vehicle Routing Problem (VRP) with constraints.
- **Streamlit**: Build an interactive web application for route visualization and schedule management.
- **Folium**: Create interactive maps for route visualization.
- **Synthetic Data**: Integrate real-time traffic data for dynamic route adjustments.

---

## Project Structure

```
streamlit-app-template
├── src
│   ├── app.py                # Main entry point of the Streamlit application
│   ├── components            # Directory for reusable UI components
│   │   └── __init__.py
│   ├── pages                 # Directory for different pages of the application
│   │   └── __init__.py
│   ├── utils                 # Directory for utility functions
│   │   └── __init__.py
├── requirements.txt          # List of dependencies for the application
├── .streamlit                # Configuration settings for Streamlit
│   ├── config.toml
├── img                       # Folder for storing images
│   └── delivery_route_network.png
├── .gitignore                # Files and directories to ignore in Git
├── README.md                 # Documentation for the project
└── LICENSE                   # Licensing information
```

---

## Installation

To get started with this Streamlit application template, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/streamlit-app-template.git
   cd streamlit-app-template
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

---

## Deployment

This application is deployed on Hugging Face Spaces and is accessible at:

🚀 [Delivery Route Optimization App](https://huggingface.co/spaces/Jing997/DeliveryRouteOptimisation)

The deployed version provides all the features of the local application without requiring any installation. You can:

- View and analyze optimized delivery routes
- Interact with the map visualization
- Explore the calendar view of scheduled deliveries
- Test different optimization parameters
- Filter deliveries by date, status, and priority

The app is automatically updated whenever changes are pushed to the connected GitHub repository.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.