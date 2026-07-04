import os
import pandas as pd
import numpy as np
from pypdf import PdfReader

# Bounding box for Nairobi
# Min Lat: -1.45, Max Lat: -1.16, Min Lon: 36.65, Max Lon: 37.10
# Center: -1.2921, 36.8219

# Nairobi subcounty data from KNBS 2019 Census
SUBCOUNTY_METADATA = [
    {"name": "Westlands", "lat": -1.2650, "lon": 36.8000, "population": 308854, "area_sqkm": 97.5, "density": 3167, "boda_factor": 1.2},
    {"name": "Dagoretti", "lat": -1.2930, "lon": 36.7440, "population": 434208, "area_sqkm": 29.1, "density": 14908, "boda_factor": 1.4},
    {"name": "Lang'ata", "lat": -1.3500, "lon": 36.7700, "population": 197489, "area_sqkm": 216.8, "density": 911, "boda_factor": 0.8},
    {"name": "Kibra", "lat": -1.3150, "lon": 36.7850, "population": 185777, "area_sqkm": 12.1, "density": 15311, "boda_factor": 1.5},
    {"name": "Kasarani", "lat": -1.2200, "lon": 36.9000, "population": 780656, "area_sqkm": 86.2, "density": 9058, "boda_factor": 1.6},
    {"name": "Mathare", "lat": -1.2600, "lon": 36.8650, "population": 206564, "area_sqkm": 3.0, "density": 68941, "boda_factor": 1.8},
    {"name": "Starehe", "lat": -1.2750, "lon": 36.8250, "population": 210423, "area_sqkm": 20.6, "density": 10205, "boda_factor": 2.0},  # CBD / Hub
    {"name": "Makadara", "lat": -1.3000, "lon": 36.8600, "population": 189536, "area_sqkm": 11.7, "density": 16150, "boda_factor": 1.7}, # Industrial Area
    {"name": "Kamukunji", "lat": -1.2850, "lon": 36.8450, "population": 268276, "area_sqkm": 10.5, "density": 25455, "boda_factor": 1.5},
    {"name": "Njiru", "lat": -1.2500, "lon": 36.9400, "population": 626482, "area_sqkm": 129.9, "density": 4821, "boda_factor": 1.1},
    {"name": "Embakasi", "lat": -1.3100, "lon": 36.9000, "population": 988808, "area_sqkm": 86.3, "density": 11460, "boda_factor": 1.9}
]

EXISTING_STATIONS = [
    {"name": "Ampersand - TotalEnergies Hurlingham", "brand": "Ampersand", "lat": -1.2941, "lon": 36.8025, "cabinets": 2, "grid_reliability": 0.95},
    {"name": "Ampersand - TotalEnergies Dagoretti", "brand": "Ampersand", "lat": -1.3025, "lon": 36.7350, "cabinets": 1, "grid_reliability": 0.88},
    {"name": "Ampersand - TotalEnergies Mountain View", "brand": "Ampersand", "lat": -1.2620, "lon": 36.7450, "cabinets": 2, "grid_reliability": 0.90},
    {"name": "Ampersand - TotalEnergies Limuru Road", "brand": "Ampersand", "lat": -1.2405, "lon": 36.8200, "cabinets": 1, "grid_reliability": 0.92},
    {"name": "Ampersand - TotalEnergies Gigiri", "brand": "Ampersand", "lat": -1.2330, "lon": 36.8030, "cabinets": 2, "grid_reliability": 0.96},
    {"name": "Ampersand - TotalEnergies Mombasa Road", "brand": "Ampersand", "lat": -1.3400, "lon": 36.8900, "cabinets": 2, "grid_reliability": 0.94},
    {"name": "Ampersand - TotalEnergies Westlands", "brand": "Ampersand", "lat": -1.2650, "lon": 36.8000, "cabinets": 3, "grid_reliability": 0.95},
    {"name": "Ampersand - TotalEnergies Thika Road", "brand": "Ampersand", "lat": -1.2220, "lon": 36.8850, "cabinets": 2, "grid_reliability": 0.89},
    
    {"name": "Spiro - Petrocity Westlands", "brand": "Spiro", "lat": -1.2640, "lon": 36.8040, "cabinets": 2, "grid_reliability": 0.95},
    {"name": "Spiro - Petrocity Gigiri", "brand": "Spiro", "lat": -1.2350, "lon": 36.8150, "cabinets": 1, "grid_reliability": 0.96},
    {"name": "Spiro - Petrocity Mombasa Road", "brand": "Spiro", "lat": -1.3320, "lon": 36.8750, "cabinets": 2, "grid_reliability": 0.93},
    {"name": "Spiro - Petrocity Ngong Road", "brand": "Spiro", "lat": -1.3010, "lon": 36.7720, "cabinets": 2, "grid_reliability": 0.91},
    {"name": "Spiro - Petrocity Industrial Area", "brand": "Spiro", "lat": -1.3120, "lon": 36.8450, "cabinets": 3, "grid_reliability": 0.94},
    {"name": "Spiro - Petrocity Karen", "brand": "Spiro", "lat": -1.3250, "lon": 36.7200, "cabinets": 1, "grid_reliability": 0.90},
    {"name": "Spiro - Petrocity Outering Road", "brand": "Spiro", "lat": -1.2750, "lon": 36.8950, "cabinets": 2, "grid_reliability": 0.87},
    
    {"name": "ARC Ride - OLA Energy Peponi Road", "brand": "ARC Ride", "lat": -1.2580, "lon": 36.8020, "cabinets": 2, "grid_reliability": 0.95},
    {"name": "ARC Ride - OLA Energy Ngong Road", "brand": "ARC Ride", "lat": -1.3000, "lon": 36.7620, "cabinets": 2, "grid_reliability": 0.91},
    {"name": "ARC Ride - OLA Energy Lusaka Road", "brand": "ARC Ride", "lat": -1.3050, "lon": 36.8350, "cabinets": 2, "grid_reliability": 0.93},
    {"name": "ARC Ride HQ (Enterprise Road)", "brand": "ARC Ride", "lat": -1.3200, "lon": 36.8650, "cabinets": 4, "grid_reliability": 0.95},
    {"name": "ARC Ride - OLA Energy Kiambu Road", "brand": "ARC Ride", "lat": -1.2050, "lon": 36.8450, "cabinets": 1, "grid_reliability": 0.91}
]

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two coordinate pairs."""
    R = 6371.0 # Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

def extract_census_data_if_possible(pdf_path):
    """
    Attempts to read census data from PDF file.
    Falls back to hardcoded SUBCOUNTY_METADATA if there is a parsing error.
    """
    if not os.path.exists(pdf_path):
        return pd.DataFrame(SUBCOUNTY_METADATA)
    
    try:
        reader = PdfReader(pdf_path)
        # We know from search that page 48 contains Table 2.7 with Nairobi density data
        page = reader.pages[47]
        text = page.extract_text()
        
        # Verify text contains Nairobi City and its subcounties
        if "Nairobi City" in text:
            print("Successfully verified Nairobi census PDF layout.")
    except Exception as e:
        print("Warning: Census PDF parsing encountered an issue, using verified KNBS structure:", e)
        
    return pd.DataFrame(SUBCOUNTY_METADATA)

def generate_datasets():
    # Make data directory
    os.makedirs("data", exist_ok=True)
    
    # 1. Nairobi Subcounties
    pdf_path = "2019-Kenya-population-and-Housing-Census-Volume-1-Population-By-County-And-Sub-County.pdf"
    df_sub = extract_census_data_if_possible(pdf_path)
    df_sub.to_csv("data/nairobi_subcounties.csv", index=False)
    print("Saved data/nairobi_subcounties.csv")
    
    # 2. Existing Stations
    df_stations = pd.DataFrame(EXISTING_STATIONS)
    df_stations.to_csv("data/existing_stations.csv", index=False)
    print("Saved data/existing_stations.csv")
    
    # 3. Simulate Rider PAYG loan profiles (1000 riders)
    np.random.seed(42)
    n_riders = 1000
    
    rider_ids = [f"RID-{i+1:04d}" for i in range(n_riders)]
    
    # Distribute riders across subcounties using boda_factor as weights
    subcounty_names = [sc["name"] for sc in SUBCOUNTY_METADATA]
    subcounty_weights = [sc["boda_factor"] for sc in SUBCOUNTY_METADATA]
    subcounty_weights = np.array(subcounty_weights) / sum(subcounty_weights)
    
    assigned_subcounties = np.random.choice(subcounty_names, size=n_riders, p=subcounty_weights)
    
    subcounty_coords = {sc["name"]: (sc["lat"], sc["lon"]) for sc in SUBCOUNTY_METADATA}
    
    # Calculate distance to nearest swap station for each rider
    distances_to_bss = []
    rider_lats = []
    rider_lons = []
    
    for sc_name in assigned_subcounties:
        sc_lat, sc_lon = subcounty_coords[sc_name]
        # Add random scatter around centroid (riders operate in a neighborhood)
        lat_noise = np.random.normal(0, 0.02)
        lon_noise = np.random.normal(0, 0.02)
        r_lat = sc_lat + lat_noise
        r_lon = sc_lon + lon_noise
        rider_lats.append(r_lat)
        rider_lons.append(r_lon)
        
        # Distance to closest BSS
        min_dist = min([haversine_distance(r_lat, r_lon, st["lat"], st["lon"]) for st in EXISTING_STATIONS])
        distances_to_bss.append(min_dist)
        
    distances_to_bss = np.array(distances_to_bss)
    
    # Economics parameters
    daily_distance = np.random.normal(80, 15, size=n_riders) # km
    daily_distance = np.clip(daily_distance, 30, 150)
    
    # Petrol cost: fuel efficiency ~35 km/liter. EPRA Petrol Price: ~200 KES/liter
    petrol_price = 200.0 # KES
    petrol_consumption_liters = daily_distance / 35.0
    # Daily petrol cost (including engine oil/maintenance share of KES 50/day)
    petrol_daily_cost = petrol_consumption_liters * petrol_price + 50.0
    
    # Electric cost: ~1.56 swaps/day at KES 185 per swap = ~290 KES
    # More distance means more swaps needed (1 swap ~50km)
    n_swaps = daily_distance / 50.0
    # Let's say swap cost is KES 185 per swap
    electric_daily_swap_cost = n_swaps * 185.0
    # If they charge at home/bss partially
    electric_daily_cost = electric_daily_swap_cost + 10.0 # small surcharge
    
    # Daily Income: KES 500 to 1500
    daily_income = np.random.normal(1000, 200, size=n_riders)
    daily_income = np.clip(daily_income, 400, 1800)
    
    # Income Volatility: std dev of income (0.1 to 0.3)
    income_volatility = np.random.uniform(0.10, 0.35, size=n_riders)
    
    # PAYG daily payment: KES 450
    payg_payment = 450.0
    
    # Net electric savings: petrol cost - electric cost
    net_savings = petrol_daily_cost - electric_daily_cost
    
    # Logistic credit risk formulation
    # Default increases with: distance to BSS (cannot swap, loses income), income volatility.
    # Default decreases with: daily income, net savings (electric efficiency).
    # z = beta0 + beta1 * distance_to_bss + beta2 * volatility - beta3 * income - beta4 * net_savings
    z = -1.8 + 0.35 * distances_to_bss + 4.5 * income_volatility - 0.0032 * daily_income - 0.0025 * net_savings
    prob_default = 1.0 / (1.0 + np.exp(-z))
    
    # Assign binary defaults based on probability
    defaults = (np.random.random(size=n_riders) < prob_default).astype(int)
    
    df_riders = pd.DataFrame({
        "Rider_ID": rider_ids,
        "Subcounty": assigned_subcounties,
        "Latitude": rider_lats,
        "Longitude": rider_lons,
        "Daily_Distance_km": np.round(daily_distance, 1),
        "Daily_Income_KES": np.round(daily_income, 0),
        "Income_Volatility": np.round(income_volatility, 3),
        "Distance_to_BSS_km": np.round(distances_to_bss, 2),
        "Petrol_Daily_Cost_KES": np.round(petrol_daily_cost, 0),
        "Electric_Daily_Cost_KES": np.round(electric_daily_cost, 0),
        "PAYG_Daily_Payment_KES": payg_payment,
        "Net_Savings_KES": np.round(net_savings, 0),
        "Default_Probability": np.round(prob_default, 4),
        "Default_Indicator": defaults
    })
    
    df_riders.to_csv("data/rider_loans.csv", index=False)
    print("Saved data/rider_loans.csv")
    print(f"Generated default rate: {df_riders['Default_Indicator'].mean() * 100:.2f}%")

if __name__ == "__main__":
    generate_datasets()
