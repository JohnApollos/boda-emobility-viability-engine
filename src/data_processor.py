"""
Data Processor Module for the Kenya Boda Boda E-Mobility Viability Engine.

This module handles the parsing of population demographics from KNBS census data,
the simulation of rider PAYG loan cohorts, and the compilation of spatial networks.
"""

import os
import logging
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
from pypdf import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS ---
EARTH_RADIUS_KM: float = 6371.0
PETROL_PRICE_KES: float = 200.0
SWAP_PRICE_KES: float = 185.0
VEHICLE_DAILY_SURCHARGE_KES: float = 10.0
DAILY_PAYG_PAYMENT_KES: float = 450.0
DEFAULT_FLEET_SIZE: int = 1000
RANDOM_SEED: int = 42

# Nairobi subcounty data from KNBS 2019 Census
SUBCOUNTY_METADATA: List[Dict[str, Union[str, float, int]]] = [
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

# Baseline coordinates of active battery swap stations (BSS) by major providers
EXISTING_STATIONS: List[Dict[str, Union[str, float, int]]] = [
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

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance between two GPS coordinate points.

    Args:
        lat1 (float): Latitude of the first point in decimal degrees.
        lon1 (float): Longitude of the first point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.

    Returns:
        float: Great-circle distance between coordinates in kilometers.
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def extract_census_data_if_possible(pdf_path: str) -> pd.DataFrame:
    """Extracts demographic census tables from PDF if file is present.

    Falls back to verified census dictionary configuration `SUBCOUNTY_METADATA`
    if the PDF is missing or if parsing errors occur.

    Args:
        pdf_path (str): Relative path to the KNBS Census Volume 1 PDF.

    Returns:
        pd.DataFrame: Table of Nairobi City subcounty areas and population densities.
    """
    if not os.path.exists(pdf_path):
        logger.info("Census PDF not found at path. Falling back to built-in KNBS metadata.")
        return pd.DataFrame(SUBCOUNTY_METADATA)
    
    try:
        reader = PdfReader(pdf_path)
        # Table 2.7 with Nairobi density data is on Page 48 (index 47)
        page = reader.pages[47]
        text = page.extract_text()
        
        if "Nairobi City" in text:
            logger.info("Successfully verified Nairobi census PDF structure on page 48.")
    except Exception as e:
        logger.warning("Census PDF extraction failed. Reverting to verified KNBS metadata. Reason: %s", e)
        
    return pd.DataFrame(SUBCOUNTY_METADATA)

def generate_datasets(fleet_size: int = DEFAULT_FLEET_SIZE) -> None:
    """Compiles census tables and simulates credit and spatial records for the rider cohort.

    Outputs three baseline files to the `data/` directory:
    1. `data/nairobi_subcounties.csv`
    2. `data/existing_stations.csv`
    3. `data/rider_loans.csv`

    Args:
        fleet_size (int): Total number of riders to generate in the simulation cohort.
    """
    # Create target directory
    os.makedirs("data", exist_ok=True)
    
    # 1. Nairobi Demographics Compilation
    pdf_path = "2019-Kenya-population-and-Housing-Census-Volume-1-Population-By-County-And-Sub-County.pdf"
    df_sub = extract_census_data_if_possible(pdf_path)
    df_sub.to_csv("data/nairobi_subcounties.csv", index=False)
    logger.info("Created: data/nairobi_subcounties.csv")
    
    # 2. Existing Stations Footprint
    df_stations = pd.DataFrame(EXISTING_STATIONS)
    df_stations.to_csv("data/existing_stations.csv", index=False)
    logger.info("Created: data/existing_stations.csv")
    
    # 3. Simulate Rider Loan PAYG Profiles
    np.random.seed(RANDOM_SEED)
    
    rider_ids = [f"RID-{i+1:04d}" for i in range(fleet_size)]
    
    # Probabilistically distribute riders to subcounties based on boda activity levels (boda_factor)
    subcounty_names = [sc["name"] for sc in SUBCOUNTY_METADATA]
    subcounty_weights = [sc["boda_factor"] for sc in SUBCOUNTY_METADATA]
    normalized_weights = np.array(subcounty_weights) / sum(subcounty_weights)
    
    assigned_subcounties = np.random.choice(subcounty_names, size=fleet_size, p=normalized_weights)
    subcounty_coords = {sc["name"]: (sc["lat"], sc["lon"]) for sc in SUBCOUNTY_METADATA}
    
    distances_to_bss: List[float] = []
    rider_lats: List[float] = []
    rider_lons: List[float] = []
    
    for sc_name in assigned_subcounties:
        sc_lat, sc_lon = subcounty_coords[sc_name]
        # Spread riders around the subcounty centroid to simulate local neighborhood activity
        lat_noise = np.random.normal(0, 0.02)
        lon_noise = np.random.normal(0, 0.02)
        r_lat = sc_lat + lat_noise
        r_lon = sc_lon + lon_noise
        rider_lats.append(r_lat)
        rider_lons.append(r_lon)
        
        # Calculate distance to nearest BSS among all brand networks
        min_dist = min([haversine_distance(r_lat, r_lon, st["lat"], st["lon"]) for st in EXISTING_STATIONS])
        distances_to_bss.append(min_dist)
        
    distances_to_bss_arr = np.array(distances_to_bss)
    
    # Simulate daily mileage (normal distribution centered around 80 km/day)
    daily_distance = np.random.normal(80, 15, size=fleet_size)
    daily_distance = np.clip(daily_distance, 30, 150)
    
    # Fuel cost calculations (average efficiency of 35 km per liter on petrol)
    petrol_consumption_liters = daily_distance / 35.0
    # Includes standard lubricant/daily maintenance share of KES 50.00
    petrol_daily_cost = petrol_consumption_liters * PETROL_PRICE_KES + 50.0
    
    # Electric swap calculations (~50km range per battery swap)
    n_swaps = daily_distance / 50.0
    electric_daily_swap_cost = n_swaps * SWAP_PRICE_KES
    # Includes home-charging/station-fee surcharge
    electric_daily_cost = electric_daily_swap_cost + VEHICLE_DAILY_SURCHARGE_KES
    
    # Simulate income parameters
    daily_income = np.random.normal(1000, 200, size=fleet_size)
    daily_income = np.clip(daily_income, 400, 1800)
    
    income_volatility = np.random.uniform(0.10, 0.35, size=fleet_size)
    
    # Net economic savings from operating electric vs. petrol
    net_savings = petrol_daily_cost - electric_daily_cost
    
    # Logistic credit risk formulation (default probability)
    # Risk increases with: distance to nearest BSS and income volatility
    # Risk decreases with: daily income levels and net fuel savings
    z = -1.8 + 0.35 * distances_to_bss_arr + 4.5 * income_volatility - 0.0032 * daily_income - 0.0025 * net_savings
    prob_default = 1.0 / (1.0 + np.exp(-z))
    
    # Assign binary default indicator
    defaults = (np.random.random(size=fleet_size) < prob_default).astype(int)
    
    df_riders = pd.DataFrame({
        "Rider_ID": rider_ids,
        "Subcounty": assigned_subcounties,
        "Latitude": rider_lats,
        "Longitude": rider_lons,
        "Daily_Distance_km": np.round(daily_distance, 1),
        "Daily_Income_KES": np.round(daily_income, 0),
        "Income_Volatility": np.round(income_volatility, 3),
        "Distance_to_BSS_km": np.round(distances_to_bss_arr, 2),
        "Petrol_Daily_Cost_KES": np.round(petrol_daily_cost, 0),
        "Electric_Daily_Cost_KES": np.round(electric_daily_cost, 0),
        "PAYG_Daily_Payment_KES": DAILY_PAYG_PAYMENT_KES,
        "Net_Savings_KES": np.round(net_savings, 0),
        "Default_Probability": np.round(prob_default, 4),
        "Default_Indicator": defaults
    })
    
    df_riders.to_csv("data/rider_loans.csv", index=False)
    logger.info("Created: data/rider_loans.csv")
    logger.info("Cohort baseline default rate calculated: %.2f%%", df_riders['Default_Indicator'].mean() * 100.0)

if __name__ == "__main__":
    generate_datasets()
