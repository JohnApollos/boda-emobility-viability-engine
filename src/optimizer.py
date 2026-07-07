"""
Spatial Optimization Module for BSS Cabinets.

Implements grid-based coverage gap identification, sample-weighted K-Means
clustering for facility location optimization, and multi-criteria scoring indexes.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURATION CONSTANTS ---
EARTH_RADIUS_KM: float = 6371.0

# Nairobi bounding box and municipal boundary config
NAIROBI_BOUNDS_LAT: Tuple[float, float] = (-1.38, -1.18)
NAIROBI_BOUNDS_LON: Tuple[float, float] = 36.68, 36.98
GRID_RESOLUTION: int = 35  # Produces a 35x35 grid of candidate points
MUNICIPAL_BOUNDARY_RADIUS_KM: float = 8.0

# Scoring index coordinates & metadata
INDUSTRIAL_GRID_ANCHOR: Tuple[float, float] = (-1.3000, 36.8600)  # Makadara
CBD_GRID_ANCHOR: Tuple[float, float] = (-1.2750, 36.8250)         # Starehe

ROAD_CORRIDORS: List[Tuple[float, float]] = [
    (-1.222, 36.885),  # Thika Road
    (-1.300, 36.762),  # Ngong Road
    (-1.332, 36.875),  # Mombasa Road
    (-1.275, 36.895),  # Outer Ring Road
    (-1.285, 36.822)   # CBD Center
]

# Multi-criteria scoring weights
WEIGHT_RIDER_POTENTIAL: float = 0.40
WEIGHT_CONNECTIVITY: float = 0.30
WEIGHT_GRID_STABILITY: float = 0.15
WEIGHT_ROAD_ACCESS: float = 0.15

class OptimizationError(Exception):
    """Raised when spatial optimization cannot be completed due to invalid grid states."""
    pass


def haversine_distance_vectorized(lat1: float, lon1: float, lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """Calculates distances in km between a coordinate pair and a vector of coordinate pairs.

    Args:
        lat1 (float): Target latitude in decimal degrees.
        lon1 (float): Target longitude in decimal degrees.
        lats2 (np.ndarray): Array of comparison latitudes.
        lons2 (np.ndarray): Array of comparison longitudes.

    Returns:
        np.ndarray: Vector of calculated great-circle distances in kilometers.
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lats2_rad, lons2_rad = np.radians(lats2), np.radians(lons2)
    
    dlat = lats2_rad - lat1_rad
    dlon = lons2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lats2_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


class NetworkOptimizer:
    """Handles GIS grid generation, gap detection, and optimized location suggestion for battery swaps."""

    def __init__(self) -> None:
        """Initializes empty spatial dataframes."""
        self.subcounties: Optional[pd.DataFrame] = None
        self.existing_stations: Optional[pd.DataFrame] = None
        self.grid_points: Optional[pd.DataFrame] = None
        
    def load_data(
        self, 
        subcounty_path: str = "data/nairobi_subcounties.csv", 
        stations_path: str = "data/existing_stations.csv"
    ) -> None:
        """Loads physical datasets and generates candidate spatial grids.

        Args:
            subcounty_path (str): Relative path to Nairobi subcounties CSV.
            stations_path (str): Relative path to existing stations CSV.
        """
        try:
            self.subcounties = pd.read_csv(subcounty_path)
            self.existing_stations = pd.read_csv(stations_path)
        except Exception as e:
            logger.error("Spatial optimizer data loading failed: %s", e)
            raise FileNotFoundError(f"Failed to load spatial data sources. Ensure datasets exist. Error: {e}")
            
        self._generate_nairobi_grid()
        logger.info("Spatial Optimizer loaded successfully with %d subcounties and %d existing stations.", 
                    len(self.subcounties), len(self.existing_stations))
        
    def _generate_nairobi_grid(self) -> None:
        """Generates a candidate grid filtering out points outside municipal bounds."""
        if self.subcounties is None:
            raise OptimizationError("Generate grid called before subcounty metadata was loaded.")
            
        lat_grid = np.linspace(NAIROBI_BOUNDS_LAT[0], NAIROBI_BOUNDS_LAT[1], GRID_RESOLUTION)
        lon_grid = np.linspace(NAIROBI_BOUNDS_LON[0], NAIROBI_BOUNDS_LON[1], GRID_RESOLUTION)
        
        points: List[Dict[str, Any]] = []
        
        for lat in lat_grid:
            for lon in lon_grid:
                # Find nearest subcounty centroid to filter out peripheral points
                dists = [haversine_distance_vectorized(lat, lon, sc["lat"], sc["lon"]) for _, sc in self.subcounties.iterrows()]
                min_idx = int(np.argmin(dists))
                min_dist = float(dists[min_idx])
                
                # Check municipal boundary threshold
                if min_dist <= MUNICIPAL_BOUNDARY_RADIUS_KM:
                    nearest_sc = self.subcounties.iloc[min_idx]
                    points.append({
                        "lat": lat,
                        "lon": lon,
                        "nearest_subcounty": str(nearest_sc["name"]),
                        "density": float(nearest_sc["density"]),
                        "boda_factor": float(nearest_sc["boda_factor"]),
                        "weight": float(nearest_sc["density"] * nearest_sc["boda_factor"])
                    })
        self.grid_points = pd.DataFrame(points)
        logger.info("Generated spatial candidate grid containing %d populated points.", len(self.grid_points))
        
    def find_coverage_gaps(self, threshold_km: float = 5.0) -> pd.DataFrame:
        """Flags grid points located outside the target operational coverage radius.

        Args:
            threshold_km (float): Radius threshold (in km) to define coverage.

        Returns:
            pd.DataFrame: Table of unserved grid coordinates with distance descriptors.
        """
        if self.grid_points is None or self.existing_stations is None:
            self.load_data()
            
        # Re-check to satisfy type checkers
        if self.grid_points is None or self.existing_stations is None:
            raise OptimizationError("Failed to initialize spatial tables.")
            
        gap_points: List[Dict[str, Any]] = []
        for _, pt in self.grid_points.iterrows():
            dists = haversine_distance_vectorized(
                pt["lat"], pt["lon"], 
                self.existing_stations["lat"].values, 
                self.existing_stations["lon"].values
            )
            min_dist = float(np.min(dists))
            
            if min_dist > threshold_km:
                pt_dict = pt.to_dict()
                pt_dict["min_dist_to_existing"] = min_dist
                gap_points.append(pt_dict)
                
        return pd.DataFrame(gap_points)

    def recommend_stations(self, n_stations: int = 10, gap_threshold_km: float = 5.0) -> pd.DataFrame:
        """Optimizes BSS placement in identified coverage gaps using sample-weighted K-Means.

        Args:
            n_stations (int): Number of facilities to recommend.
            gap_threshold_km (float): Gap radius trigger in kilometers.

        Returns:
            pd.DataFrame: Scored and ranked facility recommendations.
        """
        df_gaps = self.find_coverage_gaps(threshold_km=gap_threshold_km)
        
        if len(df_gaps) == 0:
            logger.info("Zero coverage gaps detected at the %s km threshold.", gap_threshold_km)
            return pd.DataFrame()
            
        n_clusters = min(int(n_stations), len(df_gaps))
        
        X = df_gaps[["lat", "lon"]]
        weights = df_gaps["weight"]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X, sample_weight=weights)
        
        centers = kmeans.cluster_centers_
        recommended_locations: List[Dict[str, Any]] = []
        
        for i, center in enumerate(centers):
            c_lat, c_lon = float(center[0]), float(center[1])
            
            # 1. Calculate Rider Potential Score
            # Sum of decaying densities within a 7km operational sweep
            rider_pot = 0.0
            if self.subcounties is not None:
                for _, sc in self.subcounties.iterrows():
                    d = float(haversine_distance_vectorized(c_lat, c_lon, sc["lat"], sc["lon"]))
                    if d < 1.0:
                        d = 1.0  # Cap distance scaling to prevent asymptotic divide by zero
                    if d <= 7.0:
                        rider_pot += (sc["density"] * sc["boda_factor"]) / d
            
            # Normalize to 0-100 scale based on standard observed bounds
            rider_score = min(100.0, (rider_pot / 40000.0) * 100)
            
            # 2. Grid Stability Score (Proximity to industrial feeders)
            dist_to_industrial = float(haversine_distance_vectorized(c_lat, c_lon, INDUSTRIAL_GRID_ANCHOR[0], INDUSTRIAL_GRID_ANCHOR[1]))
            dist_to_cbd = float(haversine_distance_vectorized(c_lat, c_lon, CBD_GRID_ANCHOR[0], CBD_GRID_ANCHOR[1]))
            
            closest_grid_dist = min(dist_to_industrial, dist_to_cbd)
            if closest_grid_dist <= 4.0:
                grid_stability = 0.95
            elif closest_grid_dist <= 8.0:
                grid_stability = 0.88
            else:
                grid_stability = 0.80
            grid_score = grid_stability * 100
            
            # 3. Network Connectivity Score (Distance to other existing stations)
            # Penalizes overlaps (<3km) and isolates (>8km)
            if self.existing_stations is not None:
                dists_existing = haversine_distance_vectorized(
                    c_lat, c_lon, 
                    self.existing_stations["lat"].values, 
                    self.existing_stations["lon"].values
                )
                min_existing_dist = float(np.min(dists_existing))
            else:
                min_existing_dist = 10.0
            
            if 3.0 <= min_existing_dist <= 8.0:
                connectivity_score = 100.0
            elif min_existing_dist < 3.0:
                connectivity_score = (min_existing_dist / 3.0) * 100.0
            else:
                connectivity_score = max(0.0, 100.0 - (min_existing_dist - 8.0) * 10.0)
                
            # 4. Major Road Corridor Proximity Score
            min_road_dist = min([float(haversine_distance_vectorized(c_lat, c_lon, clat, clon)) for clat, clon in ROAD_CORRIDORS])
            road_score = max(40.0, 100.0 - (min_road_dist * 12.0))
            
            # Multi-Criteria Decision Score
            overall_score = (
                (WEIGHT_RIDER_POTENTIAL * rider_score) + 
                (WEIGHT_CONNECTIVITY * connectivity_score) + 
                (WEIGHT_GRID_STABILITY * grid_score) + 
                (WEIGHT_ROAD_ACCESS * road_score)
            )
            
            # Associate nearest administrative subcounty name
            sc_dists = []
            if self.subcounties is not None:
                sc_dists = [haversine_distance_vectorized(c_lat, c_lon, sc["lat"], sc["lon"]) for _, sc in self.subcounties.iterrows()]
                nearest_sc_name = str(self.subcounties.iloc[np.argmin(sc_dists)]["name"])
            else:
                nearest_sc_name = "Nairobi City"
            
            recommended_locations.append({
                "Rec_ID": f"REC-{i+1:02d}",
                "Latitude": np.round(c_lat, 5),
                "Longitude": np.round(c_lon, 5),
                "Subcounty": nearest_sc_name,
                "Rider_Demand_Score": np.round(rider_score, 1),
                "Grid_Stability_Score": np.round(grid_score, 1),
                "Connectivity_Score": np.round(connectivity_score, 1),
                "Road_Access_Score": np.round(road_score, 1),
                "Overall_Viability_Score": np.round(overall_score, 1),
                "Distance_to_Existing_km": np.round(min_existing_dist, 2),
                "Grid_Stability_Factor": grid_stability
            })
            
        df_rec = pd.DataFrame(recommended_locations)
        return df_rec.sort_values(by="Overall_Viability_Score", ascending=False).reset_index(drop=True)

    def evaluate_expansion_impact(self, df_recs: pd.DataFrame) -> Dict[str, float]:
        """Calculates pre/post demographics improvements after hypothetical station rollout.

        Args:
            df_recs (pd.DataFrame): Table of proposed station placements.

        Returns:
            Dict[str, float]: Key metrics containing coverage before, after, average distance and improvement.
        """
        if self.grid_points is None or self.existing_stations is None:
            raise OptimizationError("Data has not been loaded before expansion impact evaluation.")
            
        if len(df_recs) == 0:
            new_lats = np.array([])
            new_lons = np.array([])
        else:
            new_lats = df_recs["Latitude"].values
            new_lons = df_recs["Longitude"].values
        
        all_lats = np.concatenate([self.existing_stations["lat"].values, new_lats])
        all_lons = np.concatenate([self.existing_stations["lon"].values, new_lons])
        
        coverage_before: List[int] = []
        coverage_after: List[int] = []
        dists_before: List[float] = []
        dists_after: List[float] = []
        
        for _, pt in self.grid_points.iterrows():
            # Before: Existing network distances
            d_ex = haversine_distance_vectorized(
                pt["lat"], pt["lon"], 
                self.existing_stations["lat"].values, 
                self.existing_stations["lon"].values
            )
            min_d_ex = float(np.min(d_ex))
            dists_before.append(min_d_ex)
            coverage_before.append(1 if min_d_ex <= 5.0 else 0)
            
            # After: Expanded network distances
            d_all = haversine_distance_vectorized(pt["lat"], pt["lon"], all_lats, all_lons)
            min_d_all = float(np.min(d_all))
            dists_after.append(min_d_all)
            coverage_after.append(1 if min_d_all <= 5.0 else 0)
            
        # Compile weighted percentages
        total_weight = self.grid_points["weight"].sum()
        pct_covered_before = (np.array(coverage_before) * self.grid_points["weight"]).sum() / total_weight * 100.0
        pct_covered_after = (np.array(coverage_after) * self.grid_points["weight"]).sum() / total_weight * 100.0
        
        return {
            "pct_covered_before": float(np.round(pct_covered_before, 1)),
            "pct_covered_after": float(np.round(pct_covered_after, 1)),
            "avg_dist_before_km": float(np.round(np.mean(dists_before), 2)),
            "avg_dist_after_km": float(np.round(np.mean(dists_after), 2)),
            "max_dist_before_km": float(np.round(np.max(dists_before), 2)),
            "max_dist_after_km": float(np.round(np.max(dists_after), 2)),
            "coverage_improvement_pct": float(np.round(pct_covered_after - pct_covered_before, 1))
        }

    def calculate_brand_coverage(self, threshold_km: float = 5.0) -> Dict[str, float]:
        """Calculates population-weighted coverage percentages for each brand network.

        Args:
            threshold_km (float): Radius defining coverage.

        Returns:
            Dict[str, float]: Key value pairs of brand names and coverage rates.
        """
        if self.grid_points is None or self.existing_stations is None:
            self.load_data()
            
        # Recheck for type checker
        if self.grid_points is None or self.existing_stations is None:
            raise OptimizationError("Failed to initialize spatial grid and existing stations.")
            
        brands = self.existing_stations["brand"].unique()
        coverage_by_brand: Dict[str, float] = {}
        total_weight = self.grid_points["weight"].sum()
        
        for brand in brands:
            brand_stations = self.existing_stations[self.existing_stations["brand"] == brand]
            covered_flags: List[int] = []
            for _, pt in self.grid_points.iterrows():
                dists = haversine_distance_vectorized(
                    pt["lat"], pt["lon"], 
                    brand_stations["lat"].values, 
                    brand_stations["lon"].values
                )
                min_dist = float(np.min(dists))
                covered_flags.append(1 if min_dist <= threshold_km else 0)
            
            coverage_pct = (np.array(covered_flags) * self.grid_points["weight"]).sum() / total_weight * 100.0
            coverage_by_brand[brand] = float(np.round(coverage_pct, 1))
            
        return coverage_by_brand

    def recalculate_rider_distances(self, df_recs: pd.DataFrame, riders_df: pd.DataFrame) -> np.ndarray:
        """Recalculates distances to nearest BSS for all riders with new locations.

        Args:
            df_recs (pd.DataFrame): Proposed stations to add.
            riders_df (pd.DataFrame): Dataframe of rider coordinates.

        Returns:
            np.ndarray: Vector of updated nearest station distances in kilometers.
        """
        if self.existing_stations is None:
            raise OptimizationError("Existing stations not loaded. Load data first.")
            
        if len(df_recs) == 0:
            return riders_df["Distance_to_BSS_km"].values
            
        new_lats = df_recs["Latitude"].values
        new_lons = df_recs["Longitude"].values
        
        all_lats = np.concatenate([self.existing_stations["lat"].values, new_lats])
        all_lons = np.concatenate([self.existing_stations["lon"].values, new_lons])
        
        new_distances: List[float] = []
        for _, rider in riders_df.iterrows():
            dists = haversine_distance_vectorized(rider["Latitude"], rider["Longitude"], all_lats, all_lons)
            new_distances.append(float(np.min(dists)))
            
        return np.round(new_distances, 2)


if __name__ == "__main__":
    import os
    if os.path.exists("data/nairobi_subcounties.csv") and os.path.exists("data/existing_stations.csv"):
        opt = NetworkOptimizer()
        opt.load_data()
        gaps = opt.find_coverage_gaps()
        logger.info("Total grid points in coverage gaps (>5km): %d", len(gaps))
        
        recs = opt.recommend_stations(n_stations=10)
        logger.info("Top Recommended Locations:\n%s", recs.head(5))
        
        impact = opt.evaluate_expansion_impact(recs)
        logger.info("Expansion Impact:\n%s", impact)
    else:
        logger.warning("Spatial data files not found. Execute data_processor.py first.")
