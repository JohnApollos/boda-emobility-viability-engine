import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def haversine_distance_vectorized(lat1, lon1, lats2, lons2):
    """Vectorized calculation of distance in km between a point and an array of points."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lats2_rad, lons2_rad = np.radians(lats2), np.radians(lons2)
    
    dlat = lats2_rad - lat1_rad
    dlon = lons2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lats2_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

class NetworkOptimizer:
    def __init__(self):
        self.subcounties = None
        self.existing_stations = None
        self.grid_points = None
        
    def load_data(self, subcounty_path="data/nairobi_subcounties.csv", stations_path="data/existing_stations.csv"):
        self.subcounties = pd.read_csv(subcounty_path)
        self.existing_stations = pd.read_csv(stations_path)
        self._generate_nairobi_grid()
        
    def _generate_nairobi_grid(self):
        """Generate a grid of points covering the habitable area of Nairobi."""
        # Nairobi bounding box bounding coordinates
        lat_min, lat_max = -1.38, -1.18
        lon_min, lon_max = 36.68, 36.98
        
        # Create grid lines (30x30 = 900 points)
        lat_grid = np.linspace(lat_min, lat_max, 35)
        lon_grid = np.linspace(lon_min, lon_max, 35)
        
        points = []
        for lat in lat_grid:
            for lon in lon_grid:
                # Find nearest subcounty centroid to filter out points far outside Nairobi
                dists = [haversine_distance_vectorized(lat, lon, sc["lat"], sc["lon"]) for _, sc in self.subcounties.iterrows()]
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]
                
                # Only include points within 8km of any subcounty centroid to match municipal shape
                if min_dist <= 8.0:
                    nearest_sc = self.subcounties.iloc[min_idx]
                    points.append({
                        "lat": lat,
                        "lon": lon,
                        "nearest_subcounty": nearest_sc["name"],
                        "density": nearest_sc["density"],
                        "boda_factor": nearest_sc["boda_factor"],
                        "weight": nearest_sc["density"] * nearest_sc["boda_factor"]
                    })
        self.grid_points = pd.DataFrame(points)
        
    def find_coverage_gaps(self, threshold_km=5.0):
        """Identifies grid points where the distance to the nearest existing station exceeds the threshold."""
        if self.grid_points is None:
            self.load_data()
            
        gap_points = []
        for _, pt in self.grid_points.iterrows():
            dists = haversine_distance_vectorized(pt["lat"], pt["lon"], self.existing_stations["lat"], self.existing_stations["lon"])
            min_dist = np.min(dists)
            
            if min_dist > threshold_km:
                pt_dict = pt.to_dict()
                pt_dict["min_dist_to_existing"] = min_dist
                gap_points.append(pt_dict)
                
        return pd.DataFrame(gap_points)

    def recommend_stations(self, n_stations=10, gap_threshold_km=5.0):
        """Runs weighted K-Means on coverage gaps and scores/ranks cluster centers."""
        df_gaps = self.find_coverage_gaps(threshold_km=gap_threshold_km)
        
        if len(df_gaps) == 0:
            print("No coverage gaps found at threshold!")
            return pd.DataFrame()
            
        # Ensure we don't ask for more clusters than data points
        n_clusters = min(int(n_stations), len(df_gaps))
        
        X = df_gaps[["lat", "lon"]]
        weights = df_gaps["weight"]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X, sample_weight=weights)
        
        centers = kmeans.cluster_centers_
        
        recommended_locations = []
        for i, center in enumerate(centers):
            c_lat, c_lon = center[0], center[1]
            
            # 1. Rider potential: sum of (subcounty density / distance) for all subcounties within 7km
            rider_pot = 0
            for _, sc in self.subcounties.iterrows():
                d = haversine_distance_vectorized(c_lat, c_lon, sc["lat"], sc["lon"])
                # Apply smooth decay
                if d < 1.0:
                    d = 1.0
                if d <= 7.0:
                    rider_pot += sc["density"] * sc["boda_factor"] / d
            
            # Normalize rider potential to a score of 0-100 (assume max observed is ~50000)
            rider_score = min(100.0, (rider_pot / 40000.0) * 100)
            
            # 2. Grid stability: simulate based on latitude (industrial East/South is higher, peri-urban is lower)
            # Industrial Makadara is around -1.30, 36.86. Starehe CBD is -1.275, 36.825
            dist_to_industrial = haversine_distance_vectorized(c_lat, c_lon, -1.3000, 36.8600)
            dist_to_cbd = haversine_distance_vectorized(c_lat, c_lon, -1.2750, 36.8250)
            
            if min(dist_to_industrial, dist_to_cbd) <= 4.0:
                grid_stability = 0.95
            elif min(dist_to_industrial, dist_to_cbd) <= 8.0:
                grid_stability = 0.88
            else:
                grid_stability = 0.80
            grid_score = grid_stability * 100
            
            # 3. Network connectivity: distance to nearest existing station
            # We want it to be > 3km (avoid redundancy) and < 10km (not isolated)
            dists_existing = haversine_distance_vectorized(c_lat, c_lon, self.existing_stations["lat"], self.existing_stations["lon"])
            min_existing_dist = np.min(dists_existing)
            
            if 3.0 <= min_existing_dist <= 8.0:
                connectivity_score = 100.0
            elif min_existing_dist < 3.0:
                connectivity_score = (min_existing_dist / 3.0) * 100.0 # Penalities for overlap
            else:
                # Decays down as it gets further than 8km
                connectivity_score = max(0.0, 100.0 - (min_existing_dist - 8.0) * 10.0)
                
            # 4. Road Accessibility: simulate Major corridor proximity
            # Major roads: Thika Rd (-1.222, 36.885), Ngong Rd (-1.300, 36.762), Mombasa Rd (-1.332, 36.875), Outer Ring (-1.275, 36.895)
            corridors = [
                (-1.222, 36.885), # Thika Road
                (-1.300, 36.762), # Ngong Road
                (-1.332, 36.875), # Mombasa Road
                (-1.275, 36.895), # Outer Ring Road
                (-1.285, 36.822)  # CBD Center
            ]
            min_road_dist = min([haversine_distance_vectorized(c_lat, c_lon, clat, clon) for clat, clon in corridors])
            road_score = max(40.0, 100.0 - (min_road_dist * 12.0))
            
            # Overall multi-criteria score
            # 40% Rider potential, 30% Connectivity, 15% Grid, 15% Road
            overall_score = (0.40 * rider_score) + (0.30 * connectivity_score) + (0.15 * grid_score) + (0.15 * road_score)
            
            # Find nearest subcounty name
            sc_dists = [haversine_distance_vectorized(c_lat, c_lon, sc["lat"], sc["lon"]) for _, sc in self.subcounties.iterrows()]
            nearest_sc_name = self.subcounties.iloc[np.argmin(sc_dists)]["name"]
            
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

    def evaluate_expansion_impact(self, df_recs):
        """Simulates how adding the recommended stations improves coverage metrics."""
        if len(df_recs) == 0:
            new_lats = np.array([])
            new_lons = np.array([])
        else:
            # Combine existing and new recommended stations
            new_lats = df_recs["Latitude"].values
            new_lons = df_recs["Longitude"].values
        
        all_lats = np.concatenate([self.existing_stations["lat"].values, new_lats])
        all_lons = np.concatenate([self.existing_stations["lon"].values, new_lons])
        
        # Calculate coverage for all grid points
        coverage_before = []
        coverage_after = []
        dists_before = []
        dists_after = []
        
        for _, pt in self.grid_points.iterrows():
            # Before: distance to existing
            d_ex = haversine_distance_vectorized(pt["lat"], pt["lon"], self.existing_stations["lat"].values, self.existing_stations["lon"].values)
            min_d_ex = np.min(d_ex)
            dists_before.append(min_d_ex)
            coverage_before.append(1 if min_d_ex <= 5.0 else 0)
            
            # After: distance to existing + new
            d_all = haversine_distance_vectorized(pt["lat"], pt["lon"], all_lats, all_lons)
            min_d_all = np.min(d_all)
            dists_after.append(min_d_all)
            coverage_after.append(1 if min_d_all <= 5.0 else 0)
            
        # Calculate population-weighted coverage percentages
        total_weight = self.grid_points["weight"].sum()
        pct_covered_before = (np.array(coverage_before) * self.grid_points["weight"]).sum() / total_weight * 100.0
        pct_covered_after = (np.array(coverage_after) * self.grid_points["weight"]).sum() / total_weight * 100.0
        
        return {
            "pct_covered_before": np.round(pct_covered_before, 1),
            "pct_covered_after": np.round(pct_covered_after, 1),
            "avg_dist_before_km": np.round(np.mean(dists_before), 2),
            "avg_dist_after_km": np.round(np.mean(dists_after), 2),
            "max_dist_before_km": np.round(np.max(dists_before), 2),
            "max_dist_after_km": np.round(np.max(dists_after), 2),
            "coverage_improvement_pct": np.round(pct_covered_after - pct_covered_before, 1)
        }

    def calculate_brand_coverage(self, threshold_km=5.0):
        """Calculates population-weighted BSS coverage percentages for each operator brand."""
        if self.grid_points is None:
            self.load_data()
            
        brands = self.existing_stations["brand"].unique()
        coverage_by_brand = {}
        total_weight = self.grid_points["weight"].sum()
        
        for brand in brands:
            brand_stations = self.existing_stations[self.existing_stations["brand"] == brand]
            covered_flags = []
            for _, pt in self.grid_points.iterrows():
                dists = haversine_distance_vectorized(pt["lat"], pt["lon"], brand_stations["lat"].values, brand_stations["lon"].values)
                min_dist = np.min(dists)
                covered_flags.append(1 if min_dist <= threshold_km else 0)
            
            coverage_pct = (np.array(covered_flags) * self.grid_points["weight"]).sum() / total_weight * 100.0
            coverage_by_brand[brand] = np.round(coverage_pct, 1)
            
        return coverage_by_brand

    def recalculate_rider_distances(self, df_recs, riders_df):
        """Recalculates distances to the nearest BSS for all riders incorporating recommended stations."""
        if len(df_recs) == 0:
            return riders_df["Distance_to_BSS_km"].values
            
        new_lats = df_recs["Latitude"].values
        new_lons = df_recs["Longitude"].values
        
        all_lats = np.concatenate([self.existing_stations["lat"].values, new_lats])
        all_lons = np.concatenate([self.existing_stations["lon"].values, new_lons])
        
        new_distances = []
        for _, rider in riders_df.iterrows():
            dists = haversine_distance_vectorized(rider["Latitude"], rider["Longitude"], all_lats, all_lons)
            new_distances.append(np.min(dists))
            
        return np.round(new_distances, 2)

if __name__ == "__main__":
    import os
    if os.path.exists("data/nairobi_subcounties.csv") and os.path.exists("data/existing_stations.csv"):
        opt = NetworkOptimizer()
        opt.load_data()
        gaps = opt.find_coverage_gaps()
        print(f"Total grid points in coverage gaps (>5km): {len(gaps)}")
        
        recs = opt.recommend_stations(n_stations=10)
        print("\nTop Recommended Locations:")
        print(recs.head(5))
        
        impact = opt.evaluate_expansion_impact(recs)
        print("\nExpansion Impact:")
        for k, v in impact.items():
            print(f" - {k}: {v}")
    else:
        print("Data files not found. Run data_processor.py first.")
