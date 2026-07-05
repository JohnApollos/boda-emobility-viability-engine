import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="Kenya Boda Boda E-Mobility Viability Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern enterprise dashboard styling (Inter Typography, Slate Colors, Muted Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"], .stText, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Dark Slate Canvas background */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* Minimalist Card style */
    .dashboard-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Banner style */
    .banner-card {
        background-color: #1e293b;
        border-bottom: 2px solid #0d9488; /* Teal Accent */
        border-radius: 0 0 8px 8px;
        padding: 24px 30px;
        margin-bottom: 30px;
    }
    
    /* Premium Metric Box with CSS Grid */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .metric-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981; /* Emerald */
        margin-bottom: 4px;
        line-height: 1.1;
    }
    
    .metric-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Custom Alerts */
    .alert-panel {
        border-left: 4px solid #ef4444; /* Muted Red */
        background-color: #1e293b;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        border-radius: 0 6px 6px 0;
        padding: 16px;
        margin: 15px 0;
        color: #f1f5f9;
        font-size: 0.9rem;
    }
    
    .success-panel {
        border-left: 4px solid #10b981; /* Muted Emerald */
        background-color: #1e293b;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        border-radius: 0 6px 6px 0;
        padding: 16px;
        margin: 15px 0;
        color: #f1f5f9;
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px 6px 0 0;
        padding: 0 16px;
        color: #64748b;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        border-color: #334155 #334155 transparent #334155 !important;
        color: #14b8a6 !important; /* Teal */
        font-weight: 600;
    }

    /* Premium Custom Table styling */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.825rem;
        color: #f1f5f9;
        margin-top: 8px;
    }
    .custom-table th {
        background-color: #0f172a;
        color: #94a3b8;
        text-align: left;
        padding: 10px 12px;
        font-weight: 600;
        border-bottom: 2px solid #334155;
        text-transform: uppercase;
        font-size: 0.725rem;
        letter-spacing: 0.05em;
    }
    .custom-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #334155;
        color: #e2e8f0;
    }
    .custom-table tr:hover {
        background-color: rgba(255, 255, 255, 0.02);
    }

    /* Responsive CSS Media Queries for Mobile Screens */
    @media (max-width: 1024px) {
        .metric-container {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }
    @media (max-width: 640px) {
        .metric-container {
            grid-template-columns: 1fr !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Imports from src
from src.data_processor import generate_datasets, haversine_distance
from src.risk_model import PAYGRiskModel
from src.optimizer import NetworkOptimizer, haversine_distance_vectorized

# Check if data exists; if not, generate it
if not os.path.exists("data/rider_loans.csv") or not os.path.exists("data/existing_stations.csv"):
    with st.spinner("Initializing system and generating realistic spatial and rider datasets..."):
        generate_datasets()

def render_html_table(df):
    """Converts a pandas DataFrame to a beautiful custom HTML table fitting the dark dashboard style."""
    return df.to_html(classes="custom-table", index=False, escape=False)

def calculate_rider_distances_custom(riders_df, stations_df, df_recs=None, enable_interop=False):
    """Calculates distances to the nearest swap station for all riders incorporating optional brand interoperability."""
    distances = []
    for idx, rider in riders_df.iterrows():
        # Fallback if Brand column is not in CSV
        if "Brand" in riders_df.columns:
            r_brand = rider["Brand"]
        elif "brand" in riders_df.columns:
            r_brand = rider["brand"]
        else:
            # Deterministic brand distribution matching market shares:
            # 40% Ampersand, 35% Spiro, 25% ARC Ride
            if idx % 20 < 8:
                r_brand = "Ampersand"
            elif idx % 20 < 15:
                r_brand = "Spiro"
            else:
                r_brand = "ARC Ride"
        
        # In interoperability mode, Ampersand and ARC Ride share networks
        if enable_interop and r_brand in ["Ampersand", "ARC Ride"]:
            allowed_brands = ["Ampersand", "ARC Ride"]
        else:
            allowed_brands = [r_brand]
            
        # Filter existing stations to allowed brands
        brand_stations = stations_df[stations_df["brand"].isin(allowed_brands)]
        
        all_lats = brand_stations["lat"].values
        all_lons = brand_stations["lon"].values
        
        # Combine with proposed recommendations if provided
        if df_recs is not None and len(df_recs) > 0:
            all_lats = np.concatenate([all_lats, df_recs["Latitude"].values])
            all_lons = np.concatenate([all_lons, df_recs["Longitude"].values])
            
        dists = haversine_distance_vectorized(rider["Latitude"], rider["Longitude"], all_lats, all_lons)
        distances.append(np.min(dists))
        
    return np.round(distances, 2)

# Load datasets
@st.cache_data
def load_cached_data():
    subcounties = pd.read_csv("data/nairobi_subcounties.csv")
    existing_stations = pd.read_csv("data/existing_stations.csv")
    riders = pd.read_csv("data/rider_loans.csv")
    return subcounties, existing_stations, riders

subcounties_df, stations_df, riders_df = load_cached_data()

# Initialize and train risk model
@st.cache_resource
def get_risk_model():
    model = PAYGRiskModel()
    model.train()
    return model

risk_model = get_risk_model()

# Initialize spatial optimizer
@st.cache_resource
def get_optimizer():
    opt = NetworkOptimizer()
    opt.load_data()
    return opt

optimizer = get_optimizer()

# Title banner
st.markdown("""
<div class="banner-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1 style="margin: 0; color: #ffffff; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em;">Kenya Boda Boda E-Mobility Viability Engine</h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95rem; max-width: 900px; line-height: 1.4;">
                A data-driven spatial optimization and credit risk modeling framework designed to resolve the network expansion chicken-and-egg problem for Nairobi's e-motorcycle swap providers.
            </p>
        </div>
        <div>
            <a href="https://github.com/JohnApollos/boda-emobility-viability-engine" target="_blank" style="text-decoration: none;">
                <div style="background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 8px 16px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">
                    VIEW ON GITHUB
                </div>
            </a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar parameters
st.sidebar.markdown("""
<div style="margin-bottom: 20px;">
    <h4 style="margin: 0; color: #14b8a6; font-weight: 700; letter-spacing: 0.05em; font-size: 0.9rem; text-transform: uppercase;">Parameters Control Panel</h4>
    <hr style="border-color: #1e293b; margin: 10px 0 20px 0;">
</div>
""", unsafe_allow_html=True)

petrol_price_input = st.sidebar.slider(
    "EPRA Petrol Price (KES / Liter)", 
    min_value=170.0, 
    max_value=240.0, 
    value=200.0, 
    step=2.5,
    help="Monthly petroleum price bulletined by EPRA."
)

daily_payment_input = st.sidebar.slider(
    "Daily PAYG Installment (KES / Day)", 
    min_value=300, 
    max_value=600, 
    value=450, 
    step=10,
    help="Standard daily loan payment offered by asset financiers (e.g., Watu, Mogo)."
)

swap_radius_input = st.sidebar.slider(
    "Desired Swap Coverage Radius (km)", 
    min_value=2.0, 
    max_value=8.0, 
    value=5.0, 
    step=0.5,
    help="Target BSS coverage threshold within which riders operate efficiently."
)

kplc_tariff = st.sidebar.slider(
    "Electricity Tariff (KES / kWh)", 
    min_value=10.0, 
    max_value=30.0, 
    value=17.0, 
    step=1.0,
    help="KPLC approved base rate for e-mobility charging is KES 17.00/kWh."
)

n_rec = st.sidebar.slider(
    "New Stations to Optimize (Count)",
    min_value=3,
    max_value=20,
    value=10,
    step=1,
    help="Number of new swap station locations to be optimized via weighted K-Means."
)

enable_interop = st.sidebar.checkbox(
    "Enable Network Interoperability (Open BaaS)",
    value=False,
    help="Simulate a shared battery-swapping network between Ampersand and ARC Ride to expand coverage without building new stations."
)

st.sidebar.markdown("""
<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 14px; margin-top: 30px;">
    <h6 style="margin: 0 0 6px 0; color: #ffffff; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Integrated Data Sources</h6>
    <ul style="margin: 0; padding-left: 16px; font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">
        <li>KNBS Census 2019 Demographics</li>
        <li>ChargeUp! Research (Imperial College)</li>
        <li>KPLC Retail Tariffs & EPRA Pricing</li>
        <li>Nairobi OSM Road Layers</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Tabs without emojis
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Boda Boda Activity Matrix",
    "Infrastructure Coverage Gaps",
    "PAYG Credit Default Risk",
    "BSS Network Optimization",
    "Fiscal Policy Sensitivity"
])

# -----------------
# TAB 1: BODA BODA ACTIVITY MATRIX
# -----------------
with tab1:
    st.markdown("### Component 1: Nairobi Boda Boda Activity Corridor Map")
    st.markdown("""
    This map models spatial travel demand across Nairobi. Density is estimated by combining 
    **subcounty population density** from the KNBS 2019 Census with **boda boda concentration weights** 
    along major transit corridors (e.g., Juja Road, Mombasa Road, Outer Ring Road, CBD, Westlands) 
    replicated from the empirical trip distribution matrices of the *ChargeUp!* research paper.
    """)
    
    # Custom HTML metrics grid
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-box">
            <div class="metric-val">{subcounties_df["population"].sum():,}</div>
            <div class="metric-lbl">Total Population</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">1.56</div>
            <div class="metric-lbl">Daily Swaps / Rider</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">4.5 km</div>
            <div class="metric-lbl">Avg Trip Length</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">20.1 km/h</div>
            <div class="metric-lbl">Avg Operating Speed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    map_col, info_col = st.columns([2, 1])
    
    with map_col:
        # Create map
        m1 = folium.Map(location=[-1.2921, 36.8219], zoom_start=11.5, tiles="CartoDB positron")
        
        # Prepare heatmap data
        heatmap_data = []
        for _, row in subcounties_df.iterrows():
            heatmap_data.append([row["lat"], row["lon"], row["boda_factor"] * 5])
            for _ in range(int(row["boda_factor"] * 30)):
                noise_lat = np.random.normal(0, 0.015)
                noise_lon = np.random.normal(0, 0.015)
                heatmap_data.append([row["lat"] + noise_lat, row["lon"] + noise_lon, row["boda_factor"]])
                
        # Heatmap layer
        HeatMap(heatmap_data, radius=18, blur=15, min_opacity=0.3).add_to(m1)
        
        # Render map
        st_folium(m1, width="100%", height=500, returned_objects=[])
        
    with info_col:
        st.markdown("""
        <div class="dashboard-card" style="margin-bottom: 12px; padding: 16px;">
            <h5 style="margin-top: 0; color: #ffffff; font-weight: 700; font-size: 0.95rem;">KNBS Census Demographics</h5>
            <p style="font-size: 0.8rem; color: #94a3b8; margin: 0 0 10px 0;">Official area and densities used to weight travel density modeling.</p>
        </div>
        """, unsafe_allow_html=True)
        
        df_sub_display = subcounties_df[["name", "population", "area_sqkm", "density"]].sort_values(by="density", ascending=False).copy()
        df_sub_display["population"] = df_sub_display["population"].apply(lambda x: f"{x:,}")
        df_sub_display["density"] = df_sub_display["density"].apply(lambda x: f"{x:,}")
        df_sub_display.columns = ["Sub-County", "Population", "Area (sq km)", "Density (Pop/sq km)"]
        st.markdown(render_html_table(df_sub_display), unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Diurnal Operating Profile: Daily Trips vs. Swap Probability")
    st.markdown("""
    Rider behavior is heavily structured. Evening trip peaks (17:00–18:00) coincide with a steep drop in battery swapping 
    probability, as riders prioritize passenger volume and avoid stopping during high-fare windows. Morning shifts, however, 
    witness high swap volumes at higher state-of-charge (SoC) levels as riders prep for their shifts.
    """)
    
    hours_list = list(range(24))
    trips_curve = [3, 2, 1, 1, 3, 10, 25, 45, 60, 65, 55, 50, 48, 52, 58, 68, 85, 100, 75, 55, 38, 25, 15, 8]
    swaps_curve = [1.5, 0.8, 0.5, 0.5, 1.2, 5.5, 18.0, 38.0, 55.0, 62.0, 52.0, 42.0, 38.0, 40.0, 42.0, 48.0, 65.0, 42.0, 58.0, 48.0, 32.0, 20.0, 10.0, 4.5]
    
    fig_diurnal = go.Figure()
    fig_diurnal.add_trace(go.Scatter(
        x=hours_list, y=trips_curve,
        mode='lines+markers',
        name='Relative Daily Trips (Passenger Volume)',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=6)
    ))
    fig_diurnal.add_trace(go.Scatter(
        x=hours_list, y=swaps_curve,
        mode='lines+markers',
        name='Battery Swap Probability',
        line=dict(color='#f59e0b', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    fig_diurnal.add_annotation(
        x=17, y=42,
        text="Evening Rush Hour Dip (Riders avoid BSS)",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ef4444",
        ax=-70, ay=-40,
        font=dict(color="#f8fafc", size=10),
        bgcolor="rgba(239, 68, 68, 0.5)"
    )
    fig_diurnal.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#f8fafc',
        xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode='linear', tick0=0, dtick=2, showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Relative Index (%)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=30, b=40),
        height=320
    )
    st.plotly_chart(fig_diurnal, use_container_width=True)



# -----------------
# TAB 2: INFRASTRUCTURE COVERAGE GAPS
# -----------------
with tab2:
    st.markdown("### Component 2: Battery Swap Station Coverage Gap Analysis")
    st.markdown(f"""
    By plotting the current network footprints (Ampersand, Spiro, and ARC Ride) and applying a **{swap_radius_input}km radius coverage buffer**, 
    we isolate the spatial coverage gaps. Areas outside these buffers are ranked by unserved rider potential to determine priority expansion centroids.
    """)
    
    # Calculate coverage
    df_gaps = optimizer.find_coverage_gaps(threshold_km=swap_radius_input)
    impact_baseline = optimizer.evaluate_expansion_impact(pd.DataFrame())
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-box">
            <div class="metric-val">{len(stations_df)}</div>
            <div class="metric-lbl">Existing Stations</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{impact_baseline["pct_covered_before"]}%</div>
            <div class="metric-lbl">Coverage Rate ({swap_radius_input}km)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{impact_baseline["avg_dist_before_km"]} km</div>
            <div class="metric-lbl">Avg Distance to BSS</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{impact_baseline["max_dist_before_km"]} km</div>
            <div class="metric-lbl">Max Distance to BSS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    map_col2, list_col2 = st.columns([2, 1])
    
    with map_col2:
        m2 = folium.Map(location=[-1.2921, 36.8219], zoom_start=11.5, tiles="CartoDB dark_matter")
        
        colors = {"Ampersand": "#10b981", "Spiro": "#3b82f6", "ARC Ride": "#f59e0b"}
        
        # Add existing stations using premium HTML dot icons
        for _, row in stations_df.iterrows():
            brand_color = colors.get(row["brand"], "#ffffff")
            # Custom styled DivIcon (Clean circular marker)
            html_icon = f"""
            <div style="
                background-color: {brand_color}; 
                border: 2px solid #ffffff; 
                width: 14px; 
                height: 14px; 
                border-radius: 50%;
                box-shadow: 0 0 6px rgba(0,0,0,0.5);
            "></div>
            """
            
            folium.Marker(
                location=[row["lat"], row["lon"]],
                popup=f"<b>{row['name']}</b><br>Brand: {row['brand']}<br>Cabinets: {row['cabinets']}",
                icon=folium.DivIcon(html=html_icon, icon_size=(14, 14), icon_anchor=(7, 7))
            ).add_to(m2)
            
            # Coverage buffer ring
            folium.Circle(
                location=[row["lat"], row["lon"]],
                radius=swap_radius_input * 1000,
                color=brand_color,
                weight=1,
                fill=True,
                fill_color=brand_color,
                fill_opacity=0.04
            ).add_to(m2)
            
        # Plot coverage gap points as small red dots
        if len(df_gaps) > 0:
            sample_gaps = df_gaps.sample(min(150, len(df_gaps)), random_state=42)
            for _, row in sample_gaps.iterrows():
                html_gap = f"""
                <div style="
                    background-color: #ef4444; 
                    width: 6px; 
                    height: 6px; 
                    border-radius: 50%;
                "></div>
                """
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    icon=folium.DivIcon(html=html_gap, icon_size=(6, 6), icon_anchor=(3, 3)),
                    popup=f"Coverage Gap Centroid<br>Nearest Station: {row['min_dist_to_existing']:.2f} km"
                ).add_to(m2)
                
        st_folium(m2, width="100%", height=500, returned_objects=[])
        
    with list_col2:
        st.markdown("""
        <div class="dashboard-card" style="margin-bottom: 12px; padding: 16px;">
            <h5 style="margin-top: 0; color: #ffffff; font-weight: 700; font-size: 0.95rem;">Unserved Rider Distribution</h5>
            <p style="font-size: 0.8rem; color: #94a3b8; margin: 0 0 10px 0;">Estimated unserved rider counts mapped by administrative subcounty.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(df_gaps) > 0:
            gap_summary = df_gaps.groupby("nearest_subcounty").size().reset_index(name="Gap Points")
            gap_summary = gap_summary.merge(subcounties_df[["name", "density"]], left_on="nearest_subcounty", right_on="name")
            gap_summary["Estimated Unserved Riders"] = np.round(gap_summary["Gap Points"] * gap_summary["density"] / 40.0, 0).astype(int)
            gap_summary = gap_summary[["nearest_subcounty", "Estimated Unserved Riders"]].sort_values(by="Estimated Unserved Riders", ascending=False)
            gap_summary["Estimated Unserved Riders"] = gap_summary["Estimated Unserved Riders"].apply(lambda x: f"{x:,}")
            gap_summary.columns = ["Sub-County", "Est. Unserved Riders"]
            st.markdown(render_html_table(gap_summary), unsafe_allow_html=True)
        else:
            st.info("No coverage gaps found. Current BSS footprint satisfies requirements.")
            
        st.write("")
        st.markdown("""
        <div class="dashboard-card" style="margin-top: 15px; padding: 16px; margin-bottom: 0px;">
            <h5 style="margin-top: 0; color: #ffffff; font-weight: 700; font-size: 0.95rem;">Competitor Coverage Share</h5>
            <p style="font-size: 0.8rem; color: #94a3b8; margin: 0 0 15px 0;">Percentage of Nairobi's populated points covered by each provider independently.</p>
        </div>
        """, unsafe_allow_html=True)
        
        brand_coverages = optimizer.calculate_brand_coverage(threshold_km=swap_radius_input)
        
        if enable_interop:
            # Combined Ampersand + ARC Ride network
            interop_stations = stations_df[stations_df["brand"].isin(["Ampersand", "ARC Ride"])]
            covered_flags = []
            total_weight = optimizer.grid_points["weight"].sum()
            for _, pt in optimizer.grid_points.iterrows():
                dists = haversine_distance_vectorized(pt["lat"], pt["lon"], interop_stations["lat"].values, interop_stations["lon"].values)
                min_dist = np.min(dists)
                covered_flags.append(1 if min_dist <= swap_radius_input else 0)
            interop_cov = np.round((np.array(covered_flags) * optimizer.grid_points["weight"]).sum() / total_weight * 100.0, 1)
            
            st.markdown(f"""
            <div style="margin-bottom: 15px; background-color: #1e293b; border: 1px solid #14b8a6; border-radius: 6px; padding: 12px; box-shadow: 0 0 10px rgba(20, 184, 166, 0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.85rem;">
                    <span style="font-weight: 700; color: #14b8a6; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em;">Shared Interop (Ampersand + ARC)</span>
                    <span style="font-weight: 700; color: #14b8a6;">{interop_cov}%</span>
                </div>
                <div style="background-color: #0f172a; border-radius: 3px; height: 8px; width: 100%; overflow: hidden; margin-top: 4px;">
                    <div style="background-color: #14b8a6; width: {interop_cov}%; height: 100%; border-radius: 3px;"></div>
                </div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 6px; line-height: 1.2;">
                    Cooperative network sharing provides a massive range expansion for both fleets without deploying new hardware.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Display each brand's coverage rate
        for brand, cov in brand_coverages.items():
            color = "#10b981" if brand == "Ampersand" else ("#3b82f6" if brand == "Spiro" else "#f59e0b")
            st.markdown(f"""
            <div style="margin-bottom: 12px; background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.85rem;">
                    <span style="font-weight: 600; color: #e2e8f0;">{brand} Network</span>
                    <span style="font-weight: 700; color: {color};">{cov}%</span>
                </div>
                <div style="background-color: #0f172a; border-radius: 3px; height: 6px; width: 100%; overflow: hidden;">
                    <div style="background-color: {color}; width: {cov}%; height: 100%; border-radius: 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# -----------------
# TAB 3: PAYG DEFAULT RISK
# -----------------
with tab3:
    st.markdown("### Component 3: Rider Economics and PAYG Default Risk Model")
    st.markdown("""
    Asset financiers (e.g., M-KOPA, Watu, Mogo) require modeling to identify credit risk. 
    This tab evaluates how physical proximity to swap infrastructure interacts with financial parameters to drive payment default probability.
    """)
    
    col_scorer, col_model = st.columns([1, 1])
    
    with col_scorer:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("<h5 style='margin-top:0; color:#ffffff; font-weight: 700;'>Interactive Rider Credit Profiler</h5>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.8rem; color:#94a3b8;'>Adjust the rider parameters to calculate default risk using the trained logistic regression model.</p>", unsafe_allow_html=True)
        
        sc_input = st.selectbox("Rider Administrative Region", subcounties_df["name"].values)
        r_dist = st.slider("Rider Operating Distance from BSS (km)", 0.2, 12.0, 4.0, 0.2)
        r_inc = st.slider("Daily Average Revenue (KES / Day)", 400, 1800, 1000, 50)
        r_vol = st.slider("Revenue Volatility (Standard Deviation %)", 10, 50, 20, 2) / 100.0
        
        # Savings
        dist = 80.0
        petrol_c = (dist / 35.0) * petrol_price_input + 50.0
        electric_c = (dist / 50.0) * 185.0 + 10.0
        savings = petrol_c - electric_c
        
        prob = risk_model.predict_probability(r_dist, r_vol, r_inc, savings)
        
        if prob < 0.15:
            color = "#10b981" # Emerald Green
            label = "Low Risk Profile"
        elif prob < 0.30:
            color = "#f59e0b" # Amber Gold
            label = "Medium Risk Profile"
        else:
            color = "#ef4444" # Muted Red
            label = "High Risk Profile"
            
        st.markdown(f"""<div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 16px; margin: 15px 0 10px 0;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span style="font-size: 0.8rem; color: #94a3b8; font-weight:600; text-transform: uppercase;">Repayment Risk Score</span>
<span style="font-size: 0.85rem; color: {color}; font-weight: 700; text-transform: uppercase;">{label}</span>
</div>
<div style="font-size: 2rem; font-weight: 800; color: #ffffff; line-height: 1;">{prob * 100:.1f}%</div>
<div style="background-color: #334155; border-radius: 4px; height: 8px; width: 100%; margin-top: 12px; overflow: hidden;">
<div style="background-color: {color}; width: {prob*100}%; height: 8px; border-radius: 4px;"></div>
</div>
</div>""", unsafe_allow_html=True)
        
        reduced_prob = risk_model.predict_probability(1.0, r_vol, r_inc, savings)
        st.markdown(f"""
        <div class="success-panel" style="border-left-color: {color}; background-color: #0f172a; border: 1px solid #1e293b; border-left: 4px solid {color}; margin-top: 15px;">
            <b>Infrastructure Credit Mitigation:</b> By placing a new swap station nearby, the rider's travel distance is reduced to <b>1.0km</b>, 
            which lowers their credit default probability from <b>{prob*100:.1f}%</b> to <b>{reduced_prob*100:.1f}%</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Portfolio Credit Mitigation Feedback Loop
        st.write("")
        st.markdown("""
        <h5 style="margin-top: 15px; margin-bottom: 10px; color: #ffffff; font-weight: 700; font-size: 0.95rem;">Portfolio Credit Risk Projections</h5>
        """, unsafe_allow_html=True)
        
        # Calculate new distances incorporating Tab 4 recommendations
        df_recs = optimizer.recommend_stations(n_stations=n_rec, gap_threshold_km=swap_radius_input)
        
        # Recalculate baseline and current distances based on interop status and recommended sites
        dists_baseline = calculate_rider_distances_custom(riders_df, stations_df, df_recs=None, enable_interop=False)
        dists_current = calculate_rider_distances_custom(riders_df, stations_df, df_recs=df_recs, enable_interop=enable_interop)
        
        risk_before = risk_model.evaluate_portfolio_risk(riders_df, dists_baseline)
        risk_after = risk_model.evaluate_portfolio_risk(riders_df, dists_current)
        
        # Expected Loss calculation (Average vehicle cost KES 450,000)
        mitigated_loss = (risk_before['expected_default_rate'] - risk_after['expected_default_rate']) / 100.0 * 1000 * 450000
        
        st.markdown(f"""<div class="dashboard-card" style="padding: 16px; margin-bottom: 0px;">
<div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; margin-bottom: 12px;">GIS-Credit Feedback Loop</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem;">
<span style="color: #94a3b8;">Portfolio Default Rate:</span>
<span style="font-weight: 700; color: #ffffff;">{risk_before['expected_default_rate']:.2f}% &rarr; <span style="color: #10b981;">{risk_after['expected_default_rate']:.2f}%</span></span>
</div>
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem;">
<span style="color: #94a3b8;">High-Risk Riders (>15%):</span>
<span style="font-weight: 700; color: #ffffff;">{risk_before['high_risk_riders_count']} &rarr; <span style="color: #10b981;">{risk_after['high_risk_riders_count']}</span></span>
</div>
<div style="border-top: 1px solid #334155; margin-top: 12px; padding-top: 12px;">
<div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Capital Exposure Mitigated</div>
<div style="font-size: 1.4rem; font-weight: 800; color: #10b981;">KES {mitigated_loss:,.0f}</div>
<div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Based on a fleet size of 1,000 riders. {"Interoperability and adding" if enable_interop else "Adding"} <b>{n_rec} new stations</b> reduces overall default exposure.</div>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_model:
        st.markdown("#### Model Standardized Feature Importances")
        st.markdown("""
        The chart below displays the standardized coefficients from the Logistic Regression. Standardizing features 
        enables direct comparison, proving that physical distance to BSS dominates credit default risk.
        """)
        
        # Color palette for Plotly
        fig_imp = px.bar(
            risk_model.importances,
            x="Coefficient_Standardized",
            y="Feature",
            color="Direction",
            color_discrete_map={"Increase Risk": "#ef4444", "Decrease Risk": "#10b981"},
            orientation="h",
            labels={"Coefficient_Standardized": "Standardized Impact (log odds)", "Feature": ""},
            title="Statistical Impact on Credit Default"
        )
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis_showgrid=True,
            xaxis_gridcolor='rgba(255,255,255,0.08)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("##### Odds Ratio Summary Table")
        or_df = risk_model.importances[["Feature", "Odds_Ratio", "Direction"]].copy()
        or_df["Odds_Ratio"] = or_df["Odds_Ratio"].apply(lambda x: f"{x:.2f}")
        or_df.columns = ["Model Feature", "Odds Ratio", "Impact direction"]
        st.markdown(render_html_table(or_df), unsafe_allow_html=True)


# -----------------
# TAB 4: BSS NETWORK OPTIMIZATION
# -----------------
with tab4:
    st.markdown("### Component 4: Optimal Swap Station Placement Recommender")
    st.markdown("""
    Given budget constraints for $N$ new stations, this optimization algorithm places cabinets inside our coverage gaps 
    to maximize rider coverage while balancing road corridors and power grid stability.
    """)
    
    col_input4, col_metrics4 = st.columns([1, 2])
    with col_input4:
        # Coordinated off-peak charging savings (ChargeUp! Page 13)
        daily_energy = n_rec * 12 * 1.5
        cost_flat = daily_energy * kplc_tariff
        cost_offpeak = daily_energy * 0.68 * kplc_tariff # ~32% ToU coordinated savings
        savings_daily = cost_flat - cost_offpeak
        savings_annual = savings_daily * 300
        
        st.markdown(f"""
        <div class="dashboard-card" style="padding: 16px; margin-bottom: 0px;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; margin-bottom: 12px;">Coordinated Grid Load Savings</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem;">
                <span style="color: #94a3b8;">Est. Charging Load ({n_rec} BSS):</span>
                <span style="font-weight: 700; color: #ffffff;">{daily_energy:,.0f} kWh / Day</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem;">
                <span style="color: #94a3b8;">Flat Tariff Cost (Daily):</span>
                <span style="font-weight: 700; color: #ef4444;">KES {cost_flat:,.0f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.85rem;">
                <span style="color: #94a3b8;">Shifted ToU Cost (Daily):</span>
                <span style="font-weight: 700; color: #10b981;">KES {cost_offpeak:,.0f}</span>
            </div>
            <div style="border-top: 1px solid #334155; margin-top: 12px; padding-top: 12px;">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Annual Energy Savings</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #10b981;">KES {savings_annual:,.0f}</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">By shifting charging loads to off-peak hours (22:00-06:00) using KPLC Time-of-Use rates.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    df_recs = optimizer.recommend_stations(n_stations=n_rec, gap_threshold_km=swap_radius_input)
    impact = optimizer.evaluate_expansion_impact(df_recs)
    
    with col_metrics4:
        st.markdown(f"""
        <div style="display: flex; gap: 15px; margin-bottom: 20px;">
            <div style="flex:1; background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{impact["pct_covered_before"]}% ➔ {impact["pct_covered_after"]}%</div>
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight:600; margin-top:2px;">Network Coverage</div>
            </div>
            <div style="flex:1; background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{impact["avg_dist_before_km"]}km ➔ {impact["avg_dist_after_km"]}km</div>
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight:600; margin-top:2px;">Avg Distance to BSS</div>
            </div>
            <div style="flex:1; background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 12px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{impact["max_dist_before_km"]}km ➔ {impact["max_dist_after_km"]}km</div>
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight:600; margin-top:2px;">Max Distance to BSS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    map_col4, table_col4 = st.columns([2, 1])
    
    with map_col4:
        st.markdown("##### Optimization Mapping Results (Green = Existing, Indigo = Proposed)")
        m4 = folium.Map(location=[-1.2921, 36.8219], zoom_start=11.5, tiles="CartoDB dark_matter")
        
        # Existing stations (Green HTML dots)
        for _, row in stations_df.iterrows():
            html_ex = """
            <div style="
                background-color: #10b981; 
                border: 2px solid #ffffff; 
                width: 12px; 
                height: 12px; 
                border-radius: 50%;
                box-shadow: 0 0 6px rgba(0,0,0,0.5);
            "></div>
            """
            folium.Marker(
                location=[row["lat"], row["lon"]],
                icon=folium.DivIcon(html=html_ex, icon_size=(12,12), icon_anchor=(6,6)),
                popup=f"Existing BSS: {row['name']}"
            ).add_to(m4)
            
        # Recommended stations (Indigo HTML dots with a pulsing indicator style)
        for _, row in df_recs.iterrows():
            html_rec = """
            <div style="
                background-color: #6366f1; 
                border: 2px solid #ffffff; 
                width: 16px; 
                height: 16px; 
                border-radius: 50%;
                box-shadow: 0 0 10px rgba(99, 102, 241, 0.8);
            "></div>
            """
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                icon=folium.DivIcon(html=html_rec, icon_size=(16,16), icon_anchor=(8,8)),
                popup=f"<b>PROPOSED NEW BSS</b><br>ID: {row['Rec_ID']}<br>Subcounty: {row['Subcounty']}<br>Suitability Score: {row['Overall_Viability_Score']}/100"
            ).add_to(m4)
            
            # Show coverage buffer for proposed stations
            folium.Circle(
                location=[row["Latitude"], row["Longitude"]],
                radius=swap_radius_input * 1000,
                color="#6366f1",
                weight=1,
                fill=True,
                fill_color="#6366f1",
                fill_opacity=0.04
            ).add_to(m4)
            
        st_folium(m4, width="100%", height=500, returned_objects=[])
        
    with table_col4:
        st.markdown("""
        <div class="dashboard-card" style="margin-bottom: 12px; padding: 16px;">
            <h5 style="margin-top: 0; color: #ffffff; font-weight: 700; font-size: 0.95rem;">Ranked Expansion Sites</h5>
            <p style="font-size: 0.8rem; color: #94a3b8; margin: 0 0 10px 0;">Cluster centers generated via weighted K-Means on gap coordinates.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(df_recs) > 0:
            display_recs = df_recs[["Rec_ID", "Subcounty", "Overall_Viability_Score", "Distance_to_Existing_km"]].copy()
            display_recs["Overall_Viability_Score"] = display_recs["Overall_Viability_Score"].apply(lambda x: f"{x:.1f}")
            display_recs["Distance_to_Existing_km"] = display_recs["Distance_to_Existing_km"].apply(lambda x: f"{x:.2f} km")
            display_recs.columns = ["Rec ID", "Sub-County", "Viability /100", "Distance to Network"]
            st.markdown(render_html_table(display_recs), unsafe_allow_html=True)
        else:
            st.info("No recommendations needed. No coverage gaps exist.")


# -----------------
# TAB 5: FISCAL POLICY SENSITIVITY
# -----------------
with tab5:
    st.markdown("### Component 5: Policy Sensitivity Analysis & Scenario Simulator")
    st.markdown("""
    This simulator models how VAT policy shifts impact rider daily economics and credit portfolios. 
    By modifying grid tariffs and petroleum pricing, we track the break-even points of electric vs. petrol operations.
    """)
    
    col_input5, col_plot5 = st.columns([1, 2])
    
    with col_input5:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("<h5 style='margin-top:0; color:#ffffff; font-weight: 700;'>Macroeconomic Variables</h5>", unsafe_allow_html=True)
        
        daily_km_slider = st.slider("Rider Average Daily Distance (km)", 40, 140, 80, 5)
        
        st.markdown("<hr style='border-color:#334155; margin:15px 0;'>", unsafe_allow_html=True)
        st.markdown("<h6 style='margin:0 0 4px 0; color:#fff;'>Scenarios Modeled:</h6>", unsafe_allow_html=True)
        st.markdown("""
        <ul style="margin: 0; padding-left: 18px; font-size: 0.85rem; color: #94a3b8; line-height:1.4;">
            <li><b>VAT Exemption (Current):</b> 0% VAT on electric vehicles and swaps.</li>
            <li><b>Partial VAT (8%):</b> Proposed intermediate tariff on BSS energy transactions.</li>
            <li><b>Full VAT (16%):</b> Fiscal reintroduction proposed under structural tax adjustments.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Economics Calculations
    gross_rev = 1100.0
    
    # 1. Petrol rider
    petrol_liters = daily_km_slider / 35.0
    petrol_cost = petrol_liters * petrol_price_input + 50.0
    petrol_net = gross_rev - petrol_cost
    
    # 2. Electric
    scaled_swap_fee = 185.0 * (kplc_tariff / 17.0)
    
    def calculate_electric_costs(vat_rate):
        swaps_needed = daily_km_slider / 50.0
        base_swap_cost = swaps_needed * scaled_swap_fee
        fcc = 8.3 * swaps_needed * 3.0
        base_charge_taxable = base_swap_cost + fcc
        vat_charge = base_charge_taxable * vat_rate
        total_operating = base_swap_cost + vat_charge + 10.0
        
        # Vehicle payment
        vehicle_payment = daily_payment_input * (1 + vat_rate if vat_rate > 0 else 1.0)
        return total_operating, vehicle_payment
        
    op_0, payg_0 = calculate_electric_costs(0.0)
    op_8, payg_8 = calculate_electric_costs(0.08)
    op_16, payg_16 = calculate_electric_costs(0.16)
    
    margin_0 = gross_rev - op_0 - payg_0
    margin_8 = gross_rev - op_8 - payg_8
    margin_16 = gross_rev - op_16 - payg_16
    
    with col_plot5:
        # Plotly comparison
        fig_margin = go.Figure()
        
        scenarios = ["Petrol Baseline", "Electric (0% VAT)", "Electric (8% VAT)", "Electric (16% VAT)"]
        operating_costs = [petrol_cost, op_0, op_8, op_16]
        asset_financing = [0.0, payg_0, payg_8, payg_16]
        net_margins = [petrol_net, margin_0, margin_8, margin_16]
        
        fig_margin.add_trace(go.Bar(
            name='Daily Operating Cost (Fuel/Swaps)',
            x=scenarios,
            y=operating_costs,
            marker_color='#ef4444'
        ))
        
        fig_margin.add_trace(go.Bar(
            name='Daily Asset Financing (PAYG Loan)',
            x=scenarios,
            y=asset_financing,
            marker_color='#f59e0b'
        ))
        
        fig_margin.add_trace(go.Bar(
            name='Daily Net Margin (Rider Take-home)',
            x=scenarios,
            y=net_margins,
            marker_color='#10b981'
        ))
        
        fig_margin.update_layout(
            barmode='stack',
            title=f"Rider Daily Margin Breakdown ({daily_km_slider} km Daily Distance)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            yaxis_title="KES / Day",
            yaxis_showgrid=True,
            yaxis_gridcolor='rgba(255,255,255,0.08)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_margin, use_container_width=True)
        
        # Muted Warning Panels
        if margin_16 < petrol_net:
            st.markdown(f"""
            <div class="alert-panel">
                <b>CRITICAL IMPACT WARNING:</b> At 16% VAT, the daily net margin of electric riders (<b>KES {margin_16:.0f}</b>) 
                falls below that of petrol riders (<b>KES {petrol_net:.0f}</b>). Under this scenario, the financial incentive 
                to switch to electric is entirely wiped out, leading to <b>100% credit default acceleration</b> for e-mobility operators.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-panel">
                <b>VIABILITY MAINTAINED:</b> Despite policy changes, electric adoption retains an economic benefit of 
                <b>KES {margin_16 - petrol_net:.0f}/day</b> over petrol. The credit profile remains stable.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#334155; margin:30px 0;'>", unsafe_allow_html=True)
    st.markdown("### Component 6: ESG Carbon Offset & Climate Finance Calculator")
    st.markdown("""
    E-mobility adoption in East Africa is heavily subsidized and financed by **international carbon credits** and ESG funding. 
    This calculator projects the annual greenhouse gas (GHG) savings of transitioning a fleet of **5,000 boda bodas** 
    to electric operations, based on the actual carbon intensity of the Kenyan power grid.
    """)
    
    col_esg_left, col_esg_right = st.columns([1, 2])
    
    # Calculate emissions metrics
    fleet_size = 5000
    operating_days = 300
    total_km_annual = fleet_size * daily_km_slider * operating_days
    
    # Petrol Baseline
    petrol_liters_annual = total_km_annual / 35.0
    co2_petrol_annual = (petrol_liters_annual * 3.1) / 1000.0
    
    # Electric (Actual Kenya Grid - 90% Hydro/Geothermal/Wind)
    energy_kwh_annual = total_km_annual * 0.06
    co2_elec_renew = (energy_kwh_annual * 0.05) / 1000.0 # 50g CO2/kWh
    
    # Electric (Fossil Backup Grid - Thermal Heavy)
    co2_elec_fossil = (energy_kwh_annual * 0.45) / 1000.0 # 450g CO2/kWh
    
    net_offset_tons = co2_petrol_annual - co2_elec_renew
    
    with col_esg_left:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("<h5 style='margin-top:0; color:#ffffff; font-weight: 700;'>Climate Finance Model</h5>", unsafe_allow_html=True)
        
        carbon_price = st.slider("Carbon Credit Value (USD / Ton CO2)", 5.0, 50.0, 15.0, 1.0)
        exchange_rate = 130.0 # KES per USD
        
        revenue_usd = net_offset_tons * carbon_price
        revenue_kes = revenue_usd * exchange_rate
        
        st.markdown(f"""
        <div style="border-top: 1px solid #334155; margin-top: 12px; padding-top: 12px;">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Annual Net CO2 Offset</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #10b981;">{net_offset_tons:,.0f} Tons</div>
        </div>
        <div style="margin-top: 12px;">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Est. Carbon Credit Revenue</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #10b981;">KES {revenue_kes:,.0f}</div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">(USD {revenue_usd:,.0f} per year at KES {exchange_rate:.0f}/USD)</div>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_esg_right:
        # Plotly bar chart
        fig_esg = go.Figure()
        fig_esg.add_trace(go.Bar(
            x=["Petrol Baseline Fleet", "Electric Fleet (Actual Grid)", "Electric Fleet (Fossil Grid)"],
            y=[co2_petrol_annual, co2_elec_renew, co2_elec_fossil],
            marker_color=["#ef4444", "#10b981", "#f59e0b"],
            text=[f"{co2_petrol_annual:,.0f}t", f"{co2_elec_renew:,.0f}t", f"{co2_elec_fossil:,.0f}t"],
            textposition='auto',
        ))
        fig_esg.update_layout(
            title="Annual Carbon Emissions (Metric Tons of CO2 / Year)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            yaxis_title="Metric Tons CO2",
            yaxis_showgrid=True,
            yaxis_gridcolor='rgba(255,255,255,0.08)'
        )
        st.plotly_chart(fig_esg, use_container_width=True)
