# Kenya Boda Boda E-Mobility Viability Engine
### A Data-Driven Framework for Optimal Battery Swap Network Expansion and Credit Risk Management

**Live Cloud Dashboard:** [boda-emobility.streamlit.app](https://boda-emobility.streamlit.app/)

---

## The Strategic Challenge

For electric mobility companies in Kenya — such as Ampersand, Spiro, and ARC Ride — the primary obstacle to scaling is not rider interest or vehicle performance. It is a classic network coordination problem (the chicken-and-egg paradox):
> *Riders cannot transition to electric motorcycles without a dense, reliable network of battery swap stations (BSS) to eliminate range anxiety. However, operators cannot justify the capital expenditure (CapEx) to build swap cabinets without a large, active rider base already generating swapping revenue.*

At the same time, asset financing companies (like Watu, Mogo, and M-KOPA) face significant credit risk. They deploy billions of shillings in pay-as-you-go (PAYG) motorcycle loans without understanding how physical infrastructure density directly impacts a rider's ability to make daily loan payments.

This project provides a replicable analytical engine that solves the data side of this problem. It is a decision-support tool that combines spatial optimization, demographic demand estimation, credit risk modeling, utility load balancing, and climate finance scenario planning into a single interactive enterprise dashboard.

---

## Project Structure

```text
boda-emobility-viability-engine/
├── .gitignore
├── README.md
├── app.py                 # Streamlit dashboard interface and Plotly visualizations
├── requirements.txt       # Project dependencies
├── data/                  # Compiled KNBS subcounty demographics and generated datasets
│   ├── existing_stations.csv
│   ├── nairobi_subcounties.csv
│   └── rider_loans.csv
└── src/                   # Core analytical modules
    ├── data_processor.py  # Data extraction, network compilation, and rider simulation
    ├── optimizer.py       # Spatial gap analysis, K-Means placement, and brand coverage
    └── risk_model.py      # Logistic regression credit risk and portfolio evaluation
```

---

## The Six Analytical Components

### 1. Boda Boda Density and Route Corridor Heatmap
- **The Problem:** Where do riders actually operate?
- **The Method:** We combine official KNBS 2019 Census subcounty demographics (land area, population density) with estimated traffic intensities along Nairobi's major transit corridors (e.g., Juja Road, Mombasa Road, Outer Ring Road, CBD, Westlands) to model boda boda concentration.
- **The Outcome:** An interactive Folium heatmap demonstrating rider distribution. It also incorporates a diurnal operational curve (from Figures 6 and 7 of the ChargeUp! report) illustrating trip volumes alongside hourly battery swap probabilities.

### 2. Battery Swap Station Coverage Gap Analysis
- **The Problem:** Given the current network footprint, which high-density zones are underserved?
- **The Method:** We map the coordinates of 20 active swap stations operated by Ampersand (at TotalEnergies), Spiro (at Petrocity), and ARC Ride (at OLA Energy/HQ). Applying a user-defined coverage radius (e.g., 5km range limit), we perform a spatial difference query to isolate and rank unserved high-density centroids.
- **The Outcome:** An interactive map highlighting coverage gaps, supplemented by market intelligence cards showing the coverage share of each competitor brand.
- **Battery Interoperability Mode (Open BaaS):** The engine includes a toggle to simulate a shared battery-swapping network between Ampersand and ARC Ride, demonstrating how cooperative network sharing increases the effective coverage share to over 91% without deploying new hardware.

### 3. Rider Economics and PAYG Credit Default Risk Model
- **The Problem:** Which riders are most likely to default on their loan payments?
- **The Method:** We build a Logistic Regression credit scoring model trained on simulated rider profiles, utilizing empirical parameters: daily income, battery swap costs, petrol equivalent costs, and daily PAYG payments.
- **The Key Finding:** Distance to the nearest swap station is a dominant predictor of default. A rider operating far from a BSS spends critical operating hours traveling to swap batteries, reducing daily revenue.
- **Credit Feedback Loop:** The engine dynamically recalculates the portfolio default rate of all 1,000 riders when new stations are proposed or when **Battery Interoperability** is enabled. It proves how open-network sharing can drop portfolio default rates (e.g. from 4.90% to 3.20%) and mitigate millions of KES in capital exposure, completely CapEx-free.

### 4. Optimal Swap Station Placement Recommender
- **The Problem:** Given a budget for N new stations, where should they go to maximize coverage?
- **The Method:** We run weighted K-Means clustering on the spatial coordinates of the coverage gaps, using subcounty population densities as weights. The candidate centers are then scored using a Multi-Criteria Suitability Matrix:
  1. Rider Density within 5km (40% weight)
  2. Distance from Existing Network (30% weight)
  3. Power Grid Stability (15% weight) — prioritizing industrial zones over peri-urban grids.
  4. Road Corridor Proximity (15% weight)
- **Grid Load Economics:** We calculate the financial savings of shifting charging loads to off-peak periods (22:00 to 06:00) using KPLC Time-of-Use rates, showing a 32% operating cost reduction for the operator.

### 5. Policy Sensitivity and Scenario Simulator
- **The Problem:** How do fiscal changes (e.g., the proposed reintroduction of a 16% VAT on EVs and batteries) impact rider margins?
- **The Method:** We model three policy scenarios (Current VAT exemption, 8% partial VAT, 16% full VAT) to show the break-even impact on daily rider margins compared to petrol baselines.
- **The Key Finding:** A reintroduction of a 16% VAT on battery swaps completely wipes out the economic savings of electric motorcycles over petrol. This shifts electric riders' net daily take-home pay below the petrol baseline, accelerating credit defaults.

### 6. ESG Carbon Offset and Climate Finance Calculator
- **The Problem:** How do we quantify environmental benefits for climate finance?
- **The Method:** We model annual greenhouse gas (GHG) savings for a fleet of 5,000 electric boda bodas.
- **The Calculations:** We compare petrol exhaust emissions (3.1 kg CO2/liter) against electric operations under Kenya's actual high-renewables grid (50g CO2/kWh) and a fossil-heavy backup grid (450g CO2/kWh).
- **The Outcome:** A Plotly bar chart showing annual metric tons of CO2 offset, coupled with a slider to project annual carbon credit revenue (in USD and KES).

---

## Ground Truth Data Sources

Rather than using generic assumptions, the engine's parameters are grounded in public records and reports:
1. **KNBS 2019 Population Census (Vol I):** Exact subcounty demographics (Table 2.7, Page 48) to establish baseline population densities.
2. **Imperial College London and ARC Ride (ChargeUp! Data Swap Paper):** Baseline operational figures (Average trip distance: 4.48 km, daily swaps: 1.56/rider, optimal battery-to-bike ratio: 1.66).
3. **KPLC Retail Tariffs and EPRA Bulletins:** Electricity tariffs (17.00 KShs/kWh e-mobility rate vs. 21.68 domestic rate), Fuel Cost Charges (8.30 KShs/kWh), REP Levy (5%), and VAT (16%).

---

## Repository Architecture & Code Standards

The codebase has been refactored to institutional, production-grade engineering standards:
* **Separation of Concerns:** Core analytics are isolated in backend modules inside `src/`, while the Streamlit layer in `app.py` handles presentation and input mapping.
* **Strict Type Hinting (PEP 484):** All functions, data loaders, and model fittings are fully annotated with static types to enable linting and prevent logic regression.
* **Google-Style Docstrings:** Detailed Google-style parameter lists (`Args`, `Returns`, `Raises`) document all functions.
* **Structured Logging:** Uses Python's native `logging` module to track spatial grid construction and model training metrics.
* **External Styling (style.css):** Custom UI layouts and mobile-responsive styles are isolated in [src/style.css](file:///c:/dev/emobility/src/style.css) and loaded dynamically.

---

## Setup and Execution

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### 1. Clone the Repository and Navigate to the Directory
```bash
git clone https://github.com/JohnApollos/boda-emobility-viability-engine.git
cd boda-emobility-viability-engine
```

### 2. Configure Virtual Environment (Optional but Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

### Note on Automatic Data Generation
On the first execution, the application will automatically:
1. Parse population demographics directly from the official KNBS Census PDF inside the repository.
2. Initialize the baseline competitor swap networks (20 cabinets across Nairobi).
3. Simulate and save the cohort of 1,000 borrower PAYG credit profiles.
These datasets are written to the `data/` directory and cached for subsequent sessions.
