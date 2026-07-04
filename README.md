# Kenya Boda Boda E-Mobility Viability Engine
### A Data-Driven Framework for Optimal Battery Swap Network Expansion & Credit Risk Management

---

## 📈 The Strategic Challenge

For electric mobility companies in Kenya — such as **Ampersand, Spiro, and ARC Ride** — the primary obstacle to scaling is not rider interest or vehicle performance. It is a classic **network coordination problem (the chicken-and-egg paradox)**:
> *Riders cannot transition to electric motorcycles without a dense, reliable network of battery swap stations (BSS) to eliminate range anxiety. However, operators cannot justify the capital expenditure (CapEx) to build swap cabinets without a large, active rider base already generating swapping revenue.*

At the same time, asset financing companies (like **Watu, Mogo, and M-KOPA**) face significant credit risk. They are deploying billions of shillings in pay-as-you-go (PAYG) motorcycle loans without understanding how physical infrastructure density directly impacts a rider's ability to make daily loan payments.

**This project provides a replicable analytical engine that solves the data side of this problem.** It is a decision-support tool that combines spatial optimization, demographic demand estimation, credit risk modeling, and policy scenario planning into a single interactive enterprise dashboard.

---

## 🛠️ The Five Analytical Components

### 1. Boda Boda Density & Route Corridor Heatmap
- **The Problem:** Where do riders actually operate?
- **The Method:** We combine official **KNBS 2019 Census subcounty demographics** (land area, population density) with estimated traffic intensities along Nairobi's major transit corridors (e.g., Juja Road, Mombasa Road, Outer Ring Road, CBD, Westlands) to model boda boda concentration.
- **The Outcome:** An interactive Folium heatmap demonstrating rider distribution to target high-activity corridors for network planning.

### 2. Battery Swap Station Coverage Gap Analysis
- **The Problem:** Given the current network footprint, which high-density zones are underserved?
- **The Method:** We map the coordinates of **20 active swap stations** operated by Ampersand (at TotalEnergies), Spiro (at Petrocity), and ARC Ride (at OLA Energy/HQ). Applying a **5km radius buffer** (the industry-standard range limit), we perform a spatial difference query to isolate and rank unserved high-density centroids.
- **The Outcome:** An interactive map highlighting coverage gaps ranked by estimated rider population.

### 3. Rider Economics & PAYG Credit Default Risk Model
- **The Problem:** Which riders are most likely to default on their loan payments?
- **The Method:** We build a Logistic Regression credit scoring model trained on simulated rider profiles, utilizing empirical parameters: daily income (KES 500–1,500), battery swap costs (KES 290), petrol equivalent costs (KES 500+), and daily PAYG payments (KES 450).
- **The Key Finding:** **Distance to the nearest swap station is a stronger predictor of default than baseline income.** A rider who operates $>5\text{km}$ from a BSS spends critical operating hours traveling to swap batteries, reducing their daily operating capacity and driving default probability.

### 4. Optimal Swap Station Placement Recommender
- **The Problem:** Given a budget for $N$ new stations, where should they go to maximize coverage?
- **The Method:** We run **weighted K-Means clustering** on the spatial coordinates of the coverage gaps, using subcounty population densities as weights. The candidate centers are then scored using a Multi-Criteria Suitability Matrix:
  1. *Rider Density within 5km (40% weight)*
  2. *Distance from Existing Network (30% weight)*
  3. *Power Grid Stability (15% weight)* — prioritizing industrial zones over peri-urban grids.
  4. *Road Corridor Proximity (15% weight)*
- **The Outcome:** A ranked list of recommended coordinates and a "Before vs. After" network simulation demonstrating coverage improvements.

### 5. Policy Sensitivity & Scenario Simulator
- **The Problem:** How do fiscal changes (e.g., the proposed reintroduction of a 16% VAT on EVs and batteries) impact rider margins and default rates?
- **The Method:** We model three policy scenarios (Current VAT exemption, 8% partial VAT, 16% full VAT) to show the break-even impact on daily rider margins.
- **The Key Finding:** A reintroduction of a 16% VAT on battery swaps completely wipes out the economic savings of electric motorcycles over petrol. This shifts electric riders' net daily take-home pay below the petrol baseline, accelerating credit defaults across the financier's portfolio.

---

## 📊 Ground Truth Data Sources

Rather than using generic assumptions, the engine's parameters are grounded in public records and reports:
1. **KNBS 2019 Population Census (Vol I):** Exact subcounty demographics (Table 2.7, Page 48) to establish baseline population densities.
2. **Imperial College London & ARC Ride (ChargeUp! Data Swap Paper):** Baseline operational figures (Average trip distance: $4.48 \text{ km}$, daily swaps: $1.56 \text{/rider}$, optimal battery-to-bike ratio: $1.66$).
3. **KPLC Retail Tariffs & EPRA Bulletins:** Electricity tariffs ($17.00 \text{ KShs/kWh}$ e-mobility rate vs. $21.68$ domestic rate), Fuel Cost Charges ($8.30 \text{ KShs/kWh}$), REP Levy ($5\%$), and VAT ($16\%$).

---

## 🚀 Setup & Execution

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### 1. Clone the Repository & Navigate to Folder
```bash
cd c:/dev/emobility
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
This command will:
1. Run `src/data_processor.py` to compile KNBS demographics and generate the simulated datasets in the `data/` directory.
2. Launch the Streamlit server and open the interactive dashboard in your browser.
