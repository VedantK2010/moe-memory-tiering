# CXL-Based Memory Tiering for MoE Models

## 📖 Concepts: What does this mean?
*   **MoE (Mixture-of-Experts):** Large AI models (like Mixtral) are divided into sub-networks called "experts". For every word processed, the AI only activates a few specific experts, leaving the rest idle.
*   **Memory Tiering (HBM + CXL):** High-Bandwidth Memory (HBM) is incredibly fast but expensive and limited in capacity. Compute Express Link (CXL) allows us to plug in massive amounts of cheaper, slower memory.
*   **Our Goal:** If we put active experts in fast HBM and idle experts in slower CXL, we save massive amounts of capacity. But since workloads shift, we need a "smart" caching strategy (like LRU or Periodic Re-profiling) to dynamically swap experts in and out of HBM without ruining the AI's speed or consuming too much power.

## 🚀 Key Findings (TL;DR)
After evaluating over 1.6 million real-world routing decisions from the Mixtral 8x7B model alongside cycle-accurate hardware simulations, we discovered that **a one-size-fits-all caching policy is sub-optimal for MoE models**:
*   **Finding 1 (Middle Layers are Reactive):** At middle layers (e.g., Layer 15), context shifts rapidly. A highly reactive **Pure LRU** caching strategy dominates here, achieving a **64.9% HBM hit rate** (beating static allocation by over 7%).
*   **Finding 2 (Deep Layers are Specialized):** At deep layers (e.g., Layer 31), experts become highly specialized for deep semantic logic. Routing becomes heavily skewed toward specific experts, allowing **Periodic Re-profiling** and Static allocation to take the lead with a **~67.2% hit rate**.
*   **Finding 3 (Significant Energy Savings):** Using cycle-accurate DRAMSim3 hardware physics, we proved that intelligent tiering doesn't just save latency—it reduces total memory power consumption by **~8%** by actively preventing expensive fetches across the CXL PCIe bus.

## 🗺️ How to Read This Project (Step-by-Step)
If you are evaluating this repository, it can look intimidating. We recommend exploring the files in this logical order:

### Step 1: The Interactive Demo
*   [**`app.py`**](dashboard/app.py): **Start here.** This is our Streamlit web dashboard. If you run it locally (`streamlit run dashboard/app.py`), it provides a complete user interface visualizing our latency trade-offs, DRAMSim3 hardware validation stats, and real-world caching hit rates.

### Step 2: The Core Logic (Source Code)
*   [**`tier_simulator.py`**](src/tier_simulator.py): The foundational math engine. It calculates exactly how much latency is added when an expert is fetched across the CXL bus rather than local HBM.
*   [**`periodic_reprofile.py`**](src/periodic_reprofile.py): Our primary experiment. This script tests different algorithms (Pure Static, Pure LRU, and Periodic Re-profiling) to see which one manages the HBM/CXL boundary best.
*   [**calc_energy.py**](src/calc_energy.py): The hardware physics. It converts raw memory access counts into actual nanoJoules (nJ) of energy consumed, based on our cycle-accurate DRAMSim3 simulations.
*   [**prefetch_simulator.py**](src/prefetch_simulator.py): Advanced machine learning. Implements a Markov Chain to predict future expert requests and prefetch them from CXL, hiding the latency penalty.

### Step 3: The Deliverables (Results)
*   [**`four_strategy_comparison.png`**](results/four_strategy_comparison.png): A visual graph showing how our different caching strategies react to sudden context shifts in the workload.
*   [**`energy_metrics.csv`**](results/energy_metrics.csv): The raw data proving that a highly reactive tiering strategy (like LRU) reduces total system memory power by ~8% by minimizing CXL usage.
*   [**`tiering_sweep_results.csv`**](results/tiering_sweep_results.csv): A spreadsheet showing the strict linear penalty of offloading experts to CXL.

### Step 4: Advanced Tools & Data (For Deep Dives)
*   [**`generate_trace.py`**](src/generate_trace.py): Generates synthetic workloads to test our caching theories before running them on real data.
*   [**`hybrid_strategy.py`**](src/hybrid_strategy.py): An advanced experiment testing a "half-static, half-dynamic" memory budget allocation.
*   [**`robustness_check.py`**](src/robustness_check.py): A cross-validation script ensuring our results hold up under different randomized mathematical seeds.
*   **`/data/real_expert_trace.csv`**: The massive 800,000+ token real-world routing datasets extracted from HuggingFace. *(Note: These raw CSV traces are excluded from GitHub via `.gitignore` to prevent repository bloat, but they power the final metrics found in the results folder).*
