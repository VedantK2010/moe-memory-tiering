import pandas as pd
from collections import OrderedDict

HBM_LATENCY_NS = 150
CXL_LATENCY_NS = 220
CAPACITY = 4

def run_prefetch_simulation(trace_path, layer_name):
    print(f"--- Running Prefetch Simulation for {layer_name} ---")
    df = pd.read_csv(trace_path)
    
    # We use expert_1 (Top-1 routing) to track sequence patterns
    experts = df["expert_1"].tolist()
    
    cache = OrderedDict()
    # Transition matrix: transitions[current_expert][next_expert] = count
    transitions = {i: {j: 0 for j in range(8)} for i in range(8)}
    
    total_accesses = 0
    lru_hits = 0
    prefetch_hits = 0
    cxl_misses = 0
    
    prefetched_expert = None
    prev_expert = None
    
    for curr_expert in experts:
        total_accesses += 1
        
        # 1. Check Memory: Is it in LRU or was it Prefetched?
        if curr_expert in cache:
            lru_hits += 1
            cache.move_to_end(curr_expert)
        elif curr_expert == prefetched_expert:
            # We guessed right! CXL latency is hidden.
            prefetch_hits += 1
            cache[curr_expert] = True
            if len(cache) > CAPACITY:
                cache.popitem(last=False)
        else:
            # We guessed wrong, and it wasn't in cache. Full CXL penalty.
            cxl_misses += 1
            cache[curr_expert] = True
            if len(cache) > CAPACITY:
                cache.popitem(last=False)
                
        # 2. Update the Machine Learning Model (Markov Chain)
        if prev_expert is not None:
            transitions[prev_expert][curr_expert] += 1
        
        # 3. Make a Prediction for the NEXT token
        best_guess = None
        max_count = -1
        for next_exp, count in transitions[curr_expert].items():
            if count > max_count:
                max_count = count
                best_guess = next_exp
                
        # Only prefetch if the predicted expert is stuck in slow CXL memory
        if best_guess is not None and best_guess not in cache:
            prefetched_expert = best_guess
        else:
            prefetched_expert = None
            
        prev_expert = curr_expert

    effective_hit_rate = (lru_hits + prefetch_hits) / total_accesses * 100
    lru_only_rate = lru_hits / total_accesses * 100
    
    total_time = (lru_hits * HBM_LATENCY_NS) + (prefetch_hits * HBM_LATENCY_NS) + (cxl_misses * CXL_LATENCY_NS)
    avg_latency = total_time / total_accesses
    
    print(f"Base LRU Hit Rate:       {lru_only_rate:.2f}%")
    print(f"Successful Prefetches:   {prefetch_hits} times latency was hidden")
    print(f"Effective Hit Rate:      {effective_hit_rate:.2f}% (LRU + Prefetch)")
    print(f"New Average Latency:     {avg_latency:.2f} ns\n")

trace_15 = r"C:\Users\11ave\OneDrive\Documents\Birla Institute Of Technology And Science\Projects\MoE Memory Tiering\data\real_expert_trace.csv"
trace_31 = r"C:\Users\11ave\OneDrive\Documents\Birla Institute Of Technology And Science\Projects\MoE Memory Tiering\data\real_expert_trace_layer31.csv"

run_prefetch_simulation(trace_15, "Layer 15")
run_prefetch_simulation(trace_31, "Layer 31")
