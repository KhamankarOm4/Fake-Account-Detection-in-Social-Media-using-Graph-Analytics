import random
import time
import os

def generate_dataset(output_path: str, target_edges: int = 1_000_000, num_fake_nodes: int = 200):
    print(f"Generating synthetic graph dataset with {target_edges:,} edges...")
    start_time = time.time()
    
    # 1. We will allocate about 10% edges to the "fake" accounts
    # The rules mandate: out_degree >= 20, out/in > 10, low clustering.
    # To ensure they are picked up in the "Top 5000" filter by the API, they need high total degree.
    edges_per_fake = 500
    total_fake_edges = num_fake_nodes * edges_per_fake
    
    total_real_edges = target_edges - total_fake_edges
    
    # Let's say the real network has a 5:1 edge-to-node ratio on average
    num_real_nodes = total_real_edges // 5 
    
    edges = []
    
    # 2. Generate Real Edges (simulate a social network)
    # To keep it fast and somewhat realistic, we simulate a hub-and-spoke 
    # model by weighting random choices so lower IDs are more popular (like celebrities).
    print(f" -> Generating {total_real_edges:,} real edges among {num_real_nodes:,} users...")
    
    # Real nodes generation
    print(f" -> Generating realistic real edges among {num_real_nodes:,} users...")
    population = []
    # Mix of hubs and generic users
    for i in range(1, min(10000, num_real_nodes + 1)):
        population.extend([i] * max(1, int(10000 / i)))
        
    for _ in range(total_real_edges):
        source = random.randint(1, num_real_nodes)
        if random.random() < 0.6: 
            target = random.choice(population)
        else:
            target = random.randint(1, num_real_nodes)
        if source != target:
            edges.append(f"{source} {target}\n")
    
    # 3. Generate Fake Edges (Messy and Varied)
    print(f" -> Generating varied fake edges from {num_fake_nodes:,} bot accounts...")
    fake_start_id = num_real_nodes + 1
    fake_end_id = fake_start_id + num_fake_nodes
    
    for fake_id in range(fake_start_id, fake_end_id):
        # Introduce randomness so every bot looks completely different! 
        # Out degree varies anywhere from 30 to 800.
        bot_degree = random.randint(30, 800)
        
        # Follow a chaotic mix of super-hubs and totally random users
        targets = set()
        while len(targets) < bot_degree:
            if random.random() < 0.4:
                targets.add(random.randint(1, 3000)) # stalk a hub
            else:
                targets.add(random.randint(3001, num_real_nodes)) # stalk random person

        for target in targets:
            edges.append(f"{fake_id} {target}\n")
            
        # Add a tiny bit of "reciprocal" following to create messy in-degrees (so in-degree isn't exactly 0)
        # Note: the reciprocal followers MUST be hubs (IDs < 3000) so they exist in the top 5000 subgraph filter!
        if random.random() < 0.6:
            mutuals = random.randint(1, 15)
            for _ in range(mutuals):
                random_hub = random.randint(1, 2000)
                edges.append(f"{random_hub} {fake_id}\n")
            
    # 4. Shuffle to mix real and fake edges randomly
    print(" -> Shuffling edges...")
    random.shuffle(edges)
    
    # 5. File writing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(edges)
        
    print(f"\nDone! Dataset written to '{output_path}' in {time.time()-start_time:.2f} seconds.")
    print(f"Total Edges Written: {len(edges):,}")
    print(f"Expected Fake Labels: {num_fake_nodes:,} (Node IDs {fake_start_id} to {fake_end_id - 1})")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "synthetic_1m.txt")
    generate_dataset(output_file, target_edges=1_000_000, num_fake_nodes=200)
