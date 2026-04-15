import os
import sys
sys.path.append("app")
from data_loader import load_edgelist
from graph_builder import build_graph, get_top_nodes_by_degree
from feature_engineering import compute_features
from detector import rule_based_detection

def debug():
    df = load_edgelist("data/synthetic_1m.txt", max_rows=1000000)
    G = build_graph(df)
    
    top_nodes = get_top_nodes_by_degree(G, 5000)
    
    # Fake nodes IDs: 180001 to 180200
    fake_ids = set(range(180001, 180201))
    
    fakes_in_top = [n for n in top_nodes if n in fake_ids]
    print(f"Fake nodes in top 5000: {len(fakes_in_top)}")
    
    if not fakes_in_top:
        print("NO FAKE NODES IN TOP 5000!")
        degrees = [G.degree(n) for n in top_nodes]
        print(f"Top 5000 degrees range: {min(degrees)} to {max(degrees)}")
        
        fake_degrees = [G.degree(n) for n in G.nodes() if n in fake_ids]
        if fake_degrees:
            print(f"Fake nodes degrees range: {min(fake_degrees)} to {max(fake_degrees)}")
        else:
            print("Fake nodes do not exist in the graph!")
            
        print("Checking raw dataset for fake node 180001:")
        print(df[df['source'] == 180001].head())
        return

    features = compute_features(G, top_nodes)
    fake_features = features[features['node'].isin(fake_ids)]
    print("\nSample Fake node features:")
    print(fake_features.head())
    
    rule_df = rule_based_detection(features)
    fake_rule = rule_df[rule_df['node'].isin(fake_ids)]
    print("\nFake node rule labels:")
    print(fake_rule['rule_label'].value_counts())
    
    if len(fake_rule) > 0:
        print("\nReasons for first fake node:")
        print(fake_rule.iloc[0]['rule_reasons'])
        
if __name__ == "__main__":
    debug()
