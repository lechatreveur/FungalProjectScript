import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sys

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

# ==== 1. Configuration ====
FINAL_4D_DIR = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/combined_analysis_outputs/fc_final_4d/"
LATENT_CSV = os.path.join(FINAL_4D_DIR, "fc_ae_4d_features.csv")
OUTPUT_CSV = os.path.join(FINAL_4D_DIR, "fc_ae_4d_clustered.csv")

def run_clustering():
    # 1. Load Data
    print("📥 Loading 4D latent features...")
    df = pd.read_csv(LATENT_CSV, index_col=0)
    
    # 2. KMeans Clustering (Choosing K=6)
    print("🧬 Performing KMeans clustering on 4D space...")
    clusterer = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_labels = clusterer.fit_predict(df[['Latent_1', 'Latent_2', 'Latent_3', 'Latent_4']])
    
    df['phenotype_cluster'] = cluster_labels
    
    # 3. Save
    df.to_csv(OUTPUT_CSV)
    print(f"✅ Created 6 clusters.")
    print(f"💾 Clustered data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_clustering()
