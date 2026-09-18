import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

def perform_customer_segmentation(rfm_df, n_clusters=None):
    """
    Apply KMeans clustering on RFM features.
    If n_clusters is None, automatically determine the optimal K.
    """
    if rfm_df.empty or len(rfm_df) < 3:
        logger.warning("RFM DataFrame is too small to segment.")
        rfm_df['segment'] = 'Regular Customers'
        return rfm_df, 1, {}

    logger.info("Preprocessing RFM data for clustering...")
    # 1. Log transform to handle skewness
    # Add small constant to avoid log(0)
    rfm_log = pd.DataFrame()
    rfm_log['Recency'] = np.log1p(rfm_df['Recency'])
    rfm_log['Frequency'] = np.log1p(rfm_df['Frequency'])
    rfm_log['Monetary'] = np.log1p(np.clip(rfm_df['Monetary'], 0.01, None))
    
    # 2. Standardize
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # Determine optimal K if not provided
    max_k = min(7, len(rfm_df) - 1)
    if n_clusters is None:
        n_clusters = find_optimal_k(rfm_scaled, max_k)
        logger.info("Automatically determined optimal K = %d", n_clusters)
    
    # 3. Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(rfm_scaled)
    
    rfm_df['Cluster'] = cluster_labels
    
    # 4. Map Clusters to business segments dynamically
    # Calculate cluster profiles
    cluster_profiles = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    # Calculate a value score: higher frequency & monetary, lower recency
    # Value Score = log(Frequency) + log(Monetary) - log(Recency + 1)
    cluster_profiles['ValueScore'] = (
        np.log1p(cluster_profiles['Frequency']) + 
        np.log1p(np.clip(cluster_profiles['Monetary'], 1, None)) - 
        np.log1p(cluster_profiles['Recency'])
    )
    
    # Sort clusters by ValueScore ascending (lowest score = least valuable, highest = most valuable)
    sorted_clusters = cluster_profiles.sort_values(by='ValueScore', ascending=True)['Cluster'].tolist()
    
    # Business segments list from lowest value to highest value
    segments_pool = [
        "Lost Customers",
        "At Risk Customers",
        "Occasional Customers",
        "Regular Customers",
        "Loyal Customers",
        "VIP Customers"
    ]
    
    # Map each cluster to a segment based on its relative rank
    # If n_clusters is smaller than segments, we select spaced elements or map them proportionally
    cluster_to_segment = {}
    k = len(sorted_clusters)
    for i, cluster_id in enumerate(sorted_clusters):
        # Scale rank (0 to k-1) to segments pool (0 to len-1)
        pool_idx = int(np.round(i * (len(segments_pool) - 1) / (k - 1))) if k > 1 else len(segments_pool) - 1
        cluster_to_segment[cluster_id] = segments_pool[pool_idx]
        
    rfm_df['segment'] = rfm_df['Cluster'].map(cluster_to_segment)
    
    # Compute silhouette score for current clustering
    try:
        # Sample for quick silhouette calculation if dataset is large
        if len(rfm_scaled) > 5000:
            idx = np.random.choice(len(rfm_scaled), 5000, replace=False)
            sil = float(silhouette_score(rfm_scaled[idx], cluster_labels[idx]))
        else:
            sil = float(silhouette_score(rfm_scaled, cluster_labels))
    except Exception:
        sil = 0.0
        
    stats = {
        'silhouette_score': round(sil, 3),
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'optimal_k': n_clusters
    }
    
    logger.info("Clustering completed with Silhouette Score of %s", stats['silhouette_score'])
    return rfm_df, n_clusters, stats

def find_optimal_k(rfm_scaled, max_k=7):
    """Find optimal K using a mix of Elbow method and Silhouette score on sample."""
    if max_k < 3:
        return 3
        
    # Sample if too large
    if len(rfm_scaled) > 10000:
        np.random.seed(42)
        idx = np.random.choice(len(rfm_scaled), 10000, replace=False)
        sample = rfm_scaled[idx]
    else:
        sample = rfm_scaled
        
    best_k = 4
    best_sil = -1
    
    # We test K from 3 to max_k
    for k in range(3, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(sample)
            sil = silhouette_score(sample, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
        except Exception:
            continue
            
    return best_k

def calculate_segment_distributions(rfm):
    """Calculate segment distributions (counts, revenue percentages, average spends)."""
    if rfm.empty or 'segment' not in rfm.columns:
        return pd.DataFrame()
        
    summary = rfm.groupby('segment').agg({
        'CustomerID': 'count',
        'Monetary': ['sum', 'mean'],
        'Recency': 'mean',
        'Frequency': 'mean'
    }).reset_index()
    
    # Flatten multiindex columns
    summary.columns = ['segment', 'customer_count', 'total_revenue', 'average_spend', 'average_recency', 'average_frequency']
    
    total_rev = summary['total_revenue'].sum()
    total_cust = summary['customer_count'].sum()
    
    summary['revenue_share_pct'] = np.round((summary['total_revenue'] / total_rev) * 100, 2) if total_rev > 0 else 0
    summary['customer_share_pct'] = np.round((summary['customer_count'] / total_cust) * 100, 2) if total_cust > 0 else 0
    
    summary['total_revenue'] = np.round(summary['total_revenue'], 2)
    summary['average_spend'] = np.round(summary['average_spend'], 2)
    summary['average_recency'] = np.round(summary['average_recency'], 1)
    summary['average_frequency'] = np.round(summary['average_frequency'], 1)
    
    return summary
