I'm really the only person using this repo, so this README is really just for me.
This is listing what I'm going to look into more

Possible solutions to the problem with individuality in rats shown by my population_avg data
1. Normalization/Standardize between rats
2. Rat-Specific Identifier as a feature
3. Train individual models for each rat, then meta-learn
4. Cluster Rtas Based on Response Types
5. Feature Engineering Focusing on patterns instead of raw data

I've implemented 1/4 but the others are on the table
Possible improvements for 1/4

1. Normalization of rats.
   The rats are already normalized. Probably not super helpful but I haven't tested it

2. Clustering rats based on response types
   GPT's feedback (i'm cut for time):
   "Explanation of Your Graphs

Your clustering algorithm grouped rats based on the statistical properties (mean, max, std) of their neural responses over time. The graphs show how rats within each cluster exhibit similar response patterns.

    Cluster 0 (Top Graph)
        Rats 1, 5, and 10 belong to this cluster.
        Their responses seem relatively low in magnitude and more centered around zero.
        The fluctuations are somewhat constrained, meaning they might exhibit less variability compared to Cluster 1.

    Cluster 1 (Bottom Graph)
        Rats 3, 11, 15, and 16 belong to this cluster.
        Their responses appear higher in magnitude and generally more positive.
        The fluctuations appear more pronounced, suggesting these rats respond more strongly compared to Cluster 0.

Should You Have Clustered Differently?

Your current approach uses mean, max, and standard deviation as features to cluster rats, which captures overall response intensity but may not fully reflect time-dependent patterns. Here are some considerations:
Potential Issues with Your Approach

    Loss of Temporal Information
        Since you aggregate statistics across time, the clustering does not consider specific response shapes over time (e.g., early vs. late responses).
        Two rats with different response timing but similar mean/std values could end up in the same cluster.

    1D Clustering Might Be Too Simple
        You are clustering using only one feature (rat_features.values.reshape(-1, 1)).
        This could oversimplify the clustering, missing more complex relationships in the data.

    Cluster Boundaries Might Be Arbitrary
        K-Means assumes clusters are spherical, which might not match the real structure of the data.

Alternative Clustering Approaches

    Use Multiple Features for Clustering
        Instead of rat_features.values.reshape(-1, 1), use all three features (mean, max, std) together:

    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(rat_features)

    This way, the algorithm considers all statistical properties instead of reducing them to a single dimension.

Use Dynamic Time Warping (DTW) for Clustering

    DTW measures similarity based on entire response trajectories, not just summary statistics.
    Hierarchical clustering with DTW distance might reveal natural groups in the data.

Principal Component Analysis (PCA) Before Clustering

    PCA can help determine the most important response characteristics and reduce noise.
    Running K-Means on PCA-reduced features may yield better clusters."

