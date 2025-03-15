Timings:

-1 - 4s
0-3 is tone


-1 is first 20, before onset (20)

400ms of 3 is initial onset
100ms gap (arbitrary)
25000ms is hearing, nothing happens
81-88 lever presentation
afterwards lever visible for 1000ms

Possible solutions to the problem with individuality in rats shown by my population_avg data
1. Normalization/Standardize between rats
2. Rat-Specific Identifier as a feature
3. Train individual models for each rat, then meta-learn
4. Cluster Rats Based on Response Types
5. Feature Engineering Focusing on patterns instead of raw data

--------------------------------------------------------------------

1. Normalization of rats.
   The rats are already normalized. Probably not super helpful but I haven't tested it

2. Clustering rats based on response types
    I implemented this but never trained my model on it. Not sure If I want to continue with this.

3. Feature Engineering Focusing on patterns instead of raw data
    Currently implementing this. Particularly using contrastive learning, encoding the data. Performs fairly well.

Would like to continue improving the feature engineering, that is truly where I believe my best results will come from.

### Evaluation Metrics for Encoding / Contrastive Learning

**Embedding Space Quality**
t-SNE / UMAP Visualization: Plot embeddings in 2D/3D to see if similar classes cluster together.

Cosine Similarity Analysis: Compute cosine similarity between embeddings of similar/dissimilar samples to see if the model is organizing them correctly.

**Analyze Loss**
If loss stagnates early, it may not be learning


### Explore Ways to Use the Different Time Windows usefully

**Learningable Attention Weights**
Implement self-attention or a simple MLP to assign importance to bins.

**Explicit Weighting Strategy**
`weights = torch.tensor([0.5]*20 + [1.0]*60 + [0.7]*20).to(device)
weighted_features = weights * feature_vector`

**Contrastive Learning with Time-Aware sampling**
Form positive pairs from different sections of the trial (e.g., compare pre-trial to main event).
This forces the model to align neural activity across different times, extracting richer features.

### Explore Using Dynamic Time Warping (DTW)

How You Might Use DTW

Preprocessing: Align trials using DTW before feeding them into the RNN.
Distance Metric: Use DTW distance to define positive pairs in contrastive learning (closer sequences = positives).
Hybrid Model: Train with NT-Xent, then fine-tune with DTW-based similarity.

### Finally, things to look into:
Use RNN for better sequential representations
Try bidirectional GRU/LSTM for more context
Use DTW for better alignment before contrastive learning
Experiment with removing the projection head

### More data augmentation
Different Positive Pair Strategies:
Currently, (x1, x2) are augmentations of the same sample, maybe:
    Generate positives from neighboring timesteps
    Mix data from similar samples (soft positives)
Hard Negatives: Instead of just randomly sampling negatives, try hard negative mining (e.g., selecting negatives that are closest to positives).

### Projection Head Improvements
Explore adding more layers to the projection head
Explore GeLU as opposed to ReLU (didn't perform very well)
Explore different activation functions
Explore Dropouts / Batchnorm / LayerNorm / Residual Connections

### Better Accuracy Metrics
ROC-AUC: Use ROC-AUC to evaluate the model's ability to distinguish between positive and negative samples
Stratified k-fold cross-validation: Use stratified sampling to ensure each fold has a similar distribution of classes.
**Comparing Two Models**
**Paired t-test (if running multiple trials per model).**
**Wilcoxon signed-rank test (non-parametric alternative).**