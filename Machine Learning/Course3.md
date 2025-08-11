# Andrew Ng's Machine Learning Specialization - Course 3: Unsupervised Learning
## Complete Study Guide with Practice Problems

---

# Week 1: K-Means Clustering

## Key Concepts Summary
**K-Means Clustering** = Group similar data points together without labels
- **Centroids**: Center points of each cluster
- **Algorithm**: Assign points → Update centroids → Repeat until convergence
- **Choose K**: Number of clusters (often requires domain knowledge)

### Analogies to Remember
- **K-Means** = Party planning - group people by interests, find center of each group, readjust groups
- **Centroids** = Meeting points where each group naturally gathers

### Essential Algorithm Steps
```
K-Means Algorithm:
1. Initialize K centroids randomly
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids to mean of assigned points

Distance Metric:
d(x,μ) = ||x - μ||² = Σᵢ(xᵢ - μᵢ)²

Cost Function (Distortion):
J = (1/m) Σᵢ ||x⁽ⁱ⁾ - μc⁽ⁱ⁾||²

Where:
- μ = centroid
- c⁽ⁱ⁾ = cluster assignment for point i
- K = number of clusters
```

### Practice Problems
1. **Points: (1,1), (1,2), (8,8), (8,9). K=2. After first iteration with centroids at (0,0), (9,9):**
   - Group 1: (1,1), (1,2) → New centroid: **(1, 1.5)**
   - Group 2: (8,8), (8,9) → New centroid: **(8, 8.5)**

2. **How to choose K when you don't know?**
   - Answer: **Elbow method** - plot cost vs K, look for "elbow" bend

3. **K-means assumes clusters are:**
   - Answer: **Roughly circular/spherical and similar sizes**

### Real-World Application
**Customer Segmentation**: Group customers by spending patterns to create targeted marketing campaigns

### Connection to Advanced Topics
Foundation for Gaussian Mixture Models, spectral clustering, hierarchical clustering

### 10-Second Revision
*"Group similar points, move centroids to center, repeat until stable, choose K carefully"*

---

# Week 2: Anomaly Detection

## Key Concepts Summary
**Anomaly Detection** = Find unusual/outlying data points
- **Gaussian Distribution**: Model normal behavior with bell curves
- **Multivariate Gaussian**: Handle correlations between features
- **Threshold ε**: Boundary between normal and anomalous

### Analogies to Remember
- **Anomaly Detection** = Airport security - most passengers are normal, flag the unusual ones
- **Gaussian Model** = Drawing a fence around "normal neighborhood" - anything outside is suspicious

### Essential Formulas to Memorize
```
Univariate Gaussian:
p(x) = (1/√(2πσ²)) e^(-(x-μ)²/(2σ²))

Multivariate Gaussian:
p(x) = (1/√((2π)ᵏ|Σ|)) e^(-½(x-μ)ᵀΣ⁻¹(x-μ))

Anomaly Detection Rule:
If p(x) < ε → Anomaly
If p(x) ≥ ε → Normal

Parameter Estimation:
μ = (1/m) Σᵢ x⁽ⁱ⁾
σ² = (1/m) Σᵢ (x⁽ⁱ⁾ - μ)²

Where:
- μ = mean
- σ² = variance  
- Σ = covariance matrix
- ε = threshold
- k = number of features
```

### Practice Problems
1. **Given μ=5, σ²=4. Is x=1 anomalous with ε=0.01?**
   - p(1) = (1/√(2π×4)) × e^(-(1-5)²/(2×4)) = **0.0027**
   - Since 0.0027 < 0.01 → **Yes, anomaly**

2. **Features are correlated. Which model?**
   - Answer: **Multivariate Gaussian** (captures correlations)

3. **What happens if ε is too small?**
   - Answer: **Miss real anomalies** (too strict)

### Real-World Application
**Credit Card Fraud**: Model normal spending patterns, flag transactions that deviate significantly

### Connection to Advanced Topics
Leads to isolation forests, one-class SVM, deep autoencoders for anomaly detection

### 10-Second Revision
*"Model normal with Gaussian, flag low probability, choose threshold ε carefully"*

---

# Week 3: Recommender Systems

## Key Concepts Summary
**Recommender Systems** = Predict user preferences for items
- **Content-Based**: Recommend based on item features
- **Collaborative Filtering**: Use other users' preferences  
- **Matrix Factorization**: Find hidden patterns in user-item interactions

### Analogies to Remember
- **Content-Based** = "You liked action movies, here's another action movie"
- **Collaborative Filtering** = "People similar to you also liked these items"
- **Matrix Factorization** = Finding hidden DNA that explains why users like certain items

### Essential Algorithms
```
Content-Based Filtering:
For user j, item i: prediction = θⱼᵀ xᵢ
Where θⱼ = user j's preferences, xᵢ = item i's features

Collaborative Filtering Cost:
J = ½ Σ(i,j) (θⱼᵀxᵢ - yᵢⱼ)² + regularization

Matrix Factorization:
R ≈ UV^T
Where:
- R = user-item rating matrix (m×n)
- U = user feature matrix (m×k)  
- V = item feature matrix (n×k)
- k = number of latent factors
```

### Practice Problems
1. **User likes: Action=5, Comedy=2, Romance=1. New movie: Action=0.8, Comedy=0.1, Romance=0.1:**
   - Prediction = 5×0.8 + 2×0.1 + 1×0.1 = **4.3**

2. **Cold start problem occurs when:**
   - Answer: **New user/item with no historical data**

3. **Matrix factorization finds:**
   - Answer: **Hidden features** that explain user-item relationships

### Real-World Application
**Netflix**: Combines content features (genre, actors) with collaborative patterns to suggest movies you'll enjoy

### Connection to Advanced Topics
Deep learning recommenders, neural collaborative filtering, attention mechanisms

### 10-Second Revision
*"Content uses features, collaborative uses others' preferences, matrix factorization finds hidden patterns"*

---

# Week 4: Principal Component Analysis (PCA)

## Key Concepts Summary
**PCA** = Reduce data dimensions while preserving most information
- **Principal Components**: Directions of maximum variance
- **Dimensionality Reduction**: Compress n features → k features (k < n)
- **Data Visualization**: Project high-D data to 2D/3D plots

### Analogies to Remember
- **PCA** = Taking a shadow of a 3D object - lose some info but keep main shape
- **Principal Components** = Finding the "main directions" where data varies most

### Essential PCA Process
```
PCA Algorithm:
1. Normalize features (zero mean, unit variance)
2. Compute covariance matrix: Σ = (1/m) XᵀX
3. Compute eigenvectors of Σ
4. Choose k largest eigenvectors (principal components)
5. Project data: z = Uₖᵀ x

Choosing k (number of components):
Keep enough to retain 95-99% of variance:
Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ⁿ λᵢ ≥ 0.95

Where:
- λᵢ = eigenvalues (variance explained)
- Uₖ = first k eigenvectors
- x = original data
- z = compressed data
```

### Practice Problems
1. **100 features, want to keep 95% variance. First 20 components explain 96% variance:**
   - Answer: **Use k=20** (exceeds 95% threshold)

2. **PCA on images reduces 1000 pixels to 50 features. Compression ratio:**
   - Answer: **20:1** (1000/50 = 20)

3. **When should you NOT use PCA?**
   - Answer: **When features have important meanings that shouldn't be mixed**

### Real-World Application
**Image Compression**: Reduce photo file sizes by keeping only the most important visual patterns

### Connection to Advanced Topics
t-SNE, UMAP for visualization; autoencoders for non-linear dimensionality reduction

### 10-Second Revision
*"Find directions of max variance, project data, keep 95-99% of information with fewer features"*

---

# Advanced Unsupervised Techniques

## Hierarchical Clustering

### Agglomerative (Bottom-up)
```
Algorithm:
1. Start: each point is its own cluster
2. Repeatedly merge closest clusters
3. Stop when desired number of clusters
4. Creates dendrogram (tree structure)

Linkage Methods:
- Single: min distance between clusters
- Complete: max distance between clusters  
- Average: average distance between clusters
```

### When to Use
- Don't know K in advance
- Want hierarchical structure
- Small to medium datasets

## Gaussian Mixture Models (GMM)

### Soft Clustering
```
Unlike K-means (hard assignment), GMM gives probabilities:
p(cluster i | x) = probability x belongs to cluster i

Parameters to learn:
- μᵢ = mean of cluster i
- Σᵢ = covariance of cluster i
- πᵢ = mixing coefficient (cluster weight)
```

### Advantages over K-means
- Handles elliptical clusters
- Gives uncertainty estimates
- More flexible cluster shapes

## Association Rule Mining

### Market Basket Analysis
```
Find patterns like: "If buy bread and milk → likely buy eggs"

Key Metrics:
- Support: P(A ∩ B) = frequency of itemset
- Confidence: P(B|A) = support(A∪B) / support(A)
- Lift: Confidence / P(B) = how much A increases probability of B
```

---

# Complete ML Pipeline

## Data Preprocessing
1. **Handle missing values**: Imputation or removal
2. **Scale features**: StandardScaler, MinMaxScaler
3. **Encode categories**: One-hot, label encoding
4. **Feature selection**: Remove irrelevant features

## Model Selection Strategy
```
Supervised Learning:
- Linear data → Linear/Logistic Regression
- Non-linear, small data → Decision Trees, SVM
- Large data, complex patterns → Neural Networks
- Structured data competitions → XGBoost

Unsupervised Learning:
- Grouping customers → K-Means
- Find outliers → Anomaly Detection
- Recommendations → Collaborative Filtering
- Reduce dimensions → PCA
```

## Validation Framework
```
1. Split: Train (60%) / Validation (20%) / Test (20%)
2. Cross-validation on train+validation
3. Hyperparameter tuning on validation set
4. Final evaluation on test set (once!)
5. Monitor for overfitting throughout
```

## Production Considerations
- **Model drift**: Performance degrades over time
- **A/B testing**: Compare new vs old models
- **Monitoring**: Track prediction quality
- **Retraining**: Update models with new data

---

# Quick Reference - Course 3

## Algorithm Comparison Table
| Algorithm | Use Case | Key Parameters | Pros | Cons |
|-----------|----------|----------------|------|------|
| K-Means | Customer segments | K, max_iter | Fast, simple | Assumes spherical clusters |
| DBSCAN | Outlier detection | eps, min_samples | Finds arbitrary shapes | Sensitive to parameters |
| PCA | Dimensionality reduction | n_components | Linear, interpretable | Only linear relationships |
| GMM | Soft clustering | n_components, covariance_type | Probabilistic, flexible shapes | More complex than K-means |

## Choosing Number of Components/Clusters
1. **K-means**: Elbow method, silhouette score
2. **PCA**: Cumulative explained variance (95-99%)
3. **GMM**: Information criteria (AIC, BIC)
4. **Hierarchical**: Dendrogram visual inspection

## Common Pitfalls
- ❌ Not scaling features before PCA/clustering
- ❌ Using PCA when features have meaning
- ❌ Setting threshold ε without validation data
- ❌ Ignoring domain knowledge when choosing K
- ❌ Not checking for data leakage in recommender systems

---

*Complete mastery of supervised + unsupervised learning! You're now ready for advanced ML and deep learning specializations.*
