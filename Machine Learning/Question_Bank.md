# Machine Learning Practice Problems Bank
## Comprehensive Active Recall Questions for Andrew Ng's ML Specialization

---

# Course 1 Practice Problems

## Linear Regression Mastery

### Quick Recall Questions
1. **Given h(x) = 2x + 3, what's the prediction for x = 5?**
   - Answer: **13** (2×5 + 3)

2. **If gradient descent cost increases, what are 3 possible causes?**
   - Answer: **Learning rate too high, bug in code, wrong derivative calculation**

3. **Cost function J = 100 initially, after 1000 iterations J = 0.1. Good or bad?**
   - Answer: **Good!** Cost decreased significantly

### Application Problems
4. **Real estate dataset: 2000 sq ft house, 3 bedrooms, downtown location. Model: h(x) = 150×sqft + 10000×bedrooms + 50000×location. Price prediction?**
   - Answer: **150×2000 + 10000×3 + 50000×1 = $380,000**

5. **Feature scaling: House sizes range 500-5000 sq ft, prices $50K-$500K. Which needs scaling first?**
   - Answer: **House sizes** (larger range, will dominate gradient updates)

### Conceptual Challenges
6. **Why do we use (prediction - actual)² instead of |prediction - actual| in cost function?**
   - Answer: **Differentiable everywhere, penalizes large errors more, easier to minimize**

7. **Learning rate α = 0.1 works for small dataset. For 100x larger dataset, should α be higher, lower, or same?**
   - Answer: **Lower** (gradient estimates more accurate, can use smaller steps)

## Logistic Regression Deep Dive

### Sigmoid Function Mastery
8. **Sigmoid(0) = ? Sigmoid(∞) = ? Sigmoid(-∞) = ?**
   - Answer: **0.5, 1, 0**

9. **For spam classification, sigmoid output = 0.3. Classify as spam or not spam?**
   - Answer: **Not spam** (0.3 < 0.5 threshold)

10. **Why not use linear regression for classification?**
    - Answer: **Can predict outside [0,1] range, not probabilities**

### Real-World Classification
11. **Medical diagnosis: 1000 patients, 100 have disease. Using accuracy alone, what's a "stupid classifier" that gets 90% accuracy?**
    - Answer: **Always predict "no disease"** (900/1000 = 90% accuracy but useless)

12. **Email features: [num_exclamation_marks=5, contains_money_words=1]. Model: h(x) = sigmoid(0.5×x₁ + 2×x₂ - 1). Spam probability?**
    - z = 0.5×5 + 2×1 - 1 = 3.5
    - Answer: **sigmoid(3.5) ≈ 0.97 → Very likely spam**

## Neural Networks Foundation

### Architecture Understanding
13. **Network: 4 inputs → 6 hidden → 3 outputs. How many total parameters (weights + biases)?**
    - Weights: 4×6 + 6×3 = 42
    - Biases: 6 + 3 = 9
    - Answer: **51 parameters**

14. **Why do we need activation functions in hidden layers?**
    - Answer: **Without them, network is just linear regression (no matter how many layers)**

### Activation Functions
15. **ReLU vs Sigmoid for hidden layers. Which is better and why?**
    - Answer: **ReLU** - avoids vanishing gradient, faster computation, sparse activation

16. **Tanh vs Sigmoid for hidden layers. Key difference?**
    - Answer: **Tanh centered at 0 (range: -1,1) vs Sigmoid (range: 0,1). Tanh often better for hidden layers.**

---

# Course 2 Practice Problems

## Deep Learning Fundamentals

### Forward Propagation
17. **3-layer network: Input [2,3] → Hidden layer with weights [[1,0.5],[0.5,1]] and bias [0.1,0.2] → ReLU. What's hidden layer output?**
    - z = [1×2+0.5×3, 0.5×2+1×3] + [0.1,0.2] = [3.6, 3.2]
    - Answer: **[3.6, 3.2]** (ReLU keeps positive values)

18. **Why do neural networks need many layers for complex tasks?**
    - Answer: **Each layer learns increasingly abstract features (edges→shapes→objects)**

### Training Challenges
19. **Training loss: 0.1, Validation loss: 2.5. What's the problem and 3 solutions?**
    - Problem: **Overfitting**
    - Solutions: **More data, regularization, simpler model**

20. **Gradient descent stuck at loss = 10 for 100 iterations. What to try?**
    - Answer: **Increase learning rate, check for bugs, try different optimizer (Adam)**

## Decision Trees Excellence

### Information Theory
21. **Dataset: 16 cats, 16 dogs. After split: Left=[14 cats, 2 dogs], Right=[2 cats, 14 dogs]. Information gain?**
    - Before: H = 1 (maximum entropy for 50-50 split)
    - After: H_left = -(14/16)log₂(14/16) - (2/16)log₂(2/16) ≈ 0.54
    - H_right ≈ 0.54 (same calculation)
    - Answer: **IG = 1 - 0.5×0.54 - 0.5×0.54 = 0.46**

22. **Which split is better? A: [10 cats, 0 dogs] vs [0 cats, 10 dogs] or B: [7 cats, 3 dogs] vs [3 cats, 7 dogs]?**
    - Answer: **Split A** (creates pure nodes, maximum information gain)

### Tree Building Strategy
23. **Deep tree (depth=20) vs Shallow tree (depth=5). Which likely overfits?**
    - Answer: **Deep tree** - memorizes training data, poor generalization

24. **1000 features, 100 training examples. Decision tree will likely:**
    - Answer: **Overfit severely** - too many features relative to data

## Ensemble Methods Mastery

### Random Forest
25. **Random Forest with 50 trees. New example: 30 predict Class A, 20 predict Class B. Final prediction?**
    - Answer: **Class A** (majority vote: 30 > 20)

26. **Random Forest vs Single Decision Tree. Why is Random Forest better?**
    - Answer: **Reduces overfitting, more robust, averages out individual tree errors**

### Boosting Concepts
27. **XGBoost training. First model error = 2.0, second model error = 1.5, third model error = 1.2. What's happening?**
    - Answer: **Each model learns from previous errors, gradually improving**

28. **When to use Random Forest vs XGBoost?**
    - Random Forest: **Fast training, good baseline**
    - XGBoost: **Better accuracy, more tuning needed**

---

# Course 3 Practice Problems

## Clustering Intelligence

### K-Means Deep Understanding
29. **Points: (0,0), (1,1), (10,10), (11,11). K=2. Optimal centroids?**
    - Cluster 1: (0,0), (1,1) → Centroid: **(0.5, 0.5)**
    - Cluster 2: (10,10), (11,11) → Centroid: **(10.5, 10.5)**

30. **K-means fails when clusters are:**
    - Answer: **Non-circular (elongated), different sizes, overlapping**

### Choosing K
31. **Elbow method shows costs: K=1→100, K=2→50, K=3→45, K=4→42, K=5→40. Best K?**
    - Answer: **K=3** (biggest drop in cost improvement)

32. **Business context: Customer segmentation for marketing. How to choose K?**
    - Answer: **Business constraints** - how many campaigns can you run? Budget for different strategies?

## Anomaly Detection Excellence

### Gaussian Modeling
33. **Server CPU usage: μ=30%, σ²=25. Usage = 50% with threshold ε=0.05. Anomaly?**
    - Calculate p(50) with Gaussian formula
    - If p(50) < 0.05 → Answer: **Yes, anomaly**

34. **Features: CPU usage, Memory usage, Network traffic. They're correlated. Which model?**
    - Answer: **Multivariate Gaussian** (captures feature correlations)

### Threshold Setting
35. **Anomaly detection for credit cards. Setting ε too low vs too high:**
    - Too low: **Miss real fraud** (false negatives)
    - Too high: **Too many false alarms** (false positives)

## Recommender Systems Expertise

### Collaborative Filtering
36. **User-Item matrix missing 60% of ratings. Matrix factorization finds:**
    - Answer: **Hidden factors that predict missing ratings**

37. **Netflix problem: New user rates 3 movies. Cold start solution?**
    - Answer: **Content-based filtering** (use movie features, not collaborative data)

### Recommendation Strategies
38. **User loved: Godfather, Goodfellas, Scarface. Recommend Romantic Comedy?**
    - Content-based: **No** (different genre)
    - Collaborative: **Maybe** (if similar users also like rom-coms)

## PCA Mastery

### Dimensionality Reduction
39. **1000 features explain variance: [λ₁=500, λ₂=200, λ₃=100, λ₄...λ₁₀₀₀=1 each]. Keep 95% variance, how many components?**
    - Total variance = 500+200+100+897×1 = 1697
    - First 3 components: 800/1697 = 47%
    - Need more calculation, but answer ≈ **first 50-100 components**

40. **Image compression: 256×256 image (65,536 pixels) → 100 features. Compression ratio?**
    - Answer: **655:1** (65,536/100 = 655.36)

### When NOT to Use PCA
41. **Medical diagnosis features: [blood_pressure, cholesterol, age]. Use PCA?**
    - Answer: **No** - features have interpretable medical meaning, mixing them loses interpretability

---

# Advanced Integration Problems

## Cross-Course Connections

### Supervised + Unsupervised
42. **Customer dataset: Use unsupervised learning to find 3 customer types, then supervised learning to predict purchase behavior for each type. Why this approach?**
    - Answer: **Unsupervised finds hidden patterns, supervised builds targeted models for each segment**

43. **Anomaly detection finds outliers in training data. Should you remove them before training supervised model?**
    - Answer: **Depends** - if data errors, remove. If rare but valid cases, keep (might be important edge cases)

### Real-World Pipeline
44. **E-commerce recommendation system: 10M users, 1M products, 95% sparse ratings. Design complete system:**
    - Preprocessing: **Handle sparsity, scale features**
    - Cold start: **Content-based for new users/items**
    - Main system: **Matrix factorization for existing users**
    - Evaluation: **Hold-out test set, A/B testing**

### Business Impact
45. **ML project ROI: Model improves accuracy from 85% to 92%. Training costs $100K, expected savings $1M/year. Continue project?**
    - Technical: **7% improvement is significant**
    - Business: **10:1 ROI in first year**
    - Answer: **Yes, strong business case**

## Debugging Scenarios

### Common Failures
46. **Scenario: Training accuracy = 45%, Random guessing = 50%. What's wrong?**
    - Answer: **Bug in code** (worse than random suggests implementation error)

47. **Model works great in testing, fails in production. Possible causes:**
    - **Data drift** (production data different from training)
    - **Label leakage** (used future information in training)
    - **Different data preprocessing** in production

### Performance Optimization
48. **Large dataset (10M examples), training too slow. Speed up strategies:**
    - **Batch gradient descent** → Mini-batch
    - **Feature selection** (remove irrelevant features)
    - **Better hardware** (GPU, more RAM)
    - **Sampling** (train on subset initially)

---

# Speed Learning Challenges
*Test your rapid recall abilities*

## 30-Second Lightning Round
Answer each in 5 seconds or less:

49. **Sigmoid derivative maximum point?** → **x=0 (output=0.25)**
50. **Decision tree pure leaf entropy?** → **0**
51. **K-means optimal K for unknown data?** → **Elbow method**
52. **PCA first component direction?** → **Maximum variance**
53. **Overfitting indicator?** → **Train accuracy > Test accuracy**
54. **Neural network universal approximator?** → **Yes, with enough hidden units**
55. **Random Forest reduce what?** → **Variance (overfitting)**
56. **Collaborative filtering needs what?** → **User-item interaction data**
57. **Anomaly detection assumes what distribution?** → **Gaussian/Normal**
58. **XGBoost learns from what?** → **Previous model errors**

## Integration Master Problems

### Complete ML Pipeline
59. **Design ML system for autonomous vehicle object detection:**
    - Data: **Camera images with labeled objects**
    - Preprocessing: **Image normalization, data augmentation**
    - Model: **Convolutional Neural Network**
    - Validation: **Hold-out test set, real-world testing**
    - Production: **Real-time inference, safety monitoring**

60. **Fraud detection system: 1M transactions/day, 0.1% fraud rate. Complete approach:**
    - Challenge: **Highly imbalanced data**
    - Model: **Ensemble (Random Forest + Neural Network)**
    - Evaluation: **Precision/Recall (not accuracy)**
    - Deployment: **Real-time scoring, human review pipeline**
    - Monitoring: **Concept drift detection, performance tracking**

---

# Answer Key Summary

## Core Formulas You Must Know
- **Linear Regression**: h(x) = wx + b, Cost = (1/2m)Σ(h(x)-y)²
- **Logistic Regression**: h(x) = 1/(1+e⁻ᶻ), Cost = -Σ[y log(h) + (1-y)log(1-h)]
- **Neural Network**: z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾, a⁽ˡ⁾ = g(z⁽ˡ⁾)
- **Entropy**: H = -Σ p_i log₂(p_i)
- **K-means Cost**: J = (1/m)Σ||x⁽ⁱ⁾ - μc⁽ⁱ⁾||²
- **Gaussian**: p(x) = (1/√(2πσ²)) e^(-(x-μ)²/(2σ²))

## Problem-Solving Framework
1. **Identify problem type** (supervised/unsupervised, regression/classification)
2. **Check data quality** (missing values, outliers, scale)
3. **Choose appropriate model** (based on data size, interpretability needs)
4. **Validate properly** (train/val/test splits, cross-validation)
5. **Evaluate with right metrics** (accuracy, precision/recall, business metrics)
6. **Deploy and monitor** (performance tracking, drift detection)

*Use these problems for spaced repetition practice - focus on areas where you struggled!*
