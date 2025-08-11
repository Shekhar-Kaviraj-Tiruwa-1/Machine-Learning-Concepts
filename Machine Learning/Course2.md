# Andrew Ng's Machine Learning Specialization - Course 2: Advanced Learning Algorithms
## Complete Study Guide with Practice Problems

---

# Week 1: Advanced Neural Networks

## Key Concepts Summary
**Deep Neural Networks** = Multiple hidden layers for complex pattern recognition
- **Forward Propagation**: Data flows input → hidden → output
- **Backpropagation**: Errors flow backward to update weights
- **Activation Functions**: Non-linear functions that add complexity

### Analogies to Remember
- **Deep Network** = Assembly line where each station adds more sophistication
- **Backpropagation** = Getting feedback from final customer, passing corrections backward through the factory

### Essential Formulas to Memorize
```
Forward Propagation:
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = g(z⁽ˡ⁾)

Common Activation Functions:
- ReLU: g(z) = max(0, z)
- Sigmoid: g(z) = 1/(1 + e⁻ᶻ)
- Tanh: g(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)

Where:
- l = layer number
- W⁽ˡ⁾ = weight matrix for layer l
- b⁽ˡ⁾ = bias vector for layer l
- a⁽ˡ⁾ = activation (output) of layer l
```

### Practice Problems
1. **Network: 3 inputs → 4 hidden → 2 hidden → 1 output. How many weight matrices?**
   - Answer: **3 weight matrices** (W⁽¹⁾, W⁽²⁾, W⁽³⁾)

2. **ReLU(−3) and ReLU(5) equal:**
   - Answer: **0 and 5** (ReLU kills negative values)

3. **Why use ReLU over sigmoid in hidden layers?**
   - Answer: **Avoids vanishing gradient problem, faster computation**

### Real-World Application
**Computer Vision**: Deep networks recognize faces by detecting edges → shapes → facial features → identity

### Connection to Advanced Topics
Foundation for CNNs (computer vision), RNNs (sequences), and Transformers (language)

### 10-Second Revision
*"Deep = many layers, ReLU for hidden, backprop updates weights, forward flows data"*

---

# Week 2: Neural Network Training

## Key Concepts Summary
**Training Process**: Optimize weights to minimize prediction error
- **Cost Function**: Measures total network error
- **Gradient Descent**: Updates all weights simultaneously
- **Learning Rate Scheduling**: Adjust α during training

### Analogies to Remember
- **Training** = Teaching orchestra - each musician (neuron) adjusts their performance based on conductor's feedback
- **Learning Rate** = Step size when hiking - too big = overshoot peak, too small = takes forever

### Essential Training Concepts
```
Cost Function (Multi-class):
J = -(1/m) Σᵢ Σₖ y_k⁽ⁱ⁾ log(a_k⁽ᴸ⁾⁽ⁱ⁾)

Gradient Descent Update:
W⁽ˡ⁾ := W⁽ˡ⁾ - α ∂J/∂W⁽ˡ⁾
b⁽ˡ⁾ := b⁽ˡ⁾ - α ∂J/∂b⁽ˡ⁾

Key Parameters:
- α = learning rate
- epochs = full passes through data
- batch size = examples processed together
```

### Practice Problems
1. **If cost increases during training, what might be wrong?**
   - Answer: **Learning rate too high, or bug in implementation**

2. **Training accuracy 99%, test accuracy 70%. What's the issue?**
   - Answer: **Overfitting** - model memorized training data

3. **Cost decreases but very slowly. What to try?**
   - Answer: **Increase learning rate, check gradient computation**

### Real-World Application
**Language Translation**: Deep networks learn to map English → French by training on millions of sentence pairs

### Connection to Advanced Topics
Leads to advanced optimizers (Adam, RMSprop), batch normalization, dropout regularization

### 10-Second Revision
*"Minimize cost, adjust α carefully, watch for overfitting, gradient descent updates all weights"*

---

# Week 3: Decision Trees

## Key Concepts Summary
**Decision Trees** = Series of yes/no questions leading to predictions
- **Entropy**: Measures how "mixed up" the data is
- **Information Gain**: Reduction in entropy after a split
- **Tree Building**: Choose splits that maximize information gain

### Analogies to Remember
- **Decision Tree** = 20 questions game - each question narrows down possibilities
- **Entropy** = How messy your room is (high = very messy, low = organized)

### Essential Formulas to Memorize
```
Entropy: H(S) = -Σ p_i log₂(p_i)

Information Gain: 
IG = H(parent) - Σ (|S_child|/|S_parent|) × H(S_child)

Where:
- p_i = proportion of class i
- S = set of examples
- |S| = number of examples in set S
- Pure node: H = 0 (all same class)
- Most impure: H = 1 (50-50 split for binary)
```

### Practice Problems
1. **Node with 8 cats, 2 dogs. What's the entropy?**
   - p_cat = 8/10 = 0.8, p_dog = 2/10 = 0.2
   - H = -(0.8×log₂(0.8) + 0.2×log₂(0.2)) = **0.72**

2. **Pure leaf node (all same class) has entropy:**
   - Answer: **0** (no uncertainty)

3. **Best split maximizes:**
   - Answer: **Information Gain** (reduces entropy most)

### Real-World Application
**Medical Diagnosis**: Is fever > 100°F? → Yes: Is cough present? → Yes: Likely flu, No: Check other symptoms

### Connection to Advanced Topics
Foundation for Random Forests, Gradient Boosting (XGBoost), ensemble methods

### 10-Second Revision
*"Yes/no questions, maximize info gain, entropy measures messiness, pure nodes have H=0"*

---

# Week 4: Ensemble Methods

## Key Concepts Summary
**Ensemble Methods** = Combine multiple models for better performance
- **Random Forest**: Many decision trees vote on final prediction
- **Bagging**: Train models on different data subsets
- **Boosting**: Sequential models, each fixes previous errors

### Analogies to Remember
- **Random Forest** = Asking advice from many experts, then taking majority vote
- **Boosting** = Study group where each person learns from previous person's mistakes

### Essential Ensemble Concepts
```
Random Forest:
1. Create B bootstrap samples
2. Train decision tree on each sample
3. For prediction: majority vote (classification) or average (regression)

Bagging Process:
- Bootstrap: Sample with replacement
- Aggregate: Combine predictions
- Reduces variance, fights overfitting

XGBoost (Gradient Boosting):
- Sequential models: F₁, F₂, ..., Fₘ
- Each model learns residual errors
- Final prediction: F(x) = F₁(x) + F₂(x) + ... + Fₘ(x)
```

### Practice Problems
1. **Random Forest with 100 trees. For new example, 60 predict class A, 40 predict class B:**
   - Answer: **Predict class A** (majority vote)

2. **Bootstrap sample from dataset of size 1000:**
   - Answer: **Sample 1000 examples with replacement** (some repeated, some missing)

3. **Boosting vs Bagging difference:**
   - Answer: **Boosting = sequential (learn from errors), Bagging = parallel (independent models)**

### Real-World Application
**Kaggle Competitions**: Ensemble methods (XGBoost, Random Forest) dominate structured data competitions

### Connection to Advanced Topics
Modern ensembles: Stacking, Blending, Neural ensemble methods

### 10-Second Revision
*"Many models vote, Random Forest parallel, Boosting sequential, reduces overfitting"*

---

# Advanced Topics Overview

## Regularization Techniques

### L1 and L2 Regularization
```
L1 (Lasso): J = Cost + λ Σ|wᵢ|
L2 (Ridge): J = Cost + λ Σwᵢ²

Effects:
- L1: Creates sparse models (some weights = 0)
- L2: Shrinks weights toward 0
- λ = regularization strength
```

### Dropout
- Randomly "turn off" neurons during training
- Prevents over-reliance on specific features
- Typical rate: 0.2-0.5

## Optimization Improvements

### Adam Optimizer
```
Combines momentum + adaptive learning rates
Better than basic gradient descent
Hyperparameters: α=0.001, β₁=0.9, β₂=0.999
```

### Learning Rate Scheduling
- Start high, gradually decrease
- Step decay, exponential decay, cosine annealing

## Model Evaluation

### Cross-Validation
```
k-Fold CV:
1. Split data into k parts
2. Train on k-1 parts, test on 1 part
3. Repeat k times
4. Average performance
```

### Metrics Beyond Accuracy
- **Precision**: TP/(TP+FP) - "When I predict positive, how often am I right?"
- **Recall**: TP/(TP+FN) - "Of all actual positives, how many did I catch?"
- **F1-Score**: 2×(Precision×Recall)/(Precision+Recall)

---

# Quick Reference - Course 2

## Architecture Decisions
| Problem Type | Best Algorithm | Why? |
|--------------|----------------|------|
| Images | Deep Neural Network | Hierarchical feature learning |
| Structured Data | XGBoost/Random Forest | Handles mixed data types well |
| Large Dataset | Neural Network | Scales well with data |
| Small Dataset | Decision Tree/SVM | Less prone to overfitting |
| Interpretability Needed | Decision Tree | Easy to explain |

## Hyperparameter Tuning Guide
1. **Learning Rate**: Start with 0.001, adjust by 3x (0.0003, 0.001, 0.003)
2. **Network Depth**: Start shallow (2-3 layers), increase if underfitting  
3. **Tree Depth**: Start with 6-10, reduce if overfitting
4. **Regularization**: Start with λ=0.01, adjust based on validation performance

## Debugging Checklist
- ✅ **High bias (underfitting)**: Add features, increase model complexity
- ✅ **High variance (overfitting)**: Add regularization, get more data
- ✅ **Slow training**: Check learning rate, batch size, network architecture
- ✅ **Poor generalization**: Cross-validation, regularization, ensemble methods

---

*Master these advanced algorithms to handle any supervised learning problem!*
