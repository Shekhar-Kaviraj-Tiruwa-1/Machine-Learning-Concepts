# Andrew Ng's Machine Learning Specialization - Course 1: Supervised Machine Learning
## Complete Study Guide with Practice Problems

---

# Week 1: Introduction to Machine Learning

## Key Concepts Summary
**Machine Learning** = Computer programs that improve performance on a task through experience
- **Supervised Learning**: Learn from input-output pairs (labeled data)
- **Unsupervised Learning**: Find patterns in data without labels

### Analogies to Remember
- **Supervised Learning** = Learning with a teacher who shows you the right answers
- **Unsupervised Learning** = Finding hidden patterns like a detective with no clues

### Essential Definitions
- **Training Set**: Data used to train the model
- **Feature (x)**: Input variable 
- **Target (y)**: Output variable to predict
- **m**: Number of training examples
- **Hypothesis h**: Function that maps x to y

### Practice Problems
1. **Classify these as supervised or unsupervised:**
   - Email spam detection → **Supervised** (has labels: spam/not spam)
   - Customer segmentation → **Unsupervised** (finding groups)

2. **Given housing data: size, bedrooms, location → price**
   - What's the feature? → **Size, bedrooms, location**
   - What's the target? → **Price**

### Real-World Application
**Netflix Recommendations**: Supervised learning uses your ratings (labels) to predict what you'll like

### Connection to Advanced Topics
This foundation leads to deep learning, where we stack multiple layers of these basic supervised learning units

### 10-Second Revision
*"Supervised = with labels, Unsupervised = find patterns, x=input, y=output"*

---

# Week 2: Linear Regression

## Key Concepts Summary
**Linear Regression** predicts continuous values using a straight line relationship
- **Goal**: Find the best line through data points
- **Cost Function**: Measures how wrong our predictions are
- **Gradient Descent**: Algorithm to find the best line

### Analogy
Like drawing the "best fit" line through scattered dots on paper - the line that's closest to most points

### Essential Formulas to Memorize
```
Hypothesis: h(x) = w₁x + w₀
Cost Function: J(w) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
Gradient Descent: w = w - α(∂J/∂w)

Where:
- w₁ = slope (weight)
- w₀ = y-intercept (bias) 
- α = learning rate
- m = number of examples
```

### Practice Problems
1. **Given h(x) = 3x + 2, predict house price for 1500 sq ft**
   - Answer: h(1500) = 3(1500) + 2 = **4502**

2. **If cost increases after gradient descent step, what's wrong?**
   - Answer: **Learning rate α is too high**

3. **Cost function value of 0 means:**
   - Answer: **Perfect predictions (no error)**

### Real-World Application
**Zillow Home Estimates**: Uses square footage, bedrooms, location to predict house prices using linear regression

### Connection to Advanced Topics
- Multiple features → Multivariate linear regression
- Non-linear relationships → Polynomial features
- Regularization prevents overfitting

### 10-Second Revision
*"Line through dots, minimize squared errors, gradient descent finds best w"*

---

# Week 3: Logistic Regression

## Key Concepts Summary
**Logistic Regression** predicts categories/classes (0 or 1)
- Uses **Sigmoid Function** to squeeze outputs between 0 and 1
- **Decision Boundary**: Line/curve that separates classes
- Different cost function than linear regression

### Analogy
Like a "smart switch" that decides ON/OFF based on multiple factors, with a smooth transition zone

### Essential Formulas to Memorize
```
Sigmoid: g(z) = 1/(1 + e⁻ᶻ)
Hypothesis: h(x) = g(w₁x + w₀) = 1/(1 + e⁻⁽ʷ¹ˣ⁺ʷ⁰⁾)
Cost Function: J(w) = -(1/m) Σ[y log(h(x)) + (1-y) log(1-h(x))]
Decision: If h(x) ≥ 0.5 → predict 1, else predict 0

Where:
- g(z) = sigmoid function
- z = w₁x + w₀ (linear combination)
```

### Practice Problems
1. **If sigmoid output = 0.7, what do we predict?**
   - Answer: **Class 1** (since 0.7 ≥ 0.5)

2. **When is sigmoid = 0.5?**
   - Answer: **When z = 0** (right at decision boundary)

3. **Email spam: features = [keyword_count, sender_domain]. If h(x) = 0.8:**
   - Answer: **Classify as SPAM** (80% probability)

### Real-World Application
**Medical Diagnosis**: Predict if patient has disease (Yes/No) based on symptoms, test results

### Connection to Advanced Topics
- Multiple classes → Softmax regression
- Non-linear boundaries → Polynomial features
- Neural networks use sigmoid in hidden layers

### 10-Second Revision
*"Sigmoid squashes to 0-1, threshold at 0.5, different cost function than linear"*

---

# Week 4: Neural Networks Basics

## Key Concepts Summary
**Neural Networks** = Multiple logistic regression units connected in layers
- **Input Layer**: Features go in
- **Hidden Layer(s)**: Process information
- **Output Layer**: Final prediction
- Each connection has a weight, each neuron has bias

### Analogy
Like a **team decision**: Input → specialists analyze → team leader decides
Brain neurons: receive signals → process → send output

### Essential Architecture
```
Input Layer → Hidden Layer → Output Layer
    x₁  →      a₁⁽²⁾    →     h(x)
    x₂  →      a₂⁽²⁾    →  
    x₃  →      a₃⁽²⁾    →

Layer notation:
- a⁽ˡ⁾ = activation in layer l
- W⁽ˡ⁾ = weights from layer l-1 to layer l
- b⁽ˡ⁾ = bias for layer l
```

### Practice Problems
1. **Network: 4 inputs → 3 hidden → 1 output. How many weights?**
   - Input to hidden: 4×3 = 12
   - Hidden to output: 3×1 = 3
   - Answer: **15 weights total**

2. **If hidden layer has 5 neurons, how many bias terms?**
   - Answer: **5 bias terms** (one per neuron)

### Real-World Application
**Image Recognition**: Each pixel = input, hidden layers detect edges→shapes→objects, output = classification

### Connection to Advanced Topics
- More layers = Deep Learning
- Convolutional layers for images
- Recurrent layers for sequences

### 10-Second Revision
*"Stacked logistic regression, layers process step-by-step, weights connect everything"*

---

# Quick Reference Cards

## Formula Sheet
| Concept | Formula | Key Variables |
|---------|---------|---------------|
| Linear Regression | h(x) = wx + b | w=weight, b=bias |
| Cost (Linear) | J = (1/2m)Σ(h(x)-y)² | m=examples |
| Logistic Regression | h(x) = 1/(1+e⁻ᶻ) | z=wx+b |
| Cost (Logistic) | J = -(1/m)Σ[y log(h) + (1-y)log(1-h)] | |
| Gradient Descent | w = w - α(∂J/∂w) | α=learning rate |

## Problem-Solving Checklist
1. **Linear or Classification?** → Choose regression type
2. **Check cost function** → Should decrease over time  
3. **Learning rate issues?** → Too high = diverges, too low = slow
4. **Overfitting?** → Add regularization
5. **Underfitting?** → Add features or complexity

## Common Mistakes to Avoid
- ❌ Using linear regression for classification
- ❌ Learning rate too high (cost explodes)
- ❌ Forgetting to normalize features
- ❌ Not checking if gradient descent converges
- ❌ Using wrong cost function

---

*This completes Course 1 foundation. Master these concepts before moving to advanced algorithms!*
