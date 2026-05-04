import numpy as np
import pandas as pd

data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
                'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal',
                 'High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong',
             'Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes',
                   'No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)
target = 'PlayTennis'

def entropy(labels):
    """
    Calculates Shannon entropy for a list/array of labels.
    H(S) = -Σ p_i * log2(p_i)
    """
    n = len(labels)
    if n == 0:
        return 0.0

    counts = np.unique(labels, return_counts=True)[1]
    probabilities = counts / n
    probabilities = probabilities[probabilities > 0]

    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(df, feature, target):
    """
    IG(S, A) = H(S) - Σ (|S_v| / |S|) * H(S_v)
    """
    total_entropy = entropy(df[target])
    n = len(df)

    weighted_entropy = 0.0
    for value in df[feature].unique():
        subset = df[df[feature] == value][target]
        weight = len(subset) / n
        weighted_entropy += weight * entropy(subset)

    return total_entropy - weighted_entropy

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

print(f"Dataset entropy H(S) = {entropy(df[target]):.4f} bits\n")
print(f"{'Feature':<15} {'IG':>10}")
print("-" * 27)

ig_scores = {}
for feature in features:
    ig = information_gain(df, feature, target)
    ig_scores[feature] = ig
    print(f"{feature:<15} {ig:>10.4f}")

best_feature = max(ig_scores, key=ig_scores.get)
print(f"\nBest feature to split on: {best_feature} (IG = {ig_scores[best_feature]:.4f})")
