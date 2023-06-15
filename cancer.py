import numpy as np
import pandas as pd

def euclidean_distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum(axis=1))

def nearest_neighbor(train_X, test_X):
    distance = euclidean_distance(train_X, test_X)
    idx = distance.argmin()
    return idx

def forward_selection(train_X, test_X, train_y, test_y):
    num_features = train_X.shape[1]
    selected_features = []
    best_accuracy = 0.0
    best_features = None
    
    for _ in range(num_features):
        feature_accuracies = []

        for feature in range(num_features):
            if feature not in selected_features:
                selected_features.append(feature)
                accuracy = evaluate(train_X[:, selected_features], test_X[:, selected_features], train_y, test_y)
                feature_accuracies.append((feature, accuracy))
                selected_features.remove(feature)
        
        best_feature, new_accuracy = max(feature_accuracies, key=lambda x: x[1])
        selected_features.append(best_feature)
        
        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
            best_features = selected_features.copy()
        
        print(f"Selected Feature: {best_feature}, Current Accuracy: {new_accuracy:.2%}, Current Features: {selected_features}")

    return best_features, best_accuracy

def backward_elimination(train_X, test_X, train_y, test_y):
    num_features = train_X.shape[1]
    selected_features = list(range(num_features))
    
    best_accuracy = evaluate(train_X[:, selected_features], test_X[:, selected_features], train_y, test_y)
    best_features = selected_features.copy()
    
    while len(selected_features) > 0:
        feature_accuracies = []

        for feature in selected_features:
            current_features = selected_features.copy()
            current_features.remove(feature)
            accuracy = evaluate(train_X[:, current_features], test_X[:, current_features], train_y, test_y)
            feature_accuracies.append((feature, accuracy))
        
        worst_feature, new_accuracy = max(feature_accuracies, key=lambda x: x[1])
        
        if new_accuracy >= best_accuracy:
            selected_features.remove(worst_feature)
            best_accuracy = new_accuracy
            best_features = selected_features.copy()
            print(f"Removed Feature: {worst_feature}, Current Accuracy: {new_accuracy:.2%}, Current Features: {selected_features}")
        else:
            break

    return best_features, best_accuracy


def evaluate(train_X, test_X, train_y, test_y):
    correct_predictions = 0

    for test_X_, test_y_ in zip(test_X, test_y):
        predicted_label = train_y[nearest_neighbor(train_X, test_X_)]
        if predicted_label == test_y_:
            correct_predictions += 1

    return correct_predictions / len(test_X)

# Function to load the data from a file
def load_data():
    df = pd.read_csv('cancer.csv')
    print(df.info())
    print(df.shape)
    return df

# User Interface
data = load_data()
data_arr = np.array(data)
y = data_arr[:, 1]

def normalize_features(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

X = data_arr[:, 2:].astype(np.float64)
X = normalize_features(X)
train_X = X[:int(len(X)*0.8), :]
train_y = y[:int(len(y)*0.8)]
test_X = X[int(len(X)*0.8):, :]
test_y = y[int(len(y)*0.8):]

search_method = input("Select the search method (1: Forward Selection, 2: Backward Elimination): ")

if search_method == '1':
    # Forward Selection feature search
    selected_features, best_accuracy = forward_selection(train_X, test_X, train_y, test_y)
    print("Final Selected Features:", selected_features)
    print(f"Best Accuracy: {best_accuracy:.2%}")
elif search_method == '2':
    # Backward Elimination feature search
    selected_features, best_accuracy = backward_elimination(train_X, test_X, train_y, test_y)
    print("Final Selected Features:", selected_features)
    for feature in selected_features:
        print(data.iloc[:, feature+2].name)
    print(f"Best Accuracy: {best_accuracy:.2%}")
else:
    print("Invalid search method selected. Please choose 1 or 2.")
