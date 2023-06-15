import numpy as np
import time

def euclidean_distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).sum(axis=1))

def nearest_neighbor(train_data, test_instance):
    distance = euclidean_distance(train_data, test_instance)
    idx = distance.argmin()
    return idx

def forward_selection(train_data, test_data):
    num_features = train_data.shape[1] - 1
    selected_features = []
    best_accuracy = 0.0
    best_features = None
    train_label = train_data[:, 0]
    train_data = train_data[:, 1:]
    test_label = test_data[:, 0]
    test_data = test_data[:, 1:]
    
    for _ in range(num_features):
        feature_accuracies = []

        for feature in range(num_features):
            if feature not in selected_features:
                selected_features.append(feature)
                accuracy = evaluate(train_data[:, selected_features], test_data[:, selected_features], train_label, test_label)
                feature_accuracies.append((feature, accuracy))
                selected_features.remove(feature)
        
        best_feature, new_accuracy = max(feature_accuracies, key=lambda x: x[1])
        selected_features.append(best_feature)
        
        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
            best_features = selected_features.copy()
        
        print(f"Selected Feature: {best_feature}, Current Accuracy: {new_accuracy:.2%}, Current Features: {selected_features}")

    return best_features, best_accuracy

def backward_elimination(train_data, test_data):
    num_features = train_data.shape[1] - 1
    selected_features = list(range(num_features))
    train_label = train_data[:, 0]
    train_data = train_data[:, 1:]
    test_label = test_data[:, 0]
    test_data = test_data[:, 1:]
    best_accuracy = evaluate(train_data[:, selected_features], test_data[:, selected_features], train_label, test_label)
    best_features = selected_features.copy()
    
    while len(selected_features) > 0:
        feature_accuracies = []

        for feature in selected_features:
            current_features = selected_features.copy()
            current_features.remove(feature)
            accuracy = evaluate(train_data[:, current_features], test_data[:, current_features], train_label, test_label)
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


def evaluate(train_data, test_data, train_label, test_label):
    correct_predictions = 0

    for test_data_, test_label_ in zip(test_data, test_label):
        predicted_label = train_label[nearest_neighbor(train_data, test_data_)]
        if predicted_label == test_label_:
            correct_predictions += 1

    return correct_predictions / len(test_data)

# Function to load the data from a file
def load_data(filename):
    data = np.loadtxt(filename)
    return data

# User Interface
filename = input("Enter the file name: ")
data = load_data(filename)

train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

search_method = input("Select the search method (1: Forward Selection, 2: Backward Elimination): ")

if search_method == '1':
    # Forward Selection feature search
    start_time = time.time()
    selected_features, best_accuracy = forward_selection(train_data, test_data)
    end_time = time.time()
    time_cost = end_time - start_time
    print("Final Selected Features:", selected_features)
    print(f"Best Accuracy: {best_accuracy:.2%}")
    print(f"Time Cost: {time_cost:.4f} s")
elif search_method == '2':
    # Backward Elimination feature search
    start_time = time.time()
    selected_features, best_accuracy = backward_elimination(train_data, test_data)
    end_time = time.time()
    time_cost = end_time - start_time
    print("Final Selected Features:", selected_features)
    print(f"Best Accuracy: {best_accuracy:.2%}")
    print(f"Time Cost: {time_cost:.4f} s")
else:
    print("Invalid search method selected. Please choose 1 or 2.")
