import numpy as np
from sklearn.metrics import average_precision_score


def simulate_map(test_set_sizes, n_classes=527, n_iterations=10):
    """
    Simulate mAP calculation for given test set sizes over multiple iterations.

    Parameters:
        test_set_sizes (list): List of test set sizes to simulate.
        n_classes (int): Number of classes (default 527).
        n_iterations (int): Number of iterations for the simulation (default 10).

    Returns:
        dict: Average and standard deviation of mAP for each test set size.
    """
    results = {size: [] for size in test_set_sizes}

    for _ in range(n_iterations):
        # Generate random labels and predictions
        # true_labels = np.random.randint(0, 2, size=(size, n_classes))
        # pred_scores = np.random.rand(size, n_classes)

        # Simulate true labels with imbalanced class distribution
        size = 20123
        label_distribution = np.random.zipf(
            a=1.3, size=n_classes
        )  # Zipf for power-law distribution
        label_distribution = (
            label_distribution / label_distribution.sum()
        )  # Normalize to probabilities
        true_labels = np.random.binomial(1, label_distribution, size=(size, n_classes))

        # Simulate predicted scores with a skew towards correct predictions
        pred_scores = np.zeros((size, n_classes))
        for c in range(n_classes):
            # For positive samples, draw scores from Beta(2, 1) (skewed towards 1)
            # For negative samples, draw scores from Beta(1, 2) (skewed towards 0)
            pred_scores[:, c] = np.where(
                true_labels[:, c] == 1,
                np.random.beta(2, 1, size),
                np.random.beta(1, 2, size),
            )
        for tsize in test_set_sizes:
            # Compute average precision for each class
            ap_per_class = [
                average_precision_score(true_labels[:tsize, c], pred_scores[:tsize, c])
                for c in range(n_classes)
                if np.sum(true_labels[:tsize, c]) > 0  # Avoid classes with no positives
            ]

            # Compute mean average precision
            map_value = np.mean(ap_per_class)
            map_value *= 100
            results[tsize].append(map_value)
            print(f"Test set size: {tsize}, mAP: {map_value}")

    summary = {
        size: {"average_map": np.mean(values), "std_dev_map": np.std(values)}
        for size, values in results.items()
    }

    return summary


# Define parameters
test_set_sizes = [18987, 20123]
n_iterations = 5

# Run simulation
simulation_results = simulate_map(test_set_sizes, n_iterations=n_iterations)
print(simulation_results)
