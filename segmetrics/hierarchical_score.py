import numpy as np
from segmetrics.overlap import label_overlap
from segmetrics.matching import _match_1_to_1
from segmetrics.intersection import intersection_over_union, intersection_over_true, intersection_over_pred

matching_criteria = {
    'iou': intersection_over_union,
    'iot': intersection_over_true,
    'iop': intersection_over_pred,
}

def optimal_cut(y_true, y_pred, forest_pred, threshold=0.5, criterion='iou'):
    """
    Find the optimal cut in the hierarchy represented by `forest_pred` to maximize segmentation scores.

    Parameters
    ----------
    y_true : ndarray
        Ground truth label image.
    y_pred : ndarray
        Initial predicted label image.
    forest_pred : dict
        Dictionary representing the hierarchical forest. Each key is a tuple (child1, child2)
        and the value is the parent node.
    threshold : float
        Threshold for score calculation.
    criterion : str
        Criterion for scoring ('iou', 'iot', 'iop').
    method : str
        Matching method ('1:1', '1:N', 'N:1').

    Returns
    -------
    dict
        A dictionary with optimal segmentation results.
    """
    # Calculate initial overlap matrix
    overlap = label_overlap(y_true, y_pred)
    current_pred = np.copy(y_pred)
    best_scores = {}

    # Initialize label mapping as a list
    label_mapping = list(np.unique(y_pred))

    def calculate_f1_from_overlap(overlap):
        # Exclude background by ignoring the first row and column
        
        scores = matching_criteria[criterion](overlap)
        # Perform 1:1 matching using the pre-defined utility function
        scores = scores[1:, 1:]
        matches = _match_1_to_1(scores, threshold)

        tp = len(matches)  # True Positives: the matched pairs
        n_true_labels = len(np.unique(y_true)) - 1  # Exclude background (0 label)
        n_pred_labels = len(np.unique(y_pred)) - 1  # Exclude background (0 label)
        fp = n_pred_labels - tp  # False Positives
        fn = n_true_labels - tp  # False Negatives

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1_score

    # Initialize leaves as the unique labels in y_pred (excluding background)
    leaves = set(np.unique(y_pred)) - {0}

    for (child1, child2), parent in forest_pred.items():
        # Only consider merging if both children are leaves
        if child1 in leaves and child2 in leaves:
            # Create merged prediction
            merged_pred = np.copy(current_pred)
            merged_pred[(current_pred == child1) | (current_pred == child2)] = parent

            # Get indices for child1 and child2 in the overlap matrix columns
            idx1 = label_mapping.index(child1)
            idx2 = label_mapping.index(child2)
            
            # Calculate the new column for the merged parent node by summing columns for child1 and child2
            new_col = np.sum(overlap[:, [idx1, idx2]], axis=1)
            
            # Create the new overlap matrix by removing child1 and child2 columns and inserting the merged column
            new_overlap = np.delete(overlap, [idx1, idx2], axis=1)  # Remove columns for child1, child2
            new_overlap = np.insert(new_overlap, len(new_overlap[0]), new_col, axis=1)

            # Calculate scores for merged and unmerged states
            score_merged = calculate_f1_from_overlap(new_overlap)
            score_unmerged = calculate_f1_from_overlap(overlap)
            
            # Only update if merging improves the score
            if score_merged >= score_unmerged:
                current_pred = merged_pred
                overlap = new_overlap
                best_scores[parent] = score_merged
                leaves.discard(child1)
                leaves.discard(child2)
                leaves.add(parent)
                
                # Update the label mapping by removing child1 and child2, then adding the parent at the end
                label_mapping = [label for i, label in enumerate(label_mapping) if i not in [idx1, idx2]]
                label_mapping.append(parent)

            else:
                best_scores[parent] = score_unmerged

    # Return the final prediction and associated score
    final_score = calculate_f1_from_overlap(overlap)
    best_scores['final_score'] = final_score
    best_scores['final_prediction'] = current_pred

    return best_scores


if __name__ == "__main__":
    y_true = np.array([
        [1, 1, 0],
        [2, 2, 0],
        [2, 2, 0]
    ])

    y_pred = np.array([
        [1, 2, 0],
        [0, 3, 4],
        [0, 6, 5]
    ])

    # Forest structure with possible merges; for example, merging 1 and 2 into 7, etc.
    forest_pred = {
        (1, 2): 7,
        (3, 6): 8,
        (8, 4): 9,
    }

    # Run optimal cut to find best segmentation configuration
    optimal_scores = optimal_cut(y_true, y_pred, forest_pred, threshold=0.5, criterion='iou')
    print(optimal_scores)
