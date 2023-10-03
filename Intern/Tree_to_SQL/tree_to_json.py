import json
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# Step 1: Generate a custom dataset
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Step 2: Train a DecisionTreeClassifier with custom parameters (max_depth)
tree_classifier = DecisionTreeClassifier(max_depth=4)  # You can adjust max_depth as needed
tree_classifier.fit(X, y)


# Step 3: Write the function to convert the tree to JSON
def convert_tree_to_json(tree):
    def recurse(node):
        if tree.tree_.children_left[node] == tree.tree_.children_right[node]:  # Leaf node
            return {
                "class": int(tree.tree_.value[node].argmax())
            }
        else:  # Non-leaf node
            return {
                "feature_index": int(tree.tree_.feature[node]),
                "threshold": round(float(tree.tree_.threshold[node]), 4),
                "left": recurse(tree.tree_.children_left[node]),
                "right": recurse(tree.tree_.children_right[node])
            }

    tree_json = recurse(0)
    tree_as_json = json.dumps(tree_json, indent=4)

    return tree_as_json


def generate_sql_query(tree_as_json, features):
    def recursive_sql(node, feature_names):
        if "class" in node:
            return str(node["class"])  # Return the class label when it's a leaf node
        else:
            feature_name = feature_names[node["feature_index"]]
            threshold = node["threshold"]
            left_sql = recursive_sql(node["left"], feature_names)
            right_sql = recursive_sql(node["right"], feature_names)
            return f"CASE WHEN {feature_name} > {threshold} THEN {right_sql} ELSE {left_sql} END"

    tree_dict = json.loads(tree_as_json)
    sql_query = 'SELECT ' + recursive_sql(tree_dict, features) + ' AS CLASS_LABEL'
    return sql_query