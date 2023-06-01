import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer, roc_auc_score, log_loss, accuracy_score
import numpy as np
import time
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
import itertools
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
import math
from itertools import combinations
import sys
import pickle
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold

class DataPreprocessor:
    def __init__(self, n_features_to_select=None, use_auto_select_features=False,
                 use_local_search=False, prioritized_features=None,
                 missing_data_method='mean_imputation', outlier_handling_method='z_score_outlier_handling',
                 exclude_method='zero', random_state=None):  # Add random_state parameter
        self.n_features_to_select = n_features_to_select
        self.use_auto_select_features = use_auto_select_features
        self.use_local_search = use_local_search
        self.prioritized_features = prioritized_features
        self.missing_data_method = missing_data_method
        self.outlier_handling_method = outlier_handling_method
        self.exclude_method = exclude_method
        self.random_state = random_state  # Set random_state as instance variable
        self.selected_features_ = None  # Initialize the attribute 

    def fit(self, X, y):
        if self.use_auto_select_features:
            # Define the feature selection model
            model = LogisticRegression(random_state=self.random_state)  # Use the random_state attribute
            # Set n_features_to_select to 'auto' if self.n_features_to_select is None
            n_features_to_select = 'auto' if self.n_features_to_select is None else self.n_features_to_select
            # Fit the SequentialFeatureSelector
            self.feature_selector_ = SequentialFeatureSelector(
                model, n_features_to_select=n_features_to_select, direction='forward', scoring='accuracy', tol=None
            )
            self.feature_selector_.fit(X, y)

    def auto_select_features(self, X, y=None):
        # If use_auto_select_features is True, transform the data using the stored feature selector
        if self.use_auto_select_features:
            X = self.feature_selector_.transform(X)
            self.selected_features_ = self.feature_selector_.get_support(indices=True)
        
        # If prioritized_features are provided, validate indices and select only those features
        if self.prioritized_features is not None:
            # Get the total number of features in the transformed matrix
            total_features = X.shape[1]
            
            # Validate indices and filter out invalid ones
            valid_indices = [idx for idx in self.prioritized_features if idx < total_features]
            
            # Select features using valid indices
            X = X[:, valid_indices]
            self.selected_features_ = np.array(valid_indices)
            
            # Add this line to print the selected_features_ attribute after it is updated
            print(f"Updated selected_features_: {self.selected_features_}")
            
            # Store the selected feature indices
            self.selected_features_ = valid_indices
            
        # Local search feature selection logic (only run if 'y' is provided during training)
        if self.use_local_search and y is not None:  # Check if 'y' is provided
            model = LogisticRegression()
            
            # Set n_features_to_select to 'auto' if self.n_features_to_select is None
            n_features_to_select = 'auto' if self.n_features_to_select is None else self.n_features_to_select
            
            # Pass n_features_to_select to SequentialFeatureSelector
            sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select,
                                            direction='forward', scoring='accuracy', tol=None)
            
            sfs.fit(X, y)  # Use the 'y' parameter
            X = sfs.transform(X)  # Update the X variable with the selected features
            self.selected_features_ = sfs.get_support(indices=True)  # Store the selected feature indices

        # Correlation-based feature selection logic (skip during prediction)
        elif self.n_features_to_select is not None and not self.use_local_search and y is not None:
            correlations = np.abs(np.corrcoef(X, y, rowvar=False)[-1, :-1])
            top_feature_indices = np.argsort(correlations)[-self.n_features_to_select:]
            X = X[:, top_feature_indices]
            
        return X
    
    def mean_imputation(self, X):
        col_means = np.nanmean(X, axis=0)
        return np.where(np.isnan(X), col_means, X)

    def deletion(self, X, y, missing_values=np.nan):
        not_missing = ~np.isnan(X).any(axis=1)
        return X[not_missing], y[not_missing]

    def z_score_outlier_handling(self, X, y, threshold=3):
        z_scores = np.abs(stats.zscore(X))
        not_outliers = (z_scores < threshold).all(axis=1)
        return X[not_outliers], y[not_outliers]

    def iqr_outlier_handling(self, X, y, k=1.5):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        not_outliers = np.logical_and(X >= lower_bound, X <= upper_bound).all(axis=1)
        return X[not_outliers], y[not_outliers]

    def handle_missing_data(self, X, y):
        if self.missing_data_method == "mean_imputation":
            return self.mean_imputation(X), y  # y is unchanged
        elif self.missing_data_method == "deletion":
            return self.deletion(X, y)
        else:
            raise ValueError("Unsupported missing_data_method. Please use 'mean_imputation' or 'deletion'.")

    def handle_outliers(self, X, y):
        if self.outlier_handling_method == "z_score_outlier_handling":
            return self.z_score_outlier_handling(X, y)
        elif self.outlier_handling_method == "iqr_outlier_handling":
            return self.iqr_outlier_handling(X, y)
        else:
            raise ValueError("Unsupported outlier_handling_method. Please use 'z_score_outlier_handling' or 'iqr_outlier_handling'.")

class SPINEX(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, distance_threshold=0.05, distance_threshold_decay=0.95, ensemble_method=None, preprocessor=None, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold
        self.distance_threshold_decay = distance_threshold_decay
        self.ensemble_method = ensemble_method
        self.preprocessor = preprocessor
        self.metric = metric
        self.feature_combination_size = None
        self.X_train_ = None
        self.y_train_ = None
        self.eps = 1e-8
    
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y

        one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.y_train_one_hot_ = one_hot_encoder.fit_transform(y.reshape(-1, 1))

        self.class_prior_ = np.bincount(y) / len(y)

        if self.ensemble_method == 'bagging':
            self.model_ = BaggingClassifier(estimator=SPINEX(ensemble_method=None),
                                            n_estimators=10, random_state=42)
        elif self.ensemble_method == 'boosting':
            self.model_ = AdaBoostClassifier(estimator=SPINEX(ensemble_method=None),
                                             n_estimators=10, random_state=42)
        elif self.ensemble_method == 'stacking':
            estimators = [('spinex1', SPINEX(ensemble_method=None)), 
                          ('spinex2', SPINEX(ensemble_method=None))]
            self.model_ = StackingClassifier(estimators=estimators, 
                                             final_estimator=LogisticRegression())
        else:
            self.model_ = None

        if self.model_ is not None:
            self.model_.fit(X, y, sample_weight)

        return self

    def _calculate_feature_combination_distances(self, instances, train_instances):
        # Calculate the distances between instances and train_instances based on feature combinations
        feature_combinations = list(combinations(range(min(train_instances.shape[1], instances.shape[1])), self.feature_combination_size))
        distances = np.zeros((instances.shape[0], train_instances.shape[0]))
        for comb in feature_combinations:
            comb_distance = np.sqrt(np.sum((train_instances[:, comb] - instances[:, comb][:, np.newaxis]) ** 2, axis=-1))
            distances += comb_distance
        overall_distance = distances / len(feature_combinations)
        return overall_distance

    def calculate_weights(self, distances):
        # Compute the weights based on the distances using the Gaussian kernel function
        sigma = np.mean(distances)
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
        return weights

    def predict_proba(self, X):
        if self.preprocessor:
            X = self.preprocessor.auto_select_features(X)

        if self.model_ is not None:
            probabilities = self.model_.predict_proba(X)
        else:
            distances = pairwise_distances(X, self.X_train_, metric=self.metric)
            nearest_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
            decayed_distance_threshold = self.distance_threshold * self.distance_threshold_decay
            nearest_distances = distances[np.arange(distances.shape[0])[:, None], nearest_indices]
            weights = 1 / (nearest_distances + decayed_distance_threshold)

            weights = weights[:, :, None]  # Add an extra dimension to match with y_train_one_hot_
            weighted_votes = self.y_train_one_hot_[nearest_indices] * weights
            weighted_votes = np.sum(weighted_votes, axis=1)  # Sum across the n_neighbors axis

            # Normalize the weighted votes to obtain probabilities
            probabilities = weighted_votes / np.sum(weighted_votes, axis=1, keepdims=True)

        return probabilities

    def predict(self, X):
        if self.preprocessor:
            X = self.preprocessor.auto_select_features(X)

        distances = pairwise_distances(X, self.X_train_, metric=self.metric)
        nearest_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        decayed_distance_threshold = self.distance_threshold * self.distance_threshold_decay
        nearest_distances = distances[np.arange(distances.shape[0])[:, None], nearest_indices]
        weights = 1 / (nearest_distances + decayed_distance_threshold)

        weights = weights[:, :, None]
        weighted_votes = self.y_train_one_hot_[nearest_indices] * weights
        weighted_votes = np.sum(weighted_votes, axis=1)

        weighted_votes = np.atleast_2d(weighted_votes)

        if self.model_ is not None:
            predictions = self.model_.predict(X)
        else:
            predictions = self.classes_[np.argmax(weighted_votes, axis=1)]

        return predictions

    def predict_contributions(self, X, instances_to_predict=None):
        if instances_to_predict is None:
            instances_to_predict = range(X.shape[0])

        # Use only selected features
        selected_features = self.X_train_.shape[1]
        X = X[:, :selected_features]

        # Calculate overall predictions (probability) for selected instances
        final_probabilities = self.predict_proba(X[instances_to_predict])

        # Define a function to calculate contributions for each feature
        def compute_contributions(i):
            # Prediction with the feature excluded (set to zero or mean value)
            X_excluded = X.copy()
            X_excluded[:, i] = 0  # You may replace this with mean imputation
            excluded_probabilities = self.predict_proba(X_excluded[instances_to_predict])

            # Contribution of the feature
            feature_contributions = final_probabilities - excluded_probabilities
            return feature_contributions

        # Calculate contributions for each feature in parallel
        with ThreadPoolExecutor() as executor:
            contributions = list(executor.map(compute_contributions, range(X.shape[1])))

        # Calculate pairwise interaction effects
        interaction_effects = []
        for i in range(X.shape[1]):
            interaction_effects_row = []
            for j in range(X.shape[1]):
                if i == j:
                    interaction_effects_row.append(np.zeros_like(final_probabilities))
                    continue
                # Interaction effect of features i and j
                interaction_effect = final_probabilities - contributions[i] - contributions[j]
                interaction_effects_row.append(interaction_effect)
            interaction_effects.append(interaction_effects_row)

        # Combine interaction effects into an array
        interaction_effects = np.array(interaction_effects, dtype=object)

        return final_probabilities, np.array(contributions), interaction_effects

    def get_feature_importance(self, X, instances_to_explain=None):
        """Get feature importance and interaction effects for the given instances."""
        if instances_to_explain is None:
            instances_to_explain = range(X.shape[0])
        
        # Use only selected features
        selected_features = self.X_train_.shape[1]
        X = X[:, :selected_features]

        # Contribution of each feature to the model's predictions for each instance
        predictions, contributions, interaction_effects = self.predict_contributions(X)
        # Calculate the contribution of each feature
        feature_importances = np.mean(np.abs(contributions), axis=(1, 2))

        # Calculate the interaction effects
        interaction_effects = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                interaction_effect = np.mean(np.abs(predictions - contributions[:, i] - contributions[:, j]), axis=0)
                interaction_effects[i, j] = interaction_effect
                interaction_effects[j, i] = interaction_effect

        return feature_importances, interaction_effects

    def get_global_interaction_effects(self, X, instances_to_explain=None):
        """Get the average interaction effects for the given instances."""
        if instances_to_explain is None:
            instances_to_explain = range(X.shape[0])
        
        # Use only selected features
        selected_features = self.X_train_.shape[1]
        X = X[:, :selected_features]

        # Contribution of each feature to the model's predictions for each instance
        predictions, contributions, _ = self.predict_contributions(X)

        # Calculate the interaction effects
        avg_interaction_effects = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(i+1, X.shape[1]):
                # Interaction effect of features i and j for each class
                interaction_effect_per_class = np.abs(predictions - contributions[i] - contributions[j])
                # Average interaction effect across classes and samples
                interaction_effect = np.mean(interaction_effect_per_class)
                avg_interaction_effects[i, j] = interaction_effect
                avg_interaction_effects[j, i] = interaction_effect

        return avg_interaction_effects

    def feature_combination_impact_analysis(self, X):
        # Use only selected or prioritized features
        selected_features = self.X_train_.shape[1]
        X = X[:, :selected_features]

        # Calculate the base predictions (probability) for the entire dataset
        base_probabilities = self.predict_proba(X)

        # Initialize an empty dictionary to store the impact of each feature combination
        feature_combination_impact = {}

        # Define a function to calculate the impact of a feature combination
        def compute_combination_impact(comb):
            # Exclude features in the current combination and calculate new predictions (probability)
            X_excluded = X.copy()
            X_excluded[:, comb] = 0  # You may replace this with mean imputation
            excluded_probabilities = self.predict_proba(X_excluded)

            # Calculate the impact of the current feature combination
            impact = np.mean(np.abs(base_probabilities - excluded_probabilities))
            return comb, impact

        # Iterate over possible feature combinations in parallel
        n_features = X.shape[1]
        with ThreadPoolExecutor() as executor:
            for combination_size in range(1, n_features + 1):
                all_combinations = itertools.combinations(range(n_features), combination_size)
                impacts = executor.map(compute_combination_impact, all_combinations)
                # Store the impact of each feature combination
                feature_combination_impact.update(impacts)

        # Sort feature combinations by their impact
        sorted_combinations = sorted(feature_combination_impact.items(), key=lambda x: x[1], reverse=True)

        return sorted_combinations

    def get_params(self, deep=True):
        return {
            'n_neighbors': self.n_neighbors,
            'distance_threshold': self.distance_threshold,
            'distance_threshold_decay': self.distance_threshold_decay,
            'ensemble_method': self.ensemble_method,
            'preprocessor': self.preprocessor,  # Data preprocessor instance
            'metric': self.metric
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Create a synthetic classification dataset
#X, y = make_classification(n_samples=500, n_features=6, n_classes=2, random_state=42)

import io
    
# Set up the Streamlit interface
st.title('SPINEX: Similarity-based Predictions with Explainable Neighbors Exploration [SPINEXClassifier - Demo]')

# Add your name, website, and email using Markdown
st.markdown("## About Us")
st.markdown("- **Creators:** M.Z. Naser, M.K. al-Bashiti, A.Z. Naser")
st.markdown("- **Website:** [www.mznaser.com](http://www.mznaser.com)")
#st.markdown("- **Email:** mznaser@clemson.edu")

# Create a sidebar with a title and text
st.sidebar.title("Read Me")
st.sidebar.markdown("Please feel free to try our still-in-development demo and read out preprint () to learn more about SPINEX, its derivatives and fuctions. You can cite our work as, Naser M.Z., Al-Bashiti M.K., Naser A.Z. (2023). SPINEX: Similarity-based Predictions with Explainable Neighbors Exploration for Regression and Classification Tasks in Machine Learning. ArXiv. Please be patient with us while we improve SPINEX. Please note that run time(s) can vary at the moment.")

# File upload
uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    
    if file_extension == "csv":
        data = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    elif file_extension == "xlsx":
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or XLSX file.")
        st.stop()

    # Extract features and ground truth
    X = data.iloc[:, :-1]  # Assuming the ground truth is the far-right column
    y = data.iloc[:, -1]

    # Define the desired transformation method (you can choose one of: 'MinMax', 'Robust', 'Log', 'Power', or 'None')
    transformation_method = 'MinMax'

    # Apply the selected transformation method
    if transformation_method == 'Standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif transformation_method == 'MinMax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif transformation_method == 'Robust':
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    elif transformation_method == 'Log':
        X = np.log1p(X)  # Apply the natural logarithm transformation
    elif transformation_method == 'Power':
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)  # You can choose 'box-cox' instead if desired
        X = transformer.fit_transform(X)
    elif transformation_method == 'None':
        pass  # No transformation is applied
    else:
        print("Invalid transformation method selected.")

    
    # Initialize the SPINEX model with desired parameters
    # (without the 'preprocessor' argument)
    SPINEXClassifier = SPINEX(
        n_neighbors=10,
        distance_threshold=0.05,
        distance_threshold_decay=0.95,
        ensemble_method=None,
        metric='manhattan'
    )

    # Instantiate the DataPreprocessor with use_auto_select_features=True, use_local_search=True, and prioritized features
    preprocessor = DataPreprocessor(          
        use_auto_select_features=False,   # auto and local are independent. 
        use_local_search=False,           #Automatic feature selection without any prioritization TFNone
        prioritized_features= None #[1,2]         #Local search-based feature selection without any prioritization FTNone
    )                                     #Prioritized feature selection without automatic or local search-based feature selection FF[0,2]
                                        #Automatic feature selection with prioritized features TF[02]

    # Fit the DataPreprocessor on the data (this step is necessary before calling auto_select_features)
    preprocessor.fit(X, y)

    # Retrieve the indices of the selected features
    selected_features_indices = preprocessor.selected_features_

    # Perform preprocessing and feature selection on the data
    X_preprocessed = preprocessor.auto_select_features(X, y)
    X_preprocessed, y_preprocessed = preprocessor.handle_missing_data(X_preprocessed, y)
    X_preprocessed, y_preprocessed = preprocessor.handle_outliers(X_preprocessed, y_preprocessed)

    # Print the shape of the original and preprocessed data
    print("Original data shape:", X.shape)
    print("Preprocessed data shape:", X_preprocessed.shape)

    # Find the selected feature indices
    selected_features = []
    for i in range(X.shape[1]):
        unique_X = np.unique(X[:, i])
        unique_X_preprocessed = np.unique(X_preprocessed[:, :])

        if len(unique_X) == len(unique_X_preprocessed) and np.allclose(unique_X, unique_X_preprocessed, rtol=1e-05, atol=1e-08):
            selected_features.append(i)

    # Print the selected features
    print("Selected features:", preprocessor.selected_features_)

    # Fit the SPINEX model on the preprocessed training data
    SPINEXClassifier.fit(X_preprocessed, y_preprocessed)

    # StackingSPINEX
    stacking_model = StackingClassifier(
        estimators=[('spinex', SPINEXClassifier), ('DT', DecisionTreeClassifier())],
        final_estimator=LogisticRegression(),
        cv=5
    )

    # BaggingSPINEX
    bagging_model = BaggingClassifier(
        estimator=SPINEXClassifier,
        n_estimators=10,
        random_state=42
    )

    # BoostingSPINEX
    boosting_model = AdaBoostClassifier(
        estimator=SPINEXClassifier,
        n_estimators=10,
        random_state=42
    )

    # Fit the ensemble models on the preprocessed training data
    stacking_model.fit(X_preprocessed, y_preprocessed)
    bagging_model.fit(X_preprocessed, y_preprocessed)
    boosting_model.fit(X_preprocessed, y_preprocessed)


    ###################################################
    # Instantiate the default DataPreprocessor
    default_preprocessor = DataPreprocessor()

    # Fit the default DataPreprocessor on the data
    default_preprocessor.fit(X, y)

    # Perform preprocessing and feature selection on the data using the default DataPreprocessor
    X_preprocessed_default = default_preprocessor.auto_select_features(X, y)
    X_preprocessed_default, y_preprocessed_default = default_preprocessor.handle_missing_data(X_preprocessed_default, y)
    X_preprocessed_default, y_preprocessed_default = default_preprocessor.handle_outliers(X_preprocessed_default, y_preprocessed_default)

    # Define the SPINEXClassifier(default) model
    SPINEXClassifierDefault = SPINEX()

    # Fit the SPINEXClassifier(default) model on the preprocessed training data using the default DataPreprocessor
    SPINEXClassifierDefault.fit(X_preprocessed_default, y_preprocessed_default)
    ###################################################

    # Print the selected features for the SPINEX model
    selected_features_custom = preprocessor.selected_features_
    print("Selected features for SPINEX:", selected_features_custom)

    # Print the selected features for the SPINEXClassifier(default) model
    selected_features_default = default_preprocessor.selected_features_
    print("Selected features for SPINEXClassifier(default):", selected_features_default)

    # Define the models to compare for classification
    models = {
        'SPINEXClassifier(default)':SPINEXClassifierDefault, #SPINEX(),
        'SPINEX': SPINEXClassifier,
        'StackingSPINEX': stacking_model,
        'BaggingSPINEX': bagging_model,
        'BoostingSPINEX': boosting_model,
        'LogisticRegression': LogisticRegression(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'CatBoostClassifier': CatBoostClassifier(verbose=False),
        'XGBClassifier': XGBClassifier(eval_metric='logloss', n_jobs=-1),
        'LGBMClassifier': LGBMClassifier(metric='binary_logloss', n_jobs=-1),
        'SVC': SVC(probability=True),
        'KNeighborsClassifier': KNeighborsClassifier()
    }

    # Define custom scoring functions
    def predictivity(y_true, y_pred, y_proba, accuracy, roc_auc):
        log_loss_error = log_loss(y_true, y_proba)
        return (accuracy * roc_auc) / log_loss_error

    def predictivity_scorer(y_true, y_pred_proba):
        if y_pred_proba.ndim == 2:
            y_pred = y_pred_proba.argmax(axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba)
        log_loss_error = log_loss(y_true, y_pred_proba)
        
        return (accuracy * auc) / log_loss_error

    def estimated_energy(model, total_time):
        model_size = sys.getsizeof(pickle.dumps(model))
        return model_size * total_time

    # Define the scoring metrics
    scoring_metrics = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'neg_log_loss': 'neg_log_loss',
        'roc_auc': 'roc_auc',
        'predictivity': make_scorer(predictivity_scorer, needs_proba=True)
    }

    # Initialize results dictionary
    results = {}

    # Perform cross-validation for each model
    for model_name, model in models.items():
        # Print the model_name
        print(f"Model: {model_name}")
        
        # Record the start time
        start_time = time.time()
        
        # Perform cross-validation and store the scores
        # Use the preprocessed data for cross-validation
        # scores = cross_validate(model, X_preprocessed, y_preprocessed, cv=10, scoring=scoring_metrics, return_train_score=False)
        kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)
        scores = cross_validate(model, X_preprocessed, y_preprocessed, cv=kfold, scoring=scoring_metrics, return_train_score=False)

        
        # Record the end time
        end_time = time.time()
        
        # Print the cross-validation scores
        print("Cross-validation scores:")
        for metric, values in scores.items():
            print(f"  {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
        
        # Calculate the model size
        model_size = sys.getsizeof(pickle.dumps(model))
        
        # Print the model size
        print(f"Model size (bytes): {model_size}")
        
        # Add a separator line between models
        print("-" * 50)
        
        # Calculate the average metrics
        mean_accuracy = np.mean(scores['test_accuracy'])
        mean_f1_macro = np.mean(scores['test_f1_macro'])
        mean_neg_log_loss = np.mean(scores['test_neg_log_loss'])
        mean_roc_auc = np.mean(scores['test_roc_auc'])
        mean_predictivity = np.mean(scores['test_predictivity'])
        
        # Calculate the average fitting and scoring times
        mean_fit_time = np.mean(scores['fit_time'])
        mean_score_time = np.mean(scores['score_time'])
        total_time = end_time - start_time
        
        # Calculate the estimated energy
        energy = estimated_energy(model, total_time)

        # Store the results
        results[model_name] = {
            'Accuracy': mean_accuracy,
            'F1 Macro': mean_f1_macro,
            'Negative Log Loss': mean_neg_log_loss,
            'ROC AUC': mean_roc_auc,
            'Predictivity': mean_predictivity,
            'Estimated Energy': energy,
            'Mean Fit Time': mean_fit_time,
            'Mean Score Time': mean_score_time,
            'Total Time': total_time,
            'Model Size (bytes)': model_size
        }

    # Display the results
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print("-" * 50)

    import matplotlib.pyplot as plt

    # Define the metrics to plot
    #metrics_to_plot = [
    #    'Accuracy', 'F1 Macro', 'Negative Log Loss', 'ROC AUC',  # Added 'Negative Log Loss' and 'ROC AUC'
    #    'Mean Fit Time', 'Mean Score Time', 'Total Time', 'Predictivity', 'Estimated Energy', 'Model Size (bytes)'
    #]

    metrics_to_plot = [
        'Accuracy', 'Negative Log Loss', 'ROC AUC',  # Added 'Negative Log Loss' and 'ROC AUC'
        'Mean Fit Time', 'Mean Score Time', 'Total Time'
    ]

    # Create a figure and subplots for each metric
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, len(metrics_to_plot) * 5))

    # Iterate through each metric and create a bar plot
    for i, metric in enumerate(metrics_to_plot):
        # Extract the values for the current metric
        metric_values = [results[model_name][metric] for model_name in models.keys()]
        
        # Create the bar plot for the current metric
        axs[i].bar(models.keys(), metric_values)
        
        # Set the title and labels for the current subplot
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        
        # Rotate the x-axis labels by 90 degrees
        axs[i].tick_params(axis='x', rotation=90)

    # Adjust the layout to prevent overlapping labels
    #plt.tight_layout()

    # Show the plots
    #plt.show()
    #st.pyplot(fig)

    import matplotlib.pyplot as plt

    # Define the metrics to plot
    #metrics_to_plot = [
    #    'Accuracy', 'F1 Macro', 'Negative Log Loss', 'ROC AUC',  # Added 'Negative Log Loss' and 'ROC AUC'
    #    'Mean Fit Time', 'Mean Score Time', 'Total Time', 'Predictivity', 'Estimated Energy', 'Model Size (bytes)'
    #]

    metrics_to_plot = [
        'Accuracy', 'Negative Log Loss', 'ROC AUC',  # Added 'Negative Log Loss' and 'ROC AUC'
        'Mean Fit Time', 'Mean Score Time', 'Total Time'
    ]

    # Create a figure and subplots for each metric
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, len(metrics_to_plot) * 5))

    # Iterate through each metric and create a bar plot
    for i, metric in enumerate(metrics_to_plot):
        # Extract and sort the values for the current metric
        metric_values = [(model_name, results[model_name][metric]) for model_name in models.keys()]
        metric_values.sort(key=lambda x: x[1], reverse=True)  # Sort in descending order for best to worst

        # Print the ranked models for the current metric
        print(f"Ranked models for {metric}:")
        for rank, (model_name, metric_value) in enumerate(metric_values, start=1):
            print(f"  {rank}. {model_name} - {metric}: {metric_value:.4f}")
        print()
        
        # Create the bar plot for the current metric
        axs[i].bar([model_name for model_name, _ in metric_values], [metric_value for _, metric_value in metric_values])
        
        # Set the title and labels for the current subplot
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        
        # Rotate the x-axis labels by 90 degrees
        axs[i].tick_params(axis='x', rotation=90)

    # Adjust the layout to prevent overlapping labels
    plt.tight_layout()

    # Show the plots
    #plt.show()
    st.pyplot(fig)

    import numpy as np
    import matplotlib.colors as mcolors
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt

    def calculate_and_plot_feature_importances(models, X, y, selected_feature_indices=None):
        def normalize_importances(importances):
            return importances / np.sum(importances)

        # Initialize the feature_importances dictionary
        normalized_feature_importances = {}

        # Initialize variable to keep track of the maximum number of features
        max_features = 0

        # If selected_feature_indices is provided, use only the selected features
        if selected_feature_indices is not None:
            X = X[:, selected_feature_indices]

        # Calculate feature importances for each model
        for model_name, model in models.items():
            # Fit the model
            model.fit(X, y)
            
            # Check if the model is an instance of SPINEX
            if isinstance(model, SPINEX):
                # Calculate contributions for each feature
                predictions, contributions, _ = model.predict_contributions(X)

                # Calculate feature importances as the average absolute contributions
                feature_importances = np.mean(np.abs(contributions), axis=(1, 2))  # Average across instances and classes
                normalized_importances = feature_importances / np.sum(feature_importances)
            else:
                # Calculate feature importances
                if hasattr(model, 'feature_importances_'):
                    raw_importances = model.feature_importances_
                else:
                    r = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='accuracy')
                    raw_importances = r.importances_mean
                # Normalize the importances
                normalized_importances = raw_importances / np.sum(raw_importances)
            
            # Print normalized feature importances
            print(f"Normalized Feature Importances for {model_name}:")
            for i, importance in enumerate(normalized_importances):
                print(f"  Feature {i}: {importance:.4f}")
            print("-" * 50)

            # Update normalized_feature_importances dictionary
            normalized_feature_importances[model_name] = normalized_importances
            # Update max_features to keep track of the maximum number of features
            max_features = max(max_features, len(normalized_importances))
        
        # Plotting the normalized feature importances
        fig, ax = plt.subplots(figsize=(15, 10))

        # Set up the y-axis labels (feature indices)
        model_names = list(normalized_feature_importances.keys())

        # Create a colormap
        colormap = mcolors.LinearSegmentedColormap.from_list("custom", plt.cm.tab20.colors, N=max(max_features, 2))

        # Initialize a variable to keep track of the current x position for the bars
        current_x = 0

        # Plot the normalized feature importances for each feature
        for i, (model_name, importances) in enumerate(normalized_feature_importances.items()):
            num_features = len(importances)
            feature_indices = np.arange(num_features)
            bar_width = 1 / (max_features + 1)
            
            for j, importance in enumerate(importances):
                # Replace "nan" values with zeros
                if np.isnan(importance):
                    importance = 0
                ax.bar(current_x + j * bar_width, importance, bar_width, color=colormap(j), 
                    label=f"Feature {j}" if i == 0 else None, edgecolor='black', linewidth=1)
        
            # Set the x-axis tick labels
            ax.text(current_x + (bar_width * num_features / 2), 0, model_name, rotation=90, ha='center', va='top')
            # Update the current x position
            current_x += num_features * bar_width + bar_width

        # Set the axis labels
        ax.set_ylabel("Normalized Feature Importance")
        ax.set_xlabel("Model")

        # Remove x-axis ticks
        ax.set_xticks([])

        # Add the legend
        # Generate dynamic legend labels based on the maximum number of features
        legend_labels = [f"Feature {i}" for i in range(max_features)]
        ax.legend(legend_labels)

        # Show the plot
        plt.tight_layout()
        #plt.show()
        st.pyplot(fig)

    # Extract the indices of the selected features
    selected_features_indices = preprocessor.selected_features_
    calculate_and_plot_feature_importances(models, X, y, selected_feature_indices=selected_features_indices)

    # Instantiate and fit the SPINEX model (code provided earlier)

    # Obtain the average interaction effects using the get_global_interaction_effects method
    # Note: Update this method call if the method name is different for the classification model
    avg_interaction_effects = SPINEXClassifier.get_global_interaction_effects(X)

    # Display the average interaction effects matrix
    print("Average Interaction Effects:")
    print(avg_interaction_effects)

    # Obtain the impact of feature combinations using the feature_combination_impact_analysis method
    # Note: Update this method call if the method name is different for the classification model
    feature_combination_impact = SPINEXClassifier.feature_combination_impact_analysis(X)

    # Display the impact of feature combinations
    print("\nFeature Combination Impact Analysis:")
    for comb, impact in feature_combination_impact:
        # Convert feature indices to names, assuming original names are like 'Feature 0', 'Feature 1', etc.
        feature_names = [f"Feature {i}" for i in comb]
        print(f"Feature Combination {feature_names}: Impact = {impact:.4f}")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Obtain the average interaction effects using the get_global_interaction_effects method
    avg_interaction_effects = SPINEXClassifier.get_global_interaction_effects(X)

    # Plot the average interaction effects matrix as a heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(avg_interaction_effects, annot=True, fmt='.4f', cmap='coolwarm', center=0)
    plt.title("Average Interaction Effects")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    #plt.show()
    st.pyplot(fig)

    # Obtain the impact of feature combinations using the feature_combination_impact_analysis method
    feature_combination_impact = SPINEXClassifier.feature_combination_impact_analysis(X)

    # Prepare data for plotting feature combination impact
    feature_combinations = [str(comb) for comb, _ in feature_combination_impact]
    impact_values = [impact for _, impact in feature_combination_impact]

    # Plot the impact of feature combinations as a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=impact_values, y=feature_combinations, orient='h', color='skyblue')
    plt.title("Feature Combination Impact Analysis")
    plt.xlabel("Impact")
    plt.ylabel("Feature Combination")
    plt.show()
    #st.pyplot(fig)

    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def plot_average_interaction_network(SPINEXClassifier, X, feature_names=None):
        # Predict contributions and interaction effects
        final_predictions, contributions, interaction_effects = SPINEXClassifier.predict_contributions(X)
        
        # Calculate the average contributions
        avg_contributions = np.mean(np.abs(contributions), axis=(0, 1))

        # Determine the number of selected features based on the shape of interaction_effects
        num_selected_features = interaction_effects.shape[0]

        # Automatically generate feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {j}' for j in range(num_selected_features)]

        # Calculate and print the average interaction effects matrix
        avg_interaction_effects = np.zeros((num_selected_features, num_selected_features))
        for i in range(num_selected_features):
            for j in range(i+1, num_selected_features):
                avg_interaction_effects[i, j] = np.mean(np.abs(interaction_effects[i][j]), axis=(0, 1))
                avg_interaction_effects[j, i] = avg_interaction_effects[i, j]
        print("Average Interaction Effects:")
        print(avg_interaction_effects)

        # Create a network graph
        G = nx.DiGraph()

        # Add nodes for features
        G.add_nodes_from(feature_names)

        # Add node for prediction
        G.add_node('Prediction')

        # Add edges for individual contributions
        for i, avg_contrib in enumerate(avg_contributions):
            G.add_edge(feature_names[i], 'Prediction', weight=abs(avg_contrib), sign=avg_contrib)

        # Add edges for pairwise interactions
        for i in range(num_selected_features):
            for j in range(i+1, num_selected_features):
                interaction_value = avg_interaction_effects[i, j]
                if interaction_value != 0:
                    G.add_edge(feature_names[i], feature_names[j], weight=abs(interaction_value), sign=interaction_value)

        # Set edge colors based on the sign of interactions/contributions
        edge_colors = ["red" if G[u][v]['sign'] < 0 else "green" for u, v in G.edges()]

        # Set edge width based on the weight of interactions/contributions (scaled down by a factor)
        edge_widths = [G[u][v]['weight'] * 25 for u, v in G.edges()]

        # Set node colors (features in blue, prediction in yellow)
        node_colors = ['skyblue' if node in feature_names else 'gold' for node in G.nodes()]

        # Draw the network graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#f0f0f0")
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors,
                font_size=10, font_weight='bold', font_color='black',
                edge_color=edge_colors, width=edge_widths, arrowstyle='-|>', arrowsize=20)
        plt.title('Average Contribution and Interaction Network Graph')
        st.pyplot(fig)


    # Plot the network graph for the average contributions and interactions
    X_prioritized = None  # Set to None since we did not perform feature prioritization
    plot_average_interaction_network(SPINEXClassifier, X_prioritized if X_prioritized is not None else X)

    import seaborn as sns

    def plot_contribution_heatmaps(SPINEXClassifier, X, feature_names=None):
        # Predict contributions and interaction effects
        final_predictions, contributions, interaction_effects = SPINEXClassifier.predict_contributions(X)
        
        # Determine the number of selected features
        num_selected_features = contributions.shape[0]
        
        # Automatically generate feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {j}' for j in range(num_selected_features)]

        # Calculate the mean contributions across classes for each feature and instance
        mean_contributions = contributions.mean(axis=2)
        
        # Create a heatmap of individual feature contributions
        plt.figure(figsize=(8, 6))
        sns.heatmap(mean_contributions.T, cmap='coolwarm', cbar=True, xticklabels=feature_names, yticklabels=False)
        plt.title('Feature Contributions Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Instances')
        plt.show()

        # Calculate the mean interaction effects for each pair of features
        pairwise_interactions = np.zeros((num_selected_features, num_selected_features))
        for i in range(num_selected_features):
            for j in range(i+1, num_selected_features):
                interaction_values = interaction_effects[i][j]
                pairwise_interactions[i, j] = np.mean(interaction_values)
                pairwise_interactions[j, i] = pairwise_interactions[i, j]

        # Create a heatmap of pairwise interactions
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pairwise_interactions, cmap='coolwarm', annot=True, fmt=".2f", cbar=True, xticklabels=feature_names, yticklabels=feature_names)
        plt.title('Pairwise Interactions Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Features')
        #plt.show()
        st.pyplot(fig)


