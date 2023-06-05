import streamlit as st

from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
import time
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import itertools
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import euclidean_distances
import math
from itertools import combinations
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ThreadPoolExecutor
import itertools
from sklearn.base import clone
import sys
import pickle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SPINEX(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, distance_threshold=0.05, distance_threshold_decay=0.95, ensemble_method='None',
                 n_features_to_select=None, auto_select_features=False,
                 use_local_search=False, prioritized_features=None,
                 missing_data_method='mean_imputation', outlier_handling_method='z_score_outlier_handling',
                 exclude_method='zero'):
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold
        self.distance_threshold_decay = distance_threshold_decay
        self.ensemble_method = ensemble_method
        self.n_features_to_select = n_features_to_select
        self.auto_select_features = auto_select_features
        self.use_local_search = use_local_search
        self.prioritized_features = prioritized_features
        self.missing_data_method = missing_data_method
        self.outlier_handling_method = outlier_handling_method
        self.exclude_method = exclude_method
        self.feature_combination_size = None
        self.X_train_ = None
        self.y_train_ = None
        self.eps = 1e-8
        self.feature_combinations = None
        self.internal_call = False  # Initialize the flag as False
        self.model = None

    def _auto_select_features(self, X, y):
        selected_features = None

        if self.use_local_search:
            # Local search feature selection logic
            model = LinearRegression()
            # Set n_features_to_select='auto' and tol=None to avoid the warning
            n_features_to_select = 'auto' if self.n_features_to_select is None else self.n_features_to_select
            sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select,
                                            direction='forward', scoring='neg_mean_squared_error', tol=None)
            if y is not None:
                sfs.fit(X, y)
            X_new = sfs.transform(X)
            selected_features = sfs.get_support(indices=True)

        elif self.n_features_to_select is not None and not self.use_local_search:
            correlations = np.abs(np.corrcoef(X, y, rowvar=False)[-1, :-1])
            top_feature_indices = np.argsort(correlations)[-self.n_features_to_select:]
            X_new = X[:, top_feature_indices]
            selected_features = top_feature_indices

        else:
            X_new = X

        return X_new, selected_features

    def fit(self, X, y):
        # External call: Apply ensemble logic (if specified) and train the model
        if not self.internal_call:
            # Perform feature selection if auto_select_features is True
            if self.auto_select_features:
                X, self.selected_features_ = self._auto_select_features(X, y)

            # Set the training data attributes
            self.X_train_ = X.copy()
            self.y_train_ = y.copy()

            # Dynamically determine the feature combination size (n) based on the number of features
            n_features = X.shape[1]
            self.feature_combination_size = math.ceil(n_features / 2)

            # Ensure that the combination size does not exceed the number of features
            self.feature_combination_size = min(self.feature_combination_size, n_features)
            self.feature_combinations = list(combinations(range(min(self.X_train_.shape[1], X.shape[1])), self.feature_combination_size))

            if self.ensemble_method == 'bagging':
                base_model = SPINEX(auto_select_features=self.auto_select_features, ensemble_method='None')  # Set other parameters as needed
                self.model = BaggingRegressor(estimator=base_model, n_estimators=10, random_state=42)
                self.internal_call = True  # Set the flag to True for internal call
                self.model.fit(X, y)
                self.internal_call = False  # Set the flag back to False for future external calls
            elif self.ensemble_method == 'boosting':
                base_model = SPINEX(auto_select_features=self.auto_select_features, ensemble_method='None')  # Set other parameters as needed
                self.model = AdaBoostRegressor(estimator=base_model, n_estimators=10, random_state=42)
                self.internal_call = True  # Set the flag to True for internal call
                self.model.fit(X, y)
                self.internal_call = False  # Set the flag back to False for future external calls
            elif self.ensemble_method == 'stacking':
                estimators = [('model1', SPINEX(ensemble_method='None')),
                              ('model2', SPINEX(ensemble_method='None'))]
                meta_estimator = ElasticNet()
                self.model = StackingRegressor(estimators=estimators, final_estimator=meta_estimator)
                self.internal_call = True  # Set the flag to True for internal call
                self.model.fit(X, y)
                self.internal_call = False  # Set the flag back to False for future external calls
            else:
                # No ensemble case (use SPINEX directly)
                self.model = None
                # Include the fitting logic directly here
                # Use the existing logic from your original fit method
                # ...
        return self

    def _calculate_feature_combination_distances(self, instances, train_instances):
        def calculate_combination_distance(comb):
            #comb_distance = np.sqrt(np.sum((train_instances[:, comb] - instances[:, comb][:, np.newaxis]) ** 2, axis=-1)) Euclidean 
            comb_distance = np.sum(np.abs(train_instances[:, comb] - instances[:, comb][:, np.newaxis]), axis=-1) #Manhattan
            return comb_distance

        distances = np.zeros((instances.shape[0], train_instances.shape[0]))

        # Parallelize the loop
        comb_distances = Parallel(n_jobs=-1)(delayed(calculate_combination_distance)(comb) for comb in self.feature_combinations)

        for comb_distance in comb_distances:
            distances += comb_distance

        overall_distance = distances / len(self.feature_combinations)
        return overall_distance

    def predict(self, X):
        # Check types and shapes
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        assert isinstance(self.y_train_, np.ndarray)
        assert len(self.y_train_.shape) == 1
        
        # Check if ensemble logic was applied
        if self.model is not None:
            if self.auto_select_features:
                X = X[:, self.selected_features_]  # Use the ensemble model for prediction
            predictions = self.model.predict(X)
        else:
            # No ensemble case (use SPINEX directly)
            # Calculate distances based on feature combinations for all instances in X
            distances = self._calculate_feature_combination_distances(X, self.X_train_)
            # Find the indices of the k-nearest neighbors for all instances in X
            sorted_indices = np.argsort(distances, axis=1)
            nearest_indices = sorted_indices[:, :self.n_neighbors]
            # Calculate the weighted average of the target values of nearest neighbors
            nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)
            # Apply distance threshold decay
            decayed_distance_threshold = self.distance_threshold * self.distance_threshold_decay
            weights = 1 / (nearest_distances + decayed_distance_threshold)
            nearest_targets = self.y_train_[nearest_indices]
            predictions = np.sum(nearest_targets * weights, axis=1) / np.sum(weights, axis=1)

        # Ensure predictions is a 1D array
        predictions = np.atleast_1d(predictions)
        assert len(predictions.shape) == 1
        return predictions

    def predict_contributions(self, X, instances_to_predict=None):
        if instances_to_predict is None:
            instances_to_predict = range(X.shape[0])
        
        # Use only selected features
        selected_features = self.X_train_.shape[1]
        X = X[:, :selected_features]

        # Check if ensemble logic was applied
        if self.model is not None:
            contributions = []
            for estimator in self.model.estimators_:
                _, est_contributions, _ = estimator.predict_contributions(X[instances_to_predict])
                contributions.append(est_contributions)
            contributions = np.mean(contributions, axis=0)
            return contributions
        else:

            # Make overall predictions for selected instances
            final_predictions = self.predict(X[instances_to_predict])

        # Calculate feature contributions
        contributions = []
        for i in range(X.shape[1]):
            # Prediction with the feature excluded (set to zero or mean value)
            X_excluded = X.copy()
            X_excluded[:, i] = 0  # You may replace this with mean imputation
            excluded_predictions = self.predict(X_excluded[instances_to_predict])

            # Contribution of the feature
            feature_contributions = final_predictions - excluded_predictions
            contributions.append(feature_contributions)

        # Combine contributions into an array
        contributions = np.array(contributions).T

        # Calculate pairwise interaction effects
        interaction_effects = []
        for i in range(X.shape[1]):
            interaction_effects_row = []
            for j in range(X.shape[1]):
                if i == j:
                    interaction_effects_row.append(0)
                    continue
                # Prediction with both features i and j excluded
                X_excluded = X.copy()
                X_excluded[:, i] = 0
                X_excluded[:, j] = 0
                excluded_predictions = self.predict(X_excluded[instances_to_predict])

                # Interaction effect of features i and j
                interaction_effect = final_predictions - excluded_predictions - contributions[:, i] - contributions[:, j]
                interaction_effects_row.append(interaction_effect)
            interaction_effects.append(interaction_effects_row)

        # Combine interaction effects into an array
        interaction_effects = np.array(interaction_effects, dtype=object)

        return final_predictions, contributions, interaction_effects

    def get_feature_importance(self, X, instances_to_explain=None):
        if instances_to_explain is None:
            instances_to_explain = range(X.shape[0])
        
        # Use only selected features
        selected_features = self.X_train_.shape[1]
        X = X[:, :selected_features]

        # Check if ensemble logic was applied
        if self.model is not None:
            feature_importances = []
            interaction_effects_list = []
            for estimator in self.model.estimators_:
                # Get feature importances and interaction effects for each base estimator
                est_importances, est_interaction_effects = estimator.get_feature_importance(X, instances_to_explain)
                feature_importances.append(est_importances)
                interaction_effects_list.append(est_interaction_effects)
            return feature_importances, interaction_effects_list
        else:
            # Contribution of each feature to the model's predictions for each instance
            predictions, contributions, interaction_effects = self.predict_contributions(X)

            # Calculate the contribution of each feature
            feature_importances = np.mean(np.abs(contributions), axis=0)

            # Calculate the interaction effects
            interaction_effects = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    interaction_effect = np.mean(predictions - contributions[:, i] - contributions[:, j])
                    interaction_effects[i, j] = interaction_effect
                    interaction_effects[j, i] = interaction_effect

            return feature_importances, interaction_effects

    def _handle_missing_data(self, X, y):
        if self.missing_data_method == 'mean_imputation':
            col_means = np.nanmean(X, axis=0)
            return np.where(np.isnan(X), col_means, X), y  # Return y unchanged
        elif self.missing_data_method == 'deletion':
            not_missing = ~np.isnan(X).any(axis=1)
            return X[not_missing], y[not_missing]
        else:
            raise ValueError("Unsupported missing_data_method. Please use 'mean_imputation' or 'deletion'.")

    def _handle_outliers(self, X, y):
        if self.outlier_handling_method == 'z_score_outlier_handling':
            z_scores = np.abs(stats.zscore(X))
            not_outliers = (z_scores < 3).all(axis=1)
            return X[not_outliers], y[not_outliers]
        elif self.outlier_handling_method == 'iqr_outlier_handling':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            not_outliers = np.logical_and(X >= lower_bound, X <= upper_bound).all(axis=1)
            return X[not_outliers], y[not_outliers]
        else:
            raise ValueError("Unsupported outlier_handling_method. Please use 'z_score_outlier_handling' or 'iqr_outlier_handling'.")

    def get_global_interaction_effects(self, X):
        # Use only selected or prioritized features
        selected_features = self.X_train_.shape[1]
        X = X[:, :selected_features]

        # Check if ensemble logic was applied
        if self.model is not None:
            avg_interaction_effects_list = []
            for estimator in self.model.estimators_:
                # Get average interaction effects for each base estimator
                est_avg_interaction_effects = estimator.get_global_interaction_effects(X)
                avg_interaction_effects_list.append(est_avg_interaction_effects)
            return avg_interaction_effects_list
        else:
            # Calculate the contribution of each feature to the model's predictions for the entire dataset
            predictions, contributions, interaction_effects = self.predict_contributions(X)

            # Calculate the average interaction effects for all pairs of features
            avg_interaction_effects = np.zeros((X.shape[1], X.shape[1]))
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    # Interaction effect of features i and j
                    interaction_effect = np.mean(interaction_effects[i][j])
                    avg_interaction_effects[i, j] = interaction_effect
                    avg_interaction_effects[j, i] = interaction_effect

            return avg_interaction_effects

    def feature_combination_impact_analysis(self, X):
        # Use only selected or prioritized features
        selected_features = self.X_train_.shape[1]
        X = X[:, :selected_features]

        # Check if ensemble logic was applied
        if self.model is not None:
            feature_combination_impact_list = []
            for estimator in self.model.estimators_:
                # Get feature combination impact analysis for each base estimator
                est_feature_combination_impact = estimator.feature_combination_impact_analysis(X)
                feature_combination_impact_list.append(est_feature_combination_impact)
            return feature_combination_impact_list
        else:
            def calculate_combination_impact(comb, X, base_predictions, model):
                X_excluded = X.copy()
                X_excluded[:, comb] = 0  # You may replace this with mean imputation
                excluded_predictions = model.predict(X_excluded)
                impact = np.mean(np.abs(base_predictions - excluded_predictions))
                return comb, impact

            # Calculate the base predictions for the entire dataset
            base_predictions = self.predict(X)

            # Initialize an empty dictionary to store the impact of each feature combination
            feature_combination_impact = {}

            # Iterate over possible feature combinations
            n_features = X.shape[1]

            # Parallelize the loop
            combination_impacts = Parallel(n_jobs=-1)(delayed(calculate_combination_impact)(comb, X, base_predictions, self) for comb in itertools.chain.from_iterable(
                itertools.combinations(range(n_features), combination_size) for combination_size in range(1, n_features + 1)))

            # Sort feature combinations by their impact
            sorted_combinations = sorted(combination_impacts, key=lambda x: x[1], reverse=True)

            return sorted_combinations


    def get_params(self, deep=True):
        return {
            'n_neighbors': self.n_neighbors,                          #The number of nearest neighbors to consider in the SPINEX model. It determines how many neighbors to use when making predictions.
            'distance_threshold': self.distance_threshold,
            'distance_threshold_decay':self.distance_threshold_decay,
            'ensemble_method': self.ensemble_method,                #The ensemble method to use. It can be set to "bagging", "boosting", or None. Bagging and boosting are ensemble techniques that combine multiple base models to improve prediction accuracy.
            'n_features_to_select': self.n_features_to_select,        #The number of features to select when auto_select_features is set to True. It specifies how many features to retain during automatic feature selection.
            'auto_select_features': self.auto_select_features,        #A boolean flag indicating whether to automatically select features. If True, the model will automatically select a subset of features based on their importance.
            'use_local_search': self.use_local_search,                #A boolean flag indicating whether to use local search for feature selection. If True, the model will perform a local search to find the best subset of features.
            'prioritized_features': self.prioritized_features,        #A list of prioritized features. If auto_select_features is False, the model will only consider the features specified in this list.
            'missing_data_method': self.missing_data_method,          #The method for handling missing data. It can be set to "mean_imputation" or "deletion".
            'outlier_handling_method': self.outlier_handling_method,  #The method for handling outliers. It can be set to "z_score_outlier_handling" or "iqr_outlier_handling".
            'exclude_method': self.exclude_method                     #The method for excluding features when calculating contributions. It can be set to "zero" or "mean".
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def imputation(X, statistic='mean'):
    """
    Fill missing values with a given statistic (mean, median, or mode).
    
    Parameters:
    X (np.array): Input data.
    statistic (str): The statistic to use for imputation. Options: 'mean', 'median', 'mode'. Default is 'mean'.
    
    Returns:
    np.array: The array with missing values replaced.
    """
    # Input validation
    assert isinstance(X, np.ndarray), "Input data should be a numpy array."
    assert statistic in ['mean', 'median', 'mode'], "Invalid statistic. Choose from 'mean', 'median', 'mode'."

    # Compute the chosen statistic
    if statistic == 'mean':
        stat_values = np.nanmean(X, axis=0)
    elif statistic == 'median':
        stat_values = np.nanmedian(X, axis=0)
    elif statistic == 'mode':
        stat_values = stats.mode(X, nan_policy='omit').mode
    
    # Return the array with missing values replaced
    return np.where(np.isnan(X), stat_values, X)


def deletion(X, y, missing_values=np.nan):
    """
    Removes rows in both X and y where X has missing values.

    Parameters:
    X (np.array): Input features.
    y (np.array): Target variable.
    missing_values: The values to be considered as "missing". Default is np.nan.

    Returns:
    Tuple[np.array, np.array]: Tuple of arrays (X, y) with rows containing missing values removed.
    """
    # Input validation
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y should be numpy arrays."

    not_missing = ~np.isnan(X).any(axis=1)
    return X[not_missing], y[not_missing]


def z_score_outlier_handling(X, y, threshold=3):
    """
    Removes outliers from X and y using Z-score method.

    Parameters:
    X (np.array): Input features.
    y (np.array): Target variable.
    threshold (float): The Z-score threshold to use for detecting outliers. Default is 3.

    Returns:
    Tuple[np.array, np.array]: Tuple of arrays (X, y) with outliers removed.
    """
    # Input validation
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y should be numpy arrays."
    assert threshold > 0, "Threshold should be a positive number."

    z_scores = np.abs(stats.zscore(X))
    not_outliers = (z_scores < threshold).all(axis=1)
    return X[not_outliers], y[not_outliers]


def iqr_outlier_handling(X, y, k=1.5):
    """
    Removes outliers from X and y using IQR method.

    Parameters:
    X (np.array): Input features.
    y (np.array): Target variable.
    k (float): The multiplier for IQR. Default is 1.5.

    Returns:
    Tuple[np.array, np.array]: Tuple of arrays (X, y) with outliers removed.
    """
    # Input validation
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y should be numpy arrays."
    assert k > 0, "k should be a positive number."

    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    not_outliers = np.logical_and(X >= lower_bound, X <= upper_bound).all(axis=1)
    return X[not_outliers], y[not_outliers]


def normalize_importances(importances):
    return importances / np.sum(importances)


import io
    
# Set up the Streamlit interface
st.title('SPINEX: Similarity-based Predictions with Explainable Neighbors Exploration [SPINEXRegressor - Demo]')

# Add your name, website, and email using Markdown
st.markdown("## About Us")
st.markdown("- **Name:** M.Z. Naser, M.K. al-Bashiti, A.Z. Naser")
st.markdown("- **Website:** [www.mznaser.com](http://www.mznaser.com)")
#st.markdown("- **Email:** mznaser@clemson.edu")

# Create a sidebar with a title and text
st.sidebar.title("Read Me")
st.sidebar.markdown("Please feel free to read out preprint () to learn more about SPINEX, its derivatives and fuctions. You can cite our work as, You can cite our work as, Naser M.Z., Al-Bashiti M.K., Naser A.Z. (2023). SPINEX: Similarity-based Predictions with Explainable Neighbors Exploration for Regression and Classification Tasks in Machine Learning. Preprint. https://doi.org/10.48550/arXiv.2306.01029. Please be patient with us while we improve SPINEX. Please note that run time(s) can vary at the moment.")

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
    
    X = X.values
    y = y.values

    # Standardize the entire dataset
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    # Define the missing data and outlier handling methods you want to use
    missing_data_method = "deletion"
    outlier_handling_method = "z_score_outlier_handling"

    # Preprocess the data using the chosen techniques before fitting the SPINEX model
    if missing_data_method == "mean_imputation":
        X = imputation(X, statistic='mean')
    elif missing_data_method == "deletion":
        X, y = deletion(X, y)

    if outlier_handling_method == "z_score_outlier_handling":
        X, y = z_score_outlier_handling(X, y)
    elif outlier_handling_method == "iqr_outlier_handling":
        X, y = iqr_outlier_handling(X, y)


    # Select only the prioritized features (features 0 and 2)
    prioritized_features = []

    # Check if there are prioritized features
    if len(prioritized_features) > 0:
        X_prioritized = X[:, prioritized_features]
    else:
        X_prioritized = None

    # Initialize the SPINEX model with desired parameters
    SPINEXRegressor = SPINEX(n_neighbors=5, distance_threshold=0.5, 
                    distance_threshold_decay=0.95, ensemble_method= None,
                    n_features_to_select=None, auto_select_features=False,
                    use_local_search=False, prioritized_features=None,
                    missing_data_method='mean_imputation', outlier_handling_method='z_score_outlier_handling',
                    exclude_method='zero')

    # Fit the model on the training data
    if X_prioritized is not None:
        SPINEXRegressor.fit(X_prioritized, y)
    else:
        SPINEXRegressor.fit(X, y)  # for not prioritized features

    # StackingSPINEX
    stacking_model = StackingRegressor(
        estimators=[('spinex', SPINEXRegressor), ('linear', LinearRegression())],
        final_estimator=LinearRegression(),
        cv=5
    )

    # BaggingSPINEX
    bagging_model = BaggingRegressor(
        estimator=SPINEXRegressor,
        n_estimators=10,
        random_state=42
    )

    # BoostingSPINEX
    boosting_model = AdaBoostRegressor(
        estimator=SPINEXRegressor,
        n_estimators=10,
        random_state=42
    )

    # Fit the ensemble models on the training data
    stacking_model.fit(X, y)
    bagging_model.fit(X, y)
    boosting_model.fit(X, y)

    # Define the models to compare
    models = {
        'SPINEX': SPINEXRegressor,
        'StackingSPINEX': stacking_model,
        'BaggingSPINEX': bagging_model,
        'BoostingSPINEX': boosting_model,
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        'Lasso': Lasso(alpha=1.0, max_iter=10000),
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'HuberRegressor': HuberRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'CatBoostRegressor': CatBoostRegressor(verbose=False),
        'XGBRegressor': XGBRegressor(objective='reg:squarederror', eval_metric='mae', n_jobs=-1),
        'LGBMRegressor': LGBMRegressor(objective='regression', metric='mae', n_jobs=-1),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor()
    } 

    models['SPINEX'] = SPINEX()

    def estimated_energy(model, total_time):
        model_size = sys.getsizeof(pickle.dumps(model))
        return model_size * total_time

    # Define the cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluation metrics
    scoring_metrics = ['neg_mean_absolute_error', 'r2']

    # Initialize results dictionary
    results = {}

    # Perform cross-validation for each model
    for model_name, model in models.items():
        # Print the model_name
        print(f"Model: {model_name}")
        
        # Record the start time
        start_time = time.time()
        
        # Perform cross-validation and store the scores
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring_metrics, return_train_score=False)
        
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
        mean_mae = -np.mean(scores['test_neg_mean_absolute_error'])
        mean_r2 = np.mean(scores['test_r2'])
        
        # Calculate the average fitting and scoring times
        mean_fit_time = np.mean(scores['fit_time'])
        mean_score_time = np.mean(scores['score_time'])
        total_time = end_time - start_time

        # Calculate the estimated energy
        energy = estimated_energy(model, total_time)

        # Store the results
        results[model_name] = {
            'Mean Absolute Error': mean_mae,
            'R^2 Score': mean_r2,
            'Mean Fit Time': mean_fit_time,
            'Mean Score Time': mean_score_time,
            'Total Time': total_time,
            'Estimated Energy': energy
        }

    # Display the results
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print("-" * 50)

    # Define the metrics to plot
    metrics_to_plot = [('Mean Absolute Error', 1.0), ('R^2 Score', 1.0), ('Mean Fit Time', -1.0), ('Mean Score Time', -1.0), ('Total Time', -1.0), ('Estimated Energy', -1.0)]

    # Calculate the composite metric and normalize the metrics
    normalized_results = {}
    for model_name, metrics in results.items():
        composite_metric = sum(metrics[metric] * weight for metric, weight in metrics_to_plot)
        normalized_metrics = {metric: metrics[metric] / max(results[model][metric] for model in results) for metric, _ in metrics_to_plot}
        normalized_metrics['Composite Metric'] = composite_metric
        normalized_results[model_name] = normalized_metrics

    # Add the composite metric to the list of metrics to plot
    metrics_to_plot.append(('Composite Metric', 1.0))

    # Create a figure and subplots for each metric
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, len(metrics_to_plot) * 5))

    # Iterate through each metric and create a bar plot
    for i, (metric, weight) in enumerate(metrics_to_plot):
        # Extract and sort the values for the current metric
        metric_values = [(model_name, normalized_results[model_name][metric]) for model_name in models.keys()]
        metric_values.sort(key=lambda x: x[1], reverse=weight > 0)  # Sort based on the weight

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
#################################################################################
#################################################################################
    import numpy as np
    import matplotlib.colors as mcolors
    from sklearn.inspection import permutation_importance

    def normalize_importances(importances):
        return importances / np.sum(importances)

    # Initialize the feature_importances dictionary
    normalized_feature_importances = {}
    processed_models = set()

    # Initialize variable to keep track of the maximum number of features
    max_features = 0

    # Define the models dictionary (as before)
    models = {
        'SPINEX': SPINEXRegressor,
        'StackingSPINEX': stacking_model,
        'BaggingSPINEX': bagging_model,
        'BoostingSPINEX': boosting_model,
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        'Lasso': Lasso(alpha=1.0, max_iter=10000),
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'HuberRegressor': HuberRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'CatBoostRegressor': CatBoostRegressor(verbose=False),
        'XGBRegressor': XGBRegressor(objective='reg:squarederror', eval_metric='mae', n_jobs=-1),
        'LGBMRegressor': LGBMRegressor(objective='regression', metric='mae', n_jobs=-1),
        'SVR': SVR(),
        'KNeighborsRegressor': KNeighborsRegressor()
    } 

    # Perform cross-validation for each model
    for model_name, model in models.items():
        # Fit the model on the training data
        if model_name == "SPINEX":
            # Use the appropriate data for the SPINEX model
            model.fit(X_prioritized if X_prioritized is not None else X, y)
            
            # Fit and print normalized feature importances and interaction effects for SPINEX
            feature_importances, interaction_effects = SPINEXRegressor.get_feature_importance(X_prioritized if X_prioritized is not None else X)

            if isinstance(feature_importances, np.ndarray):
                # Single model case
                feature_importances = [feature_importances]  # Convert numpy array to list

            # Now, you can iterate over feature_importances whether it's a list of numpy arrays (ensemble case) or a list containing a single numpy array (single model case)
            for estimator_idx, est_importances in enumerate(feature_importances):
                normalized_importances = est_importances / np.sum(est_importances)
                print(f"Normalized Feature Importances for Base Estimator {estimator_idx} in SPINEX:")
                for i, importance in enumerate(normalized_importances):
                    print(f"  Feature {i}: {importance:.4f}")
                if interaction_effects is not None:
                    print(f"Interaction Effects for Base Estimator {estimator_idx} in SPINEX:")
                    print(interaction_effects[estimator_idx])
            else:
                # Ensemble case
                for estimator_idx, (est_importances, est_interaction_effects) in enumerate(zip(feature_importances, interaction_effects)):
                    normalized_importances = est_importances / np.sum(est_importances)
                    print(f"Normalized Feature Importances for Base Estimator {estimator_idx} in SPINEX:")
                    for i, importance in enumerate(normalized_importances):
                        print(f"  Feature {i}: {importance:.4f}")
                    print(f"Interaction Effects for Base Estimator {estimator_idx} in SPINEX:")
                    print(est_interaction_effects)
        else:
            # Use all features for other models
            model.fit(X, y)
            # Calculate feature importances
            if hasattr(model, 'feature_importances_'):
                raw_importances = model.feature_importances_
            else:
                r = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error')
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
###########################################################################
###########################################################################
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Instantiate and fit the SPINEX model (code provided earlier)

    # Obtain the average interaction effects using the get_global_interaction_effects method
    avg_interaction_effects_list = SPINEXRegressor.get_global_interaction_effects(X)

    # Check if ensemble logic was applied
    if isinstance(avg_interaction_effects_list, list):
        # Ensemble case: Plot interaction effects for each base estimator
        for estimator_idx, avg_interaction_effects in enumerate(avg_interaction_effects_list):
            fig, ax = plt.subplots(figsize=(10, 8))  # Modified line
            sns.heatmap(avg_interaction_effects, annot=True, fmt='.4f', cmap='coolwarm', center=0, ax=ax)  # Modified line
            ax.set_title(f"Average Interaction Effects for Base Estimator {estimator_idx}")  # Modified line
            ax.set_xlabel("Feature Index")  # Modified line
            ax.set_ylabel("Feature Index")  # Modified line
            st.pyplot(fig)  # Modified line
    else:
        # Single model case: Plot interaction effects
        fig, ax = plt.subplots(figsize=(10, 8))  # Modified line
        sns.heatmap(avg_interaction_effects_list, annot=True, fmt='.4f', cmap='coolwarm', center=0, ax=ax)  # Modified line
        ax.set_title("Average Interaction Effects")  # Modified line
        ax.set_xlabel("Feature Index")  # Modified line
        ax.set_ylabel("Feature Index")  # Modified line
        st.pyplot(fig)  # Modified line

    # Obtain the average interaction effects using the get_global_interaction_effects method
    avg_interaction_effects_list = SPINEXRegressor.get_global_interaction_effects(X)

    # Obtain the impact of feature combinations using the feature_combination_impact_analysis method
    feature_combination_impact_list = SPINEXRegressor.feature_combination_impact_analysis(X)

    # Check if ensemble logic was applied
    if isinstance(feature_combination_impact_list[0], list):
        # Ensemble case: Plot feature combination impact for each base estimator
        for estimator_idx, feature_combination_impact in enumerate(feature_combination_impact_list):
            feature_combinations = [str(comb) for comb, _ in feature_combination_impact]
            impact_values = [impact for _, impact in feature_combination_impact]
            fig, ax = plt.subplots(figsize=(10, 6))  # Modified line
            sns.barplot(x=impact_values, y=feature_combinations, orient='h', color='skyblue', ax=ax)  # Modified line
            ax.set_title(f"Feature Combination Impact Analysis for Base Estimator {estimator_idx}")  # Modified line
            ax.set_xlabel("Impact")  # Modified line
            ax.set_ylabel("Feature Combination")  # Modified line
            st.pyplot(fig)  # Modified line
    else:
        # Single model case: Plot feature combination impact
        feature_combinations = [str(comb) for comb, _ in feature_combination_impact_list]
        impact_values = [impact for _, impact in feature_combination_impact_list]
        fig, ax = plt.subplots(figsize=(10, 6))  # Modified line
        sns.barplot(x=impact_values, y=feature_combinations, orient='h', color='skyblue', ax=ax)  # Modified line
        ax.set_title("Feature Combination Impact Analysis")  # Modified line
        ax.set_xlabel("Impact")  # Modified line
        ax.set_ylabel("Feature Combination")  # Modified line
        st.pyplot(fig)  #Sure! Here's the updated code block with the inclusion of Streamlit's `st.pyplot(fig)` for each plotting function:

###########################################################################
###########################################################################

    import networkx as nx
    import numpy as np
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import streamlit as st

    def plot_average_interaction_network(avg_interaction_effects, feature_names=None):
        # Determine the number of selected features
        num_selected_features = avg_interaction_effects.shape[0]

        # Automatically generate feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature {j}' for j in range(num_selected_features)]

        # Create a network graph
        G = nx.DiGraph()

        # Add nodes for features
        G.add_nodes_from(feature_names)

        # Add node for prediction
        G.add_node('Prediction')

        # Add edges for pairwise interactions
        for i in range(num_selected_features):
            for j in range(i+1, num_selected_features):
                interaction_value = avg_interaction_effects[i, j]
                if interaction_value != 0:
                    G.add_edge(feature_names[i], feature_names[j], weight=abs(interaction_value), sign=interaction_value)

        # Set edge colors based on the sign of interactions/contributions
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "white", "green"])
        edge_colors = [cmap(G[u][v]['sign'] / 2 + 0.5) for u, v in G.edges()]

        # Set edge width based on the weight of interactions/contributions (scaled down by a factor)
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]

        # Set node colors (features in blue, prediction in yellow)
        node_colors = ['skyblue' if node in feature_names else 'gold' for node in G.nodes()]

        # Draw the network graph
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.gca().set_facecolor("#f0f0f0")
        pos = nx.circular_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors,
                font_size=10, font_weight='bold', font_color='black',
                edge_color=edge_colors, width=edge_widths, arrowstyle='-|>', arrowsize=20)
        plt.title('Average Contribution and Interaction Network Graph')

        # Show the plot in Streamlit
        st.pyplot(fig)

    # Obtain the average interaction effects using the get_global_interaction_effects method
    avg_interaction_effects_list = SPINEXRegressor.get_global_interaction_effects(X)

    # Check if ensemble logic was applied
    if isinstance(avg_interaction_effects_list, list):
        # Ensemble case: Plot interaction network for each base estimator
        for estimator_idx, avg_interaction_effects in enumerate(avg_interaction_effects_list):
            print(f"Interaction Network for Base Estimator {estimator_idx}:")
            plot_average_interaction_network(avg_interaction_effects)
    else:
        # Single model case: Plot interaction network
        plot_average_interaction_network(avg_interaction_effects_list)




###########################################################################
###########################################################################

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def plot_all_feature_contributions(model, X):
        _, contributions, _ = model.predict_contributions(X)

        # Create a DataFrame with the feature values and contributions
        data = pd.DataFrame(contributions, columns=[f'Feature {i}' for i in range(X.shape[1])])

        # Melt the DataFrame to a long format
        data = data.melt(var_name='Feature', value_name='Contribution')

        # Create a new column for the x-axis values (same for all features)
        data['X Value'] = np.tile(X[:, 0], X.shape[1])  # Replace 0 with the index of the feature you want to use for the x-axis

        # Use seaborn's scatterplot function to plot the data
        fig, ax = plt.subplots(figsize=(8, 6)) # Add this line
        sns.scatterplot(data=data, x='X Value', y='Contribution', hue='Feature', alpha=0.7, ax=ax) # Modify this line
        ax.set_title('Feature Contributions for All Features') # Modify this line

        # Show the plot in Streamlit
        st.pyplot(fig)

    # Assuming you have your training data in X_train and y_train
    model = SPINEX()  # Initialize your model
    model.fit(X, y)

    plot_all_feature_contributions(model, X)

