import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ridge_regression(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 0,
    alpha: float = 100.0,
    use_cv: bool = False,
    cv_folds: int = 5,
) -> float:
    """Fit a Ridge regression model with optional cross-validation and return the R^2 score.

    This function splits the data into training and test sets, fits either Ridge or RidgeCV
    (using the provided `alphas` grid, or a default logarithmic grid), and computes the
    variance-weighted R² score on the held-out test set.

    Args:
        x (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target array of shape (n_samples,) or (n_samples, n_targets).
        test_size (float, optional): Proportion of the dataset to include in the test split.
            Must be between 0 and 1. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 0.
        alpha (float, optional): Regularization strength for Ridge when `use_cv` is False.
            Must be a positive float. Defaults to 1.0.
        use_cv (bool, optional): Whether to perform cross-validated ridge regression
            using `RidgeCV`. If True, the `alpha` parameter is ignored. Defaults to False.
        cv_folds (int, optional): Number of folds for cross-validation when `use_cv` is True.
            Defaults to 5.

    Returns:
        float: The R² score (variance-weighted) computed on the test set.
    """
    # split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # select model
    if use_cv:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge_cv', RidgeCV(alphas=np.logspace(-3, 4, 50), cv=cv_folds, fit_intercept=True)),
        ])
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, fit_intercept=True)),
        ])

    # train model
    model.fit(x_train, y_train)
    # evaluate model
    score = r2_score(
        y_test,
        model.predict(x_test),
        multioutput="variance_weighted"
    )
    return score
