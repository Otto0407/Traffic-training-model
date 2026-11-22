"""Standard tabular models the MLP is compared against.

Gradient-boosted trees are the usual strong choice for tabular data, and
multinomial logistic regression is the simplest learned model. Both use the
same preprocessed features and the same train / validation / test split as
the MLP, so the comparison is like for like.
"""
import warnings

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


def fit_logistic_regression(data, seed: int = 0):
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(data.X_train, data.y_train)
    return model


def fit_lightgbm(data, seed: int = 0):
    """LightGBM with early stopping on the validation set, mirroring the MLP."""
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    # eval_set is deprecated in newer LightGBM but is the only form older 4.x accepts.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
        model.fit(
            data.X_train,
            data.y_train,
            eval_set=[(data.X_val, data.y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    return model
