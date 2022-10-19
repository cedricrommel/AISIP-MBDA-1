import pickle
from pathlib import Path
from logging import warning
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit

import torchio as tio

from utils.difumo_utils import project_to_difumo


def preprocess_label(Y_t, use_dict=None, return_dict=False):
    """
    Preprocess label so that they match classifier like format

    Parameters
    ----------
    Y_t: pandas DataFrame
        labels to preprocess
    use_dict: dictionary
        If a dictionary labels -> class number is already known
    return_dict: bool
        If true, returns the dictionary labels -> class number
    Returns
    --------
    Y: np array
        np array where each label is replaced by its class number
    dict (optional): dict
        dictionary label -> class number (only returned if return_dict is True)
    """
    if use_dict is None:
        print("     >>> creating Y_dict")
        Y_dict = {v: k for k, v in enumerate(Y_t["contrast"].unique())}
    else:
        Y_dict = use_dict

    print("     >>> Applying Y_dict")
    if return_dict:
        return Y_t["contrast"].apply(lambda x: Y_dict[x]).values, Y_dict
    else:
        return Y_t["contrast"].apply(lambda x: Y_dict[x]).values


class AugmentedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, f):
        """
        Trains a classifier using an augmentation method.
        Parameters
        ----------
        model: BaseEstimator
            The classifier used
        f: function (X, Y) -> X_fake, Y_fake
            The data augmentation function that
            generates fake (labeled) data from
            input data
        """

        self.model = model
        self.f = f

    def fit(self, X, Y):
        if self.f is None:
            return self.model.fit(X, Y)
        else:
            X_fake, Y_fake = self.f(X, Y)
            return self.model.fit(X_fake, Y_fake)

    def predict(self, X, y):
        return self.model.predict(X, y)

    def score(self, X, y):
        return self.model.score(X, y)


def do_classif(
    images_path, Z_inv, mask, labels, f, method_name, output_fname, models_dir,
    train_size, n_splits=5, n_jobs=5
):
    """
    Tries 4 different classifier with the given augmentation method
    Parameters
    ----------
    X: np array of shape (n_samples, n_features)
        Input data
    Y: np array of shape (n_samples,)
        Labels
    f: function of X, Y
        f returns fake data and fake labels
    method_name: str
        the name of the method used to produce fake data
    filename: str
        filename is the path result file
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    n_jobs: int, default: None
        The maximum number of concurrently running jobs,
    """

    models = []
    # parameters for MLP only
    params_2L = {
        "activation": "relu",
        "solver": "adam",
        "learning_rate": "constant",
        "momentum": 0.9,
        "learning_rate_init": 0.0001,
        "alpha": 0.00001,
        "random_state": 0,
        "batch_size": 32,
        "hidden_layer_sizes": (1024, 1024),
        "max_iter": 20000,
    }

    models.append(
        (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "LDA")
    )
    models.append((RandomForestClassifier(verbose=True), "RF"))
    # models.append((MLPClassifier(verbose=True, **params_2L), "MLP"))
    # models.append(
    #     (
    #         GridSearchCV(
    #             LogisticRegression(
    #                 solver="lbfgs",
    #                 tol=1e-4,
    #                 random_state=11,
    #                 penalty="l2",
    #                 max_iter=20000,
    #                 n_jobs=1,
    #                 verbose=True,
    #             ),
    #             {"C": [0.1, 0.01, 0.001, 1]},
    #             cv=5,
    #         ),
    #         "LogReg",
    #     )
    # )

    def do_split(split, data_paths, projected_X, Y, f, model, subjects, n_jobs, checkpoint_base_name, i_split):
        checkpoint_name = f"{checkpoint_base_name}_{i_split}.pkl"
        train, test = split
        train, test = subjects[train], subjects[test]
        train = Y["subject"].isin(train)
        test = Y["subject"].isin(test)
        print(" >> preprocessing labels")
        Y_train, labels_dict = preprocess_label(Y[train], return_dict=True)
        try:
            Y_test = preprocess_label(Y[test], use_dict=labels_dict)
        except KeyError:
            warning("NOT PASSING TRAIN LABELS DICTIONARY TO TEST SET")
            Y_test = preprocess_label(Y[test])
        X_train = np.array(data_paths)[train.values]
        print(" >> Loading test projected images")
        X_test = projected_X[test.values]
        clf = AugmentedClassifier(model, f)
        print(" >> starting fit")
        clf.fit(X_train, Y_train)
        with open(checkpoint_name, "wb") as file:
            pickle.dump(clf, file)
        print(" >> starting score")
        score_split = clf.score(X_test, Y_test)
        return score_split

    scores = []
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    projected_X = np.load('../hcp900_difumo_matrics/difumo_data.npy')
    for model, name in models:
        subjects = labels["subject"].unique()
        sf = ShuffleSplit(
            n_splits=n_splits, train_size=train_size, random_state=0
        )
        model_fname = models_dir / name
        scores_split = [
            do_split(
                split=split,
                data_paths=images_path,
                projected_X=projected_X,
                Y=labels,
                f=f,
                model=model,
                subjects=subjects,
                n_jobs=n_jobs,
                checkpoint_base_name=model_fname,
                i_split=i_split,
            )
            for i_split, split in enumerate(sf.split(range(len(subjects))))
        ]
        for i_split, score_split in enumerate(scores_split):
            scores.append((method_name, name, score_split, i_split))

    scores = pd.DataFrame(
        scores, columns=["method_name", "algo", "score", "split"]
    )
    scores.to_csv(output_fname)
