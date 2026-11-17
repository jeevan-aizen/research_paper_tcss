"""TF-IDF features and classical classifiers. The vectoriser is always fitted
on the training split only."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from . import config as C


def fit_tfidf(train_texts):
    vec = TfidfVectorizer(**C.TFIDF)
    X = vec.fit_transform(train_texts)
    return vec, X


def make_model(name: str, seed: int, **overrides):
    """Model factory. '_bal' variants use inverse-frequency class weights
    w_c = n / (K * n_c); for MNB the analogue is a uniform class prior."""
    if name == "mnb":
        return MultinomialNB(alpha=overrides.get("alpha", C.MNB_ALPHA))
    if name == "mnb_bal":
        return MultinomialNB(alpha=overrides.get("alpha", C.MNB_ALPHA), fit_prior=False)
    if name in ("lr", "lr_bal"):
        return LogisticRegression(C=overrides.get("C", C.LR_C), max_iter=3000,
                                  class_weight="balanced" if name == "lr_bal" else None,
                                  random_state=seed)
    if name in ("svm", "svm_bal"):
        return SoftmaxSVC(C=overrides.get("C", C.SVM_C),
                          class_weight="balanced" if name == "svm_bal" else None,
                          random_state=seed)
    raise ValueError(name)


class SoftmaxSVC(LinearSVC):
    """LinearSVC whose scores are turned into probabilities by a softmax over
    the decision function. The predicted class is therefore the SVM's own
    decision. (Platt scaling with CalibratedClassifierCV re-fits per-class
    sigmoids on imbalanced folds, which moved predictions back toward the
    majority class and cancelled the effect of class weighting.) The
    probabilities are calibrated afterwards by temperature scaling on the
    validation split, like every other model."""

    def predict_proba(self, X):
        z = self.decision_function(X)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


CLASSICAL_MODELS = ["mnb", "mnb_bal", "lr", "lr_bal", "svm", "svm_bal"]


def predict_proba_3(model, X) -> np.ndarray:
    """Return (n, 3) probabilities in label order 0,1,2 even if a class was
    absent from training."""
    p = model.predict_proba(X)
    out = np.zeros((X.shape[0], 3))
    out[:, model.classes_] = p
    return out
