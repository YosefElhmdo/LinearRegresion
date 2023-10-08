
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral

import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit

from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
)
from ..preprocessing._data import _is_constant_feature
from ..utils import check_array, check_random_state
from ..utils._array_api import get_namespace
from ..utils._seq_dataset import (
    ArrayDataset32,
    ArrayDataset64,
    CSRDataset32,
    CSRDataset64,
)
from ..utils.extmath import _incremental_mean_and_var, safe_sparse_dot
from ..utils.parallel import Parallel, delayed
from ..utils.sparsefuncs import inplace_column_scale, mean_variance_axis
from ..utils.validation import FLOAT_DTYPES, _check_sample_weight, check_is_fitted


SPARSE_INTERCEPT_DECAY = 0.01


def _deprecate_normalize(normalize, estimator_name):

    if normalize not in [True, False, "deprecated"]:
        raise ValueError(
            "Leave 'normalize' to its default value or set it to True or False"
        )

    if normalize == "deprecated":
        _normalize = False
    else:
        _normalize = normalize

    pipeline_msg = (
        "If you wish to scale the data, use Pipeline with a StandardScaler "
        "in a preprocessing stage. To reproduce the previous behavior:\n\n"
        "from sklearn.pipeline import make_pipeline\n\n"
        "model = make_pipeline(StandardScaler(with_mean=False), "
        f"{estimator_name}())\n\n"
        "If you wish to pass a sample_weight parameter, you need to pass it "
        "as a fit parameter to each step of the pipeline as follows:\n\n"
        "kwargs = {s[0] + '__sample_weight': sample_weight for s "
        "in model.steps}\n"
        "model.fit(X, y, **kwargs)\n\n"
    )

    alpha_msg = ""
    if "LassoLars" in estimator_name:
        alpha_msg = "Set parameter alpha to: original_alpha * np.sqrt(n_samples). "

    if normalize != "deprecated" and normalize:
        warnings.warn(
            "'normalize' was deprecated in version 1.2 and will be removed in 1.4.\n"
            + pipeline_msg
            + alpha_msg,
            FutureWarning,
        )
    elif not normalize:
        warnings.warn(
            (
                "'normalize' was deprecated in version 1.2 and will be "
                "removed in 1.4. "
                "Please leave the normalize parameter to its default value to "
                "silence this warning. The default behavior of this estimator "
                "is to not do any normalization. If normalization is needed "
                "please use sklearn.preprocessing.StandardScaler instead."
            ),
            FutureWarning,
        )

    return _normalize


def make_dataset(X, y, sample_weight, random_state=None):
   
    rng = check_random_state(random_state)
    # seed should never be 0 in SequentialDataset64
    seed = rng.randint(1, np.iinfo(np.int32).max)

    if X.dtype == np.float32:
        CSRData = CSRDataset32
        ArrayData = ArrayDataset32
    else:
        CSRData = CSRDataset64
        ArrayData = ArrayDataset64

    if sp.issparse(X):
        dataset = CSRData(X.data, X.indptr, X.indices, y, sample_weight, seed=seed)
        intercept_decay = SPARSE_INTERCEPT_DECAY
    else:
        X = np.ascontiguousarray(X)
        dataset = ArrayData(X, y, sample_weight, seed=seed)
        intercept_decay = 1.0

    return dataset, intercept_decay


def _preprocess_data(
    X,
    y,
    fit_intercept,
    normalize=False,
    copy=True,
    copy_y=True,
    sample_weight=None,
    check_input=True,
):
   
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    if check_input:
        X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)
        y = check_array(y, dtype=X.dtype, copy=copy_y, ensure_2d=False)
    else:
        y = y.astype(X.dtype, copy=copy_y)
        if copy:
            if sp.issparse(X):
                X = X.copy()
            else:
                X = X.copy(order="K")

    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis=0, weights=sample_weight)
        else:
            if normalize:
                X_offset, X_var, _ = _incremental_mean_and_var(
                    X,
                    last_mean=0.0,
                    last_variance=0.0,
                    last_sample_count=0.0,
                    sample_weight=sample_weight,
                )
            else:
                X_offset = np.average(X, axis=0, weights=sample_weight)

            X_offset = X_offset.astype(X.dtype, copy=False)
            X -= X_offset

        if normalize:
            X_var = X_var.astype(X.dtype, copy=False)
            
            constant_mask = _is_constant_feature(X_var, X_offset, X.shape[0])
            if sample_weight is None:
                X_var *= X.shape[0]
            else:
                X_var *= sample_weight.sum()
            X_scale = np.sqrt(X_var, out=X_var)
            X_scale[constant_mask] = 1.0
            if sp.issparse(X):
                inplace_column_scale(X, 1.0 / X_scale)
            else:
                X /= X_scale
        else:
            X_scale = np.ones(X.shape[1], dtype=X.dtype)

        y_offset = np.average(y, axis=0, weights=sample_weight)
        y -= y_offset
    else:
        X_offset = np.zeros(X.shape[1], dtype=X.dtype)
        X_scale = np.ones(X.shape[1], dtype=X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype=X.dtype)

    return X, y, X_offset, y_offset, X_scale


def _rescale_data(X, y, sample_weight, inplace=False):
    
    n_samples = X.shape[0]
    sample_weight_sqrt = np.sqrt(sample_weight)

    if sp.issparse(X) or sp.issparse(y):
        sw_matrix = sparse.dia_matrix(
            (sample_weight_sqrt, 0), shape=(n_samples, n_samples)
        )

    if sp.issparse(X):
        X = safe_sparse_dot(sw_matrix, X)
    else:
        if inplace:
            X *= sample_weight_sqrt[:, np.newaxis]
        else:
            X = X * sample_weight_sqrt[:, np.newaxis]

    if sp.issparse(y):
        y = safe_sparse_dot(sw_matrix, y)
    else:
        if inplace:
            if y.ndim == 1:
                y *= sample_weight_sqrt
            else:
                y *= sample_weight_sqrt[:, np.newaxis]
        else:
            if y.ndim == 1:
                y = y * sample_weight_sqrt
            else:
                y = y * sample_weight_sqrt[:, np.newaxis]
    return X, y, sample_weight_sqrt


class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    def predict(self, X):
        
        return self._decision_function(X)

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_"""
        if self.fit_intercept:
            # We always want coef_.dtype=X.dtype. For instance, X.dtype can differ from
            # coef_.dtype if warm_start=True.
            self.coef_ = np.divide(self.coef_, X_scale, dtype=X_scale.dtype)
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.0

    def _more_tags(self):
        return {"requires_y": True}


# XXX Should this derive from LinearModel? It should be a mixin, not an ABC.
# Maybe the n_features checking can be moved to LinearModel.
class LinearClassifierMixin(ClassifierMixin):
    
    def decision_function(self, X):
       
        check_is_fitted(self)
        xp, _ = get_namespace(X)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return xp.reshape(scores, (-1,)) if scores.shape[1] == 1 else scores

    def predict(self, X):
       
        xp, _ = get_namespace(X)
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = xp.astype(scores > 0, int)
        else:
            indices = xp.argmax(scores, axis=1)

        return xp.take(self.classes_, indices)

    def _predict_proba_lr(self, X):
        
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob


class SparseCoefMixin:
  

    def densify(self):
     
        msg = "Estimator, %(name)s, must be fitted before densifying."
        check_is_fitted(self, msg=msg)
        if sp.issparse(self.coef_):
            self.coef_ = self.coef_.toarray()
        return self

    def sparsify(self):
       
        msg = "Estimator, %(name)s, must be fitted before sparsifying."
        check_is_fitted(self, msg=msg)
        self.coef_ = sp.csr_matrix(self.coef_)
        return self


class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
   
    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "n_jobs": [None, Integral],
        "positive": ["boolean"],
    }

    def __init__(
        self,
        *,
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
      
        n_jobs_ = self.n_jobs

        accept_sparse = False if self.positive else ["csr", "csc", "coo"]

        X, y = self._validate_data(
            X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
        )

        has_sw = sample_weight is not None
        if has_sw:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=X.dtype, only_non_negative=True
            )

        copy_X_in_preprocess_data = self.copy_X and not sp.issparse(X)

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=copy_X_in_preprocess_data,
            sample_weight=sample_weight,
        )

        if has_sw:
            
            X, y, sample_weight_sqrt = _rescale_data(
                X, y, sample_weight, inplace=copy_X_in_preprocess_data
            )

        if self.positive:
            if y.ndim < 2:
                self.coef_ = optimize.nnls(X, y)[0]
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(optimize.nnls)(X, y[:, j]) for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
        elif sp.issparse(X):
            X_offset_scale = X_offset / X_scale

            if has_sw:

                def matvec(b):
                    return X.dot(b) - sample_weight_sqrt * b.dot(X_offset_scale)

                def rmatvec(b):
                    return X.T.dot(b) - X_offset_scale * b.dot(sample_weight_sqrt)

            else:

                def matvec(b):
                    return X.dot(b) - b.dot(X_offset_scale)

                def rmatvec(b):
                    return X.T.dot(b) - X_offset_scale * b.sum()

            X_centered = sparse.linalg.LinearOperator(
                shape=X.shape, matvec=matvec, rmatvec=rmatvec
            )

            if y.ndim < 2:
                self.coef_ = lsqr(X_centered, y)[0]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(lsqr)(X_centered, y[:, j].ravel())
                    for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
        else:
            self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


def _check_precomputed_gram_matrix(
    X, precompute, X_offset, X_scale, rtol=None, atol=1e-5
):
    

    n_features = X.shape[1]
    f1 = n_features // 2
    f2 = min(f1 + 1, n_features - 1)

    v1 = (X[:, f1] - X_offset[f1]) * X_scale[f1]
    v2 = (X[:, f2] - X_offset[f2]) * X_scale[f2]

    expected = np.dot(v1, v2)
    actual = precompute[f1, f2]

    dtypes = [precompute.dtype, expected.dtype]
    if rtol is None:
        rtols = [1e-4 if dtype == np.float32 else 1e-7 for dtype in dtypes]
        rtol = max(rtols)

    if not np.isclose(expected, actual, rtol=rtol, atol=atol):
        raise ValueError(
            "Gram matrix passed in via 'precompute' parameter "
            "did not pass validation when a single element was "
            "checked - please check that it was computed "
            f"properly. For element ({f1},{f2}) we computed "
            f"{expected} but the user-supplied value was "
            f"{actual}."
        )


def _pre_fit(
    X,
    y,
    Xy,
    precompute,
    normalize,
    fit_intercept,
    copy,
    check_input=True,
    sample_weight=None,
):
    
    n_samples, n_features = X.shape

    if sparse.isspmatrix(X):
        # copy is not needed here as X is not modified inplace when X is sparse
        precompute = False
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy=False,
            check_input=check_input,
            sample_weight=sample_weight,
        )
    else:
        # copy was done in fit if necessary
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy=copy,
            check_input=check_input,
            sample_weight=sample_weight,
        )
        # Rescale only in dense case. Sparse cd solver directly deals with
        # sample_weight.
        if sample_weight is not None:
            # This triggers copies anyway.
            X, y, _ = _rescale_data(X, y, sample_weight=sample_weight)

    # FIXME: 'normalize' to be removed in 1.4
    if hasattr(precompute, "__array__"):
        if (
            fit_intercept
            and not np.allclose(X_offset, np.zeros(n_features))
            or normalize
            and not np.allclose(X_scale, np.ones(n_features))
        ):
            warnings.warn(
                (
                    "Gram matrix was provided but X was centered to fit "
                    "intercept, or X was normalized : recomputing Gram matrix."
                ),
                UserWarning,
            )
            # recompute Gram
            precompute = "auto"
            Xy = None
        elif check_input:
  
            _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale)

    # precompute if n_samples > n_features
    if isinstance(precompute, str) and precompute == "auto":
        precompute = n_samples > n_features

    if precompute is True:
        # make sure that the 'precompute' array is contiguous.
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype, order="C")
        np.dot(X.T, X, out=precompute)

    if not hasattr(precompute, "__array__"):
        Xy = None  # cannot use Xy if precompute is not Gram

    if hasattr(precompute, "__array__") and Xy is None:
        common_dtype = np.result_type(X.dtype, y.dtype)
        if y.ndim == 1:
            # Xy is 1d, make sure it is contiguous.
            Xy = np.empty(shape=n_features, dtype=common_dtype, order="C")
            np.dot(X.T, y, out=Xy)
        else:

            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype, order="F")
            np.dot(y.T, X, out=Xy.T)

    return X, y, X_offset, y_offset, X_scale, precompute, Xy

