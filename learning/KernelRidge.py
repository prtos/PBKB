import numpy as np
import PBKB.inference.BB.preimage as bb
import PBKB.inference.graph_based.preimage as gb
from PBKB.kernels.gskernel import GSKernel
from sklearn.model_selection._search import check_is_fitted
from sklearn.model_selection import KFold
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model.ridge import _solve_cholesky_kernel



class KernelRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1, amino_acid_file="", sigma_position=0.01,
                 sigma_amino_acid=0.01, substring_length=1, is_normalized=True,
                 center_kernel=False, verbose=1):
        super(KernelRidge, self).__init__()
        self.alpha = alpha
        self.amino_acid_file = amino_acid_file
        self.sigma_position = sigma_position
        self.sigma_amino_acid = sigma_amino_acid
        self.substring_length = substring_length
        self.is_normalized = is_normalized
        self.center_kernel = center_kernel
        self.verbose = verbose

    def _get_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        return self.kernel(X, Y)

    def fit(self, X, y=None, sample_weight=None):
        self.kernel = GSKernel(amino_acid_file_name=self.amino_acid_file,
                               sigma_position=self.sigma_position,
                               sigma_amino_acid=self.sigma_amino_acid,
                               n=self.substring_length,
                               is_normalized=self.is_normalized)
        K = self._get_kernel(X)
        if self.center_kernel:
            self.compute_train_stats(K, y)
            K = self.kernel_centering(K)
            y = (y - self.y_mean)

        alpha = np.atleast_1d(self.alpha)

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        copy = False
        self.dual_coef_ = _solve_cholesky_kernel(K, y, alpha, sample_weight, copy)
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X

        return self

    def predict(self, X):
        """Predict using the kernel ridge model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self, ["X_fit_", "dual_coef_"])
        K = self._get_kernel(X, self.X_fit_)
        alpha0 = 0
        if self.center_kernel:
            K = self.kernel_centering(K)
            alpha0 = self.y_mean
        return np.dot(K, self.dual_coef_) + alpha0

    def kernel_centering(self, K):
        mean_over_training = np.mean(K, axis=1)
        temp = np.add.outer(mean_over_training, self.col_mean_training)
        return (K - temp + self.mu_norm2) / self.sigma_square

    def compute_train_stats(self, K, y):
        self.mu_norm2 = np.mean(K)*1.0
        self.sigma_square = np.mean(np.diagonal(K)) - self.mu_norm2
        self.col_mean_training = np.mean(K, axis=1)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        assert self.sigma_square >= 0

    def inference(self, n_predictions, prediction_length, max_time=3600):
        if self.kernel.is_normalized:
            optimizer = bb.Preimage(self.kernel.alphabet, self.kernel.n, self.kernel, max_time=max_time)
            optimizer.fit(self.X_fit_, self.dual_coef_, prediction_length)
            peptides, bioactivities = optimizer.predict(n_predictions)
            if self.verbose > 0:
                print('Learner HP')
                print dict(alpha=self.alpha, sigma_position=self.kernel.sigma_position,
                           sigma_amino_acid=self.kernel.sigma_amino_acid,
                           substring_length=self.kernel.n)
                print('Statistics')
                stats = optimizer.get_stats()
                print('Number of training examples', stats['n_examples_train'])
                print('Train time', stats['train_time'])
                print('Prediction time', stats['test_time'])
                print('Number of iterations', stats['search_mean_n_iterations'])
                print('Branch and bound search time', stats['search_mean_time'])
                print('Is optimal (B&B finished before allowed time or max number of iterations)',
                      stats['search_percentage_of_approximates'] == 0)

        else:
            optimizer = gb.Preimage(prediction_length, self.X_fit_, self.dual_coef_, self.kernel.amino_acid_file_name,
                                    self.kernel.sigma_position, self.kernel.sigma_amino_acid, self.kernel.n)
            peptides, bioactivities = optimizer.k_longest_path(n_predictions)
        return peptides, bioactivities


if __name__ == "__main__":
    from sklearn.model_selection import GridSearchCV
    from PBKB.utils.loader import aa_file, aa_file_oboc, dataset_dir, from_pwd
    from PBKB.utils.graphics import heatmap
    x_train = np.array(["AAAAA", "WSWSW", "SWSWS"])
    y_train = np.array([1.2, 40, 54])
    x_test = np.array(["AWWSA"])

    rgr = KernelRidge(alpha=1,
                      amino_acid_file=aa_file,
                      sigma_position=3,
                      sigma_amino_acid=3,
                      substring_length=1,
                      is_normalized=False)
    rgr.fit(x_train, y_train)
    print rgr.predict(x_test)
    print rgr.inference(10, 5)

    ################################################################################################
    filename = from_pwd(dataset_dir, "data.txt")
    with open(filename, "r") as f:
        temp = [line[:-1].split("\t") for line in f.readlines()]
        inputs, targets = zip(*temp)
        targets = map(float, targets)
        inputs, targets = np.array(inputs), np.array(targets)

    v, n = 2, 4
    cv_params = dict(sigma_position=np.logspace(-v, v, n),
                     sigma_amino_acid=np.logspace(-v, v, n),
                     substring_length=np.arange(1, 3, 1),
                     alpha=np.logspace(-4, 4, n),
                     is_normalized=[True])
    base_learner = KernelRidge(amino_acid_file=aa_file_oboc)
    inner_cv = KFold(n_splits=3, shuffle=False)
    outer_cv = KFold(n_splits=3, shuffle=False)  # LeaveOneOut()
    targets_pred, targets_true = [], []
    str_format = "{:^30} | {}"
    all_best_hp = []
    fold = 1
    for otrain_index, otest_index in outer_cv.split(inputs):
        inputs_train, inputs_test = inputs[otrain_index], inputs[otest_index]
        targets_train, targets_test = targets[otrain_index], targets[otest_index]

        rgr = GridSearchCV(base_learner, cv_params, n_jobs=8, cv=inner_cv, verbose=1)
        # rgr = RandomizedSearchCV(base_learner, cv_params, n_iter=1000, n_jobs=8, cv=inner_cv, verbose=1)
        rgr.fit(inputs_train, targets_train)
        targets_pred += rgr.predict(inputs_test).tolist()
        targets_true += targets_test.tolist()

        print str_format.format("Best CV parameters", rgr.best_params_)
        print str_format.format("Best CV MSE", rgr.best_score_)
        all_best_hp.append(rgr.best_params_)
        heatmap(rgr, prefix_filename='hmap_gridsearch',
                params_of_interest=["sigma_position", "sigma_amino_acid", "alpha"])

        fold += 1
        if fold:
            break

