from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.base import is_classifier, clone
from sklearn.model_selection._search import _fit_and_score, Parallel, delayed, check_cv, check_scoring, indexable, \
    Sized, partial, rankdata, ParameterGrid
from sklearn.model_selection._validation import logger, _index_param_value, _score, FitFailedWarning, _num_samples
import time, numbers, warnings, sys
import numpy as np
from collections import defaultdict


def _fit_and_score2(estimator, X_train, y_train, X_test, y_test, scorer, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise'):
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                    for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X_train, v, np.arange(len(X_train))))
                       for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    # kernel_before = estimator.kernel
    estimator.kernel = "precomputed"
    start_time = time.time()

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_score = _score(estimator, X_test, y_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg, logger.short_format_time(score_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score, test_score] if return_train_score else [test_score]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret


def compute_gram_matrix(base_estimator, X, p):
    clss = clone(base_estimator)
    clss.set_params(**p)
    params = str({k: p[k] for k in sorted(p)})
    return params, clss._get_kernel(X)


class GridSearchCV(GSCV):

    def _fit(self, X, y, groups, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        if self.verbose > 0 and isinstance(parameter_iterable, Sized):
            n_candidates = len(parameter_iterable)
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        if hasattr(base_estimator, "_get_kernel") and base_estimator.kernel != "precomputed" and \
            base_estimator.kernel == "gs_kernel":
            n_jobs = self.n_jobs
            verbose = self.verbose
            reduced_param_set = list(parameter_iterable)
            for x in reduced_param_set:
                if "alpha" in x:
                    del x["alpha"]
            reduced_param_set = list({str({k: p[k] for k in sorted(p)}): p for p in reduced_param_set}.values())

            grams = Parallel(
                    n_jobs=n_jobs, verbose=verbose,
                    pre_dispatch=pre_dispatch
                )(delayed(compute_gram_matrix)(base_estimator, X, p) for p in reduced_param_set)
            grams = dict(grams)

            def get_grams():
                for p in parameter_iterable:
                    x = p.copy()
                    if "alpha" in x:
                        del x["alpha"]
                    key = str({k: x[k] for k in sorted(x)})
                    yield grams[key], p

            out = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch
            )(delayed(_fit_and_score2)(clone(base_estimator), K[train][:, train], y[train],
                                       K[test][:, train], y[test],
                                       self.scorer_, self.verbose, parameters,
                                       fit_params=self.fit_params,
                                       return_train_score=self.return_train_score,
                                       return_n_test_samples=True,
                                       return_times=True, return_parameters=True,
                                       error_score=self.error_score)
              for K, parameters in get_grams()
              for train, test in cv.split(K, y, groups))
        elif hasattr(base_estimator, "_get_kernel") and base_estimator.kernel != "precomputed":
            def get_gram_matrix(param_iter):
                for p in param_iter:
                    clss = clone(base_estimator)
                    clss.set_params(**p)
                    yield clss._get_kernel(X), p

            out = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch
            )(delayed(_fit_and_score2)(clone(base_estimator), K[train][:, train], y[train],
                                       K[test][:, train], y[test],
                                       self.scorer_, self.verbose, parameters,
                                       fit_params=self.fit_params,
                                       return_train_score=self.return_train_score,
                                       return_n_test_samples=True,
                                       return_times=True, return_parameters=True,
                                       error_score=self.error_score)
              for K, parameters in get_gram_matrix(parameter_iterable)
              for train, test in cv.split(K, y, groups))
        else:
            out = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch
            )(delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                      train, test, self.verbose, parameters,
                                      fit_params=self.fit_params,
                                      return_train_score=self.return_train_score,
                                      return_n_test_samples=True,
                                      return_times=True, return_parameters=True,
                                      error_score=self.error_score)
              for parameters in parameter_iterable
              for train, test in cv.split(X, y, groups))

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        _store('test_score', test_scores, splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        _store('train_score', train_scores, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one np.MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(np.ma.masked_all, (n_candidates,),
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


    def fit(self, X, y=None, groups=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        """
        return self._fit(X, y, groups, ParameterGrid(self.param_grid))
