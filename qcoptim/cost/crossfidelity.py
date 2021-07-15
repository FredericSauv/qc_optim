"""
Cross-fidelity cost classes and functions
"""

import warnings
import functools

import numpy as np
import scipy as sp

# from qiskit import QiskitError
from qiskit.result import Result

from ..utilities import (
    FastCountsResult,
    bootstrap_resample,
    RandomMeasurementHandler,
)

from .core import CostInterface


class CrossFidelity(CostInterface):
    """
    Cost class to implement offline CrossFidelity measurements between two
    quantum states (arxiv:1909.01282).
    """
    def __init__(
        self,
        ansatz,
        instance=None,
        nb_random=None,
        seed=None,
        mode='1qHaar',
        comparison_results=None,
        num_bootstraps=1000,
        prefixA='CrossFid',
        prefixB='CrossFid',
        transpiler='instance',
        rand_meas_handler=None,
    ):
        """
        Parameters
        ----------
        ansatz : object implementing AnsatzInterface
            The ansatz object that this cost can be optimsed over
        instance : qiskit.utils.QuantumInstance
            Will be used to generate internal transpiled circuits
        nb_random : int, optional
            The number of random unitaries to average over
        seed : int, optional
            Seed used to generate random unitaries
        mode : str, optional
            How to generate the random measurements, passed to 
            RandomMeasurementHandler constructor. Supported options:
                'identity' : trivial case, do nothing
                '1qHaar' : single qubit Haar random unitaries
                'rypiOver3' : 1/3 of qubits are acted on by identities, 1/3 by
                              Ry(pi/3), and 1/3 by Ry(2pi/3)
                'RzRy' : single qubit Haar random unitaries, generated from
                         selecting euler angles using numpy random functions
                         instead of qiskit random unitary function
        comparison_results : {dict, None, qiskit.results.Result}
            The use cases where None would be passed is if we are using
            this object to generate the comparison_results object for a
            future instance of CrossFidelity. This robustly ensures that
            the results objs to compare are compatible.
            If dict is passed it should be a qiskit results object that
            has been converted to a dict using its `to_dict` method.
            Ideally this would have been tagged with CrossFidelity
            metadata using this classes `tag_results_metadata` method.
            Can also accept a qiskit results obj
        num_bootstraps : int, optional
            Number of bootstrap resamples to use to estimate the standard error
            w/r/t/ the random unitaries
        prefixA : str, optional
            Prefix string to use on circuits generate to characterise A system.
        prefixB : str, optional
            Prefix string to use to extract system B's results from
            comparison_results.
        transpiler : str, optional
            Choose how to transpile circuits, current options are:
                'instance' : use quantum instance
                'pytket' : use pytket compiler
        rand_meas_handler : None, optional
            Can pass an already initialised RandomMeasurementHandler obj to use
            to generate random basis circuits internally. This can be shared
            with other users to avoid repeated random characterisation of the
            same state.

            Will raise ValueError if rand_meas_handler's ansatz or instance are
            different from the args, unless `ansatz=None` and `instance=None`.

            `rand_meas_handler.num_random` can be different from nb_random as
            long as nb_random is smaller.
        """
        self._prefixB = prefixB
        self._num_bootstraps = num_bootstraps

        # set default for nb_random if passed as None
        if nb_random is None:
            if rand_meas_handler is None:
                nb_random = 5
            else:
                nb_random = rand_meas_handler.num_random
        elif (not isinstance(nb_random, int)) or (nb_random < 1):
            raise ValueError('nb_random is invalid.')

        # make internal RandomMeasurementHandler in none passed
        def circ_name(idx):
            return prefixA + f'{idx}'
        if rand_meas_handler is None:
            self._rand_meas_handler = RandomMeasurementHandler(
                ansatz,
                instance,
                nb_random,
                seed=seed,
                circ_name=circ_name,
                transpiler=transpiler,
                mode=mode,
            )
        else:
            if ansatz is not None and ansatz != rand_meas_handler.ansatz:
                raise ValueError('Ansatz passed different from'
                                 + ' rand_meas_handler obj.')
            if instance is not None and instance != rand_meas_handler.instance:
                raise ValueError('Quantum instance passed different from'
                                 + ' rand_meas_handler obj.')
            if nb_random > rand_meas_handler.num_random:
                raise ValueError('nb_random larger than num_random of'
                                 + ' rand_meas_handler obj.')
            self._rand_meas_handler = rand_meas_handler
        self.nb_random = nb_random

        # run setter (see below)
        self.comparison_results = comparison_results

        # related to last evaluation
        self.last_evaluation = None

    @property
    def seed(self):
        return self._rand_meas_handler.seed

    @property
    def ansatz(self):
        return self._rand_meas_handler.ansatz

    @property
    def instance(self):
        return self._rand_meas_handler.instance

    @property
    def random_unitary_mode(self):
        return self._rand_meas_handler.mode

    @property
    def comparison_results(self):
        return self._comparison_results

    @property
    def meas_circuits(self):
        """
        Returns list of measurement circuits needed to evaluate the cost function
        """
        circs = self._rand_meas_handler._meas_circuits
        return circs

    @comparison_results.setter
    def comparison_results(self, results):
        """
        Setter for comparison_results, perform validations
        """
        if results is not None:

            if (
                isinstance(results, dict)
                and 'cross-fidelity' in results.keys()
            ):

                # compare metadata
                comparison_metadata = results['crossfidelity_metadata']
                if self.seed != comparison_metadata['seed']:
                    raise ValueError(
                        'Comparison results use different random seed.'
                    )
                if self.nb_random > comparison_metadata['nb_random']:
                    raise ValueError(
                        'Comparison results use fewer random basis.'
                    )
                if self._prefixB != comparison_metadata['prefixA']:
                    raise ValueError(
                        'Passed prefixB does not match prefixA of comparison'
                        + ' results.'
                    )

                # bug fix, need counts dict keys to be hex values
                for val in results['results']:
                    new_counts = {}
                    for ckey, cval in val['data']['counts'].items():
                        # detect it is not hex
                        if not ckey[:2] == '0x':
                            ckey = hex(int(ckey, 2))
                        new_counts[ckey] = cval
                    val['data']['counts'] = new_counts

            else:
                warnings.warn(
                    'Warning, input results is not a dictionary containing'
                    + ' crossfidelity_metadata and so we cannot confirm that'
                    + ' the results are compatible. If the input results'
                    + ' object was collecting by this class consider using'
                    + ' the tag_results_metadata method to add the'
                    + ' crossfidelity_metadata.'
                )

            if isinstance(results, Result):
                self._comparison_results = results
            elif isinstance(results, dict):
                self._comparison_results = FastCountsResult(
                    Result.from_dict(results))
            else:
                raise TypeError(
                    'comparison_results recieved type '+f'{type(results)}'
                )

    # def tag_results_metadata(self, results):
    #     """
    #     Adds in CrossFidelity metadata to a results object. This can be
    #     used to ensure that two results sets are compatible.

    #     Parameters
    #     ----------
    #     results : qiskit.result.Result, dict
    #         The results data to process

    #     Returns
    #     -------
    #     results : dict
    #         Results dictionary with the CrossFidelity metadata added
    #     """

    #     # convert results to dict if needed
    #     if not isinstance(results, dict):
    #         results = results.to_dict()
    #     # add CrossFidelity metadata
    #     results.update({
    #         'crossfidelity_metadata': {
    #             'seed': self.seed,
    #             'nb_random': self.nb_random,
    #             'prefixA': self._rand_meas_handler.circ_name(''),
    #             }
    #         })
    #     return results

    def bind_params_to_meas(self, params=None, params_names=None):
        """
        Return measurement circuits bound at `params`

        Parameters
        ----------
        params : numpy.ndarray, optional
            Point (1d) or points (2d) to bind circuits at
        params_names : None, optional
            Description

        Returns
        -------
        list[qiskit.QuantumCircuit]
            Evaluation circuits
        """
        if params is None:
            raise ValueError('Bound circuits requested without params given.')

        params = np.atleast_2d(params)
        if isinstance(params_names, str):
            params_names = [params_names]
        if params_names is None:
            params_names = [None] * len(params)
        elif not len(params_names) == len(params):
            raise ValueError(
                'params_names passed has different lengh to params.'
            )

        # get circuits from rand_meas_handler
        return self._rand_meas_handler.circuits(params)

    def evaluate_cost(
        self,
        results,
        name='',
        vectorise=False,
        **kwargs
    ):
        """
        Calculates the cross-fidelity using two sets of qiskit results.
        The variable names are chosen to match arxiv:1909.01282 as close
        as possible.

        Parameters
        ----------
        results : qiskit.result.Result
            Results to calculate cross-fidelity with, against the stored
            results dictionary.
        name : str, optional
            Prefix on results names to find results data
        **kwargs
            Description

        Returns
        -------
        float
            Evaluated cross-fidelity
        """
        return self.evaluate_cost_and_std(
            results, name=name, vectorise=vectorise, **kwargs)[0]

    def evaluate_cost_and_std(
        self,
        results,
        name='',
        vectorise=False,
        **kwargs
    ):
        """
        Calculates the cross-fidelity using two sets of qiskit results.
        The variable names are chosen to match arxiv:1909.01282 as close
        as possible.

        Parameters
        ----------
        results : Qiskit results type
            Results to calculate cross-fidelity with, against the stored
            results dictionary.
        name : str, optional
            Prefix on results names to find results data
        **kwargs
            Description

        Returns
        -------
        mean : float
            Evaluated cross-fidelity
        std : float
            Standard error on cross-fidelity estimation, obtained from
            bootstrap resampling
        """

        # we make it possible to instance a CrossFidelity obj without a
        # comparison_results dict so that we can easily generate the
        # comparison data using the same setup (e.g. seed, prefix). But
        # in that case cannote evaluate the cost.
        if self._comparison_results is None:
            raise ValueError('No comparison results set has been passed to'
                             + ' CrossFidelity obj.')

        def circ_namesA(idx):
            return name + self._rand_meas_handler.circ_name(idx)

        def circ_namesB(idx):
            return self._prefixB + f'{idx}'

        if self._num_bootstraps > 0:

            # case using bootstrapping to estimate standard error from dist 
            # of U's

            dist_tr_rhoA_rhoB = _crosscorrelation_per_u(
                results, self._comparison_results, self.nb_random,
                circ_namesA=circ_namesA, circ_namesB=circ_namesB,
                vectorise=vectorise,
            )
            dist_tr_rhoA_2 = _purity_per_u(
                results, self.nb_random, names=circ_namesA, vectorise=vectorise)
            dist_tr_rhoB_2 = _purity_per_u(
                self._comparison_results, self.nb_random, names=circ_namesB, 
                vectorise=vectorise)

            # bootstrap resample for means and std-errs
            tr_rhoA_rhoB, tr_rhoA_rhoB_err = bootstrap_resample(
                np.mean, dist_tr_rhoA_rhoB, self._num_bootstraps,
                random_seed=self.seed,
            )
            tr_rhoA_2, tr_rhoA_2_err = bootstrap_resample(
                np.mean, dist_tr_rhoA_2, self._num_bootstraps,
                random_seed=self.seed,
            )
            tr_rhoB_2, tr_rhoB_2_err = bootstrap_resample(
                np.mean, dist_tr_rhoB_2, self._num_bootstraps,
                random_seed=self.seed,
            )

        else:

            # case without bootstrapping, this only includes standard error
            # arising from counts spread (currently not even that)

            if isinstance(results, FastCountsResult):
                tmp_results = results
            elif isinstance(results, Result):
                tmp_results = FastCountsResult(results)
            else:
                raise TypeError(
                    'results type not recognised: '+f'{type(results)}')
            tmp_results.combine_counts([
                circ_namesA(idx) for idx in range(self.nb_random)],
                circ_namesA('')
            )

            if isinstance(self._comparison_results, FastCountsResult):
                tmp_comparison_results = self._comparison_results
            elif isinstance(self._comparison_results, Result):
                tmp_comparison_results = FastCountsResult(
                    self._comparison_results)
            else:
                raise TypeError(
                    'results type not recognised: '
                    + f'{type(self._comparison_results)}')
            tmp_comparison_results.combine_counts([
                circ_namesB(idx) for idx in range(self.nb_random)],
                circ_namesB('')
            )

            def tmp_circ_namesA(idx):
                return circ_namesA('')

            def tmp_circ_namesB(idx):
                return circ_namesB('')

            tr_rhoA_rhoB = _crosscorrelation_per_u(
                tmp_results, tmp_comparison_results, 1,
                circ_namesA=tmp_circ_namesA, circ_namesB=tmp_circ_namesB,
                vectorise=vectorise,
            )[0]
            tr_rhoA_2 = _purity_per_u(
                tmp_results, 1, names=tmp_circ_namesA, vectorise=vectorise)[0]
            tr_rhoB_2 = _purity_per_u(
                tmp_comparison_results, 1, names=tmp_circ_namesB, 
                vectorise=vectorise)[0]

            tr_rhoA_rhoB_err, tr_rhoA_2_err, tr_rhoB_2_err = 0, 0, 0

        # divide by largest
        if tr_rhoA_2 > tr_rhoB_2:
            mean = tr_rhoA_rhoB / tr_rhoA_2
            std = (mean**2) * (
                (tr_rhoA_rhoB_err**2) / (tr_rhoA_rhoB)**2
                + (tr_rhoA_2_err)**2 / (tr_rhoA_2)**2
            )
        else:
            mean = tr_rhoA_rhoB / tr_rhoB_2
            std = (mean**2) * (
                (tr_rhoA_rhoB_err**2) / (tr_rhoA_rhoB)**2
                + (tr_rhoB_2_err)**2 / (tr_rhoB_2)**2
            )

        # store results of evaluation
        self.last_evaluation = {
            'tr_rhoA_rhoB': tr_rhoA_rhoB,
            'tr_rhoA_rhoB_err': tr_rhoA_rhoB_err,
            'tr_rhoA_2': tr_rhoA_2,
            'tr_rhoA_2_err': tr_rhoA_2_err,
            'tr_rhoB_2': tr_rhoB_2,
            'tr_rhoB_2_err': tr_rhoB_2_err,
            'mean': mean,
            'std': std,
        }

        return mean, std


def _load_experiment_single_u(
    experiment_idx, results, nb_qubits, counts_transform=None
):
    """
    Parameters
    ----------
    experiment_idx : int
        Index of experiment to access
    results : qiskit.result.Result OR FastCountsResult
        Results object
    nb_qubits : int, or None
        If not None, will raise an error if the number of qubits found in the
        experiment is not equal to passed value
    counts_transform : callable, optional
        Function to be called on counts dict e.g. downsampling, or
        measurement error mitigation

    Returns
    -------
    bin_counts_keys : list[str]
        Keys of the counts, given as binary strings
    int_counts_keys : list[int]
        Keys of the counts, given as ints
    counts : numpy.ndarray
        Numpy array with the measurement counts
    nb_qubits : int
        Number of qubits measured in experiment
    """

    counts_dict = results.get_counts(experiment_idx)
    if counts_transform is not None:
        counts_dict = counts_transform(counts_dict)
    bin_counts_keys = list(counts_dict.keys())
    num_qubits = len(bin_counts_keys[0])
    counts = np.array(list(counts_dict.values()))

    if isinstance(results, FastCountsResult):
        int_counts_keys = results.int_keys[experiment_idx]

        # quick consistency check
        if not int_counts_keys[0] == int(bin_counts_keys[0], 2):
            raise ValueError(
                'something has gone wrong with loading processed results!'
            )

    else:
        int_counts_keys = np.array(
            [int(binval, 2) for binval in bin_counts_keys]
        )

    # use this to check number of qubits has been consistent
    # over all random unitaries
    if nb_qubits is not None and nb_qubits != num_qubits:
        raise ValueError(
            'stored nb_qubits='+f'{nb_qubits}' + ', new num_qubits='
            + f'{num_qubits}'
        )

    return bin_counts_keys, int_counts_keys, counts, num_qubits


def _get_results_access_map(results):
    """ """
    if isinstance(results, FastCountsResult):
        return results.results_access_map
    elif isinstance(results, Result):
        # make results access maps for speed
        return {
            res.header.name: idx for idx, res in enumerate(results.results)
        }

    raise TypeError('Type of results not recognised: '+f'{type(results)}')


def _purity_per_u(
    results,
    nb_random,
    names=str,
    vectorise=False,
    counts_transform=None,
):
    """
    Extract the contributions towards the evaluation of the purity of a quantum
    state using random single qubit measurements (arxiv:1909.01282), resolved
    by each random unitary.

    Parameters
    ----------
    results : qiskit.result.Result
        Results to estimate purity
    nb_random : int
        Number of random basis used
    names : Callable, optional
        Function that maps index of a random circuit to a name of a circuit in
        the qiskit results object
    vectorise : boolean, optional
        If True, will vectorise internal calculations. WARNING: this will
        generate exponentially large matrices in number of qubits.
    counts_transform : callable, optional
        Function to be called on counts dict e.g. downsampling, or
        measurement error mitigation

    Returns
    -------
    tr_rho_2 : numpy.ndarray
        Contributions towards random measurement purity estimate, resolved by
        each single random measurement
    """
    nb_qubits = None
    results_access_map = _get_results_access_map(results)

    # iterate over the different random unitaries
    tr_rho_2 = np.zeros(nb_random)
    for uidx in range(nb_random):

        try:
            experiment_idx = results_access_map[names(uidx)]
        except KeyError as missing_exp:
            raise ValueError('Cannot extract matching experiment data to'
                             + ' calculate cross-fidelity.') from missing_exp

        (bin_counts_keys,
         int_counts_keys,
         counts,
         nb_qubits) = _load_experiment_single_u(
            experiment_idx, results, nb_qubits,
            counts_transform=counts_transform)

        if vectorise:
            auto_func = _vectorised_auto_cross_correlation_single_u
        else:
            auto_func = _auto_cross_correlation_single_u

        tr_rho_2[uidx] = auto_func(bin_counts_keys, int_counts_keys, counts,)

    # normalisation
    tr_rho_2 = (2**nb_qubits)*tr_rho_2

    return tr_rho_2


def _crosscorrelation_per_u(
    resultsA,
    resultsB,
    nb_random,
    circ_namesA=None,
    circ_namesB=None,
    vectorise=False,
    counts_transform=None,
):
    """
    Function to calculate the offline CrossFidelity between two quantum states
    (arxiv:1909.01282).

    Parameters
    ----------
    resultsA : qiskit.result.Result
        Results for system A
    resultsB : qiskit.result.Result
        Results for system B
    nb_random : int
        Number of random measurement basis
    circ_namesA : callable, optional
        Naming function for A results, `int -> str`
    circ_namesB : callable, optional
        Naming function for B results, `int -> str`
    vectorise : boolean, optional
        If True, will vectorise internal calculations. WARNING: this will
        generate exponentially large matrices in number of qubits.
    counts_transform : callable, optional
        Function to be called on counts dict e.g. downsampling, or
        measurement error mitigation

    Returns
    -------
    float
        tr_rhoA_rhoB, unnormalised cross-fidelity
    float
        tr_rhoA_2, purity for system A
    float
        tr_rhoB_2, purity for system B
    """
    nb_qubits = None

    # default circuit naming functions
    if circ_namesA is None:
        def circ_namesA(idx):
            return 'CrossFid' + f'{idx}'
    if circ_namesB is None:
        def circ_namesB(idx):
            return 'CrossFid' + f'{idx}'

    resultsA_access_map = _get_results_access_map(resultsA)
    resultsB_access_map = _get_results_access_map(resultsB)

    # iterate over the different random unitaries
    tr_rhoA_rhoB = np.zeros(nb_random)
    for uidx in range(nb_random):

        try:
            experiment_idx_A = resultsA_access_map[circ_namesA(uidx)]
            experiment_idx_B = resultsB_access_map[circ_namesB(uidx)]
        except KeyError as missing_exp:
            raise ValueError('Cannot extract matching experiment data to'
                             + ' calculate cross-fidelity.') from missing_exp

        (P_A_binstrings,
         P_A_ints,
         P_A_counts,
         nb_qubits) = _load_experiment_single_u(
            experiment_idx_A, resultsA, nb_qubits,
            counts_transform=counts_transform)
        (P_B_binstrings,
         P_B_ints,
         P_B_counts,
         nb_qubits) = _load_experiment_single_u(
            experiment_idx_B, resultsB, nb_qubits,
            counts_transform=counts_transform)

        if vectorise:
            cross_func = _vectorised_cross_correlation_single_u
        else:
            cross_func = _cross_correlation_single_u

        tr_rhoA_rhoB[uidx] = cross_func(
            P_A_binstrings, P_A_ints, P_A_counts,
            P_B_binstrings, P_B_ints, P_B_counts,
        )

    # normalisations
    tr_rhoA_rhoB = (2**nb_qubits)*tr_rhoA_rhoB

    return tr_rhoA_rhoB


def _cross_correlation_single_u(
    P_1_strings,
    P_1_ints,
    P_1_counts,
    P_2_strings,
    P_2_ints,
    P_2_counts,
):
    """
    Carries out the inner loop calculation of the Cross-Fidelity. In
    contrast to the paper, arxiv:1909.01282, it makes sense for us to
    make the sum over sA and sA' the inner loop. So this computes the
    sum over sA and sA' for fixed random U.

    Parameters
    ----------
    P_1_strings : tuple[str]
        List of measurement strings in little-endian binary for distribution 1
    P_1_ints : tuple[int]
        List of int repr of measurement strings on qubit 1
    P_1_counts : np.ndarray
        Counts histogram for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is the set of
        strings (in corresponding order) given by P_1_strings
    P_2_strings : tuple[str]
        List of measurement strings in little-endian binary for distribution 2
    P_2_ints : tuple[int]
        List of int repr of measurement strings on qubit 2
    P_2_counts : np.ndarray
        Counts histogram for the measurments on qubit 2
        P^{(2)}_U(s_B) = Tr[ U_B rho_2 U^dagger_B |s_B rangle langle s_B| ]
        where U is a fixed, randomly chosen unitary, and s_B is the set of
        strings (in corresponding order) given by P_2_strings

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    # normalise counts
    P_1_dist = P_1_counts / np.sum(P_1_counts)
    P_2_dist = P_2_counts / np.sum(P_2_counts)

    # iterate over the elements of the computational basis (that
    # appear in the measurement results)sublimes
    corr_fixed_u = 0
    for sA, counts_1_sA, P_1_sA in zip(P_1_strings, P_1_counts, P_1_dist):

        # skip if counts has a zero
        if counts_1_sA == 0:
            continue

        for sAprime, counts_2_sAprime, P_2_sAprime in zip(P_2_strings,
                                                          P_2_counts,
                                                          P_2_dist):

            # skip if counts has a zero
            if counts_2_sAprime == 0:
                continue

            # add up contribution
            hamming_distance = int(
                len(sA)*sp.spatial.distance.hamming(list(sA), list(sAprime))
            )
            corr_fixed_u += (
                (-2.)**(-hamming_distance) * P_1_sA*P_2_sAprime
            )

    # normalise counts / (np.sum(P_1_counts) * np.sum(P_2_counts))
    return corr_fixed_u 


def _auto_cross_correlation_single_u(P_1_strings, P_1_ints, P_1_counts):
    """
    Carries out the inner loop purity calculation of arxiv:1909.01282 and
    arxiv:1801.00999, etc. In contrast to the two-source calculation above
    (_cross_correlation_fixed_u), in this case there is only one measured
    distribution of bit string probabilities so we need to take additional care
    to avoid estimator bias when computing cross-correlations.

    Parameters
    ----------
    P_1_strings : tuple[str]
        List of measurement strings in little-endian binary for distribution 1
    P_1_ints : tuple[int]
        List of int repr of measurement strings on qubit 1
    P_1_counts : np.ndarray
        Counts histogram for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is the set of
        strings (in corresponding order) given by P_1_strings

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    # normalise counts
    num_measurements = np.sum(P_1_counts)
    P_1_dist = P_1_counts / num_measurements

    # iterate over the elements of the computational basis (that
    # appear in the measurement results)sublimes
    corr_fixed_u = 0
    for sA, counts_1_sA, P_1_sA in zip(P_1_strings, P_1_counts, P_1_dist):

        # skip if counts has a zero
        if counts_1_sA == 0:
            continue

        for sAprime, counts_1_sAprime, P_1_sAprime in zip(P_1_strings,
                                                          P_1_counts,
                                                          P_1_dist):

            # skip if counts has a zero
            if counts_1_sAprime == 0:
                continue

            if sA == sAprime:
                # bias corrected
                corr_fixed_u += (
                    P_1_sA * (num_measurements*P_1_sA - 1)
                    / (num_measurements - 1)
                )
            else:
                hamming_distance = int(
                    len(sA)*sp.spatial.distance.hamming(list(sA),
                                                        list(sAprime))
                )
                # bias corrected
                corr_fixed_u += (
                    (-2.)**(-hamming_distance) * P_1_sA*P_1_sAprime
                    * num_measurements / (num_measurements - 1)
                )

    # normalise counts / num_measurements**2
    return corr_fixed_u


@functools.lru_cache(maxsize=1)
def _make_full_expon_hamming_distance_matrix(num_qubits):
    """ """
    return (-2.) ** (-1.*np.count_nonzero(
        (
            np.array(
                [list(format(val, '0'+str(num_qubits)+'b'))
                 for val in range(2**num_qubits)]
            )[:, np.newaxis]
            != np.array(
                [list(format(val, '0'+str(num_qubits)+'b'))
                 for val in range(2**num_qubits)]
            )[np.newaxis, :]
        ),
        axis=2,
    ))


def _make_expon_hamming_distance_matrix(
    int_keys_a, int_keys_b, num_qubits
):
    """ """
    full_distance_matrix = _make_full_expon_hamming_distance_matrix(num_qubits)
    return full_distance_matrix[np.ix_(int_keys_a, int_keys_b)]


def _vectorised_cross_correlation_single_u(
    P_1_strings,
    P_1_ints,
    P_1_counts,
    P_2_strings,
    P_2_ints,
    P_2_counts,
):
    """
    Carries out the inner loop calculation of the Cross-Fidelity. In
    contrast to the paper, arxiv:1909.01282, it makes sense for us to
    make the sum over sA and sA' the inner loop. So this computes the
    sum over sA and sA' for fixed random U.

    Vectorised calculation should be slightly faster.

    Parameters
    ----------
    P_1_strings : tuple[str]
        List of measurement strings in little-endian binary for distribution 1
    P_1_ints : tuple[int]
        List of int repr of measurement strings on qubit 1
    P_1_counts : np.ndarray
        Counts histogram for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is the set of
        strings (in corresponding order) given by P_1_strings
    P_2_strings : tuple[str]
        List of measurement strings in little-endian binary for distribution 2
    P_2_ints : tuple[int]
        List of int repr of measurement strings on qubit 2
    P_2_counts : np.ndarray
        Counts histogram for the measurments on qubit 2
        P^{(2)}_U(s_B) = Tr[ U_B rho_2 U^dagger_B |s_B rangle langle s_B| ]
        where U is a fixed, randomly chosen unitary, and s_B is the set of
        strings (in corresponding order) given by P_2_strings

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    num_qubits = len(P_1_strings[0])

    # normalise counts
    P_1_dist = P_1_counts / np.sum(P_1_counts)
    P_2_dist = P_2_counts / np.sum(P_2_counts)

    # full_distance_matrix = _make_full_expon_hamming_distance_matrix(
    #     num_qubits)
    # _slice_indexes = np.ix_(P_1_ints, P_2_ints)
    # return np.dot(
    #     P_1_dist, np.dot(
    #         full_distance_matrix[_slice_indexes], P_2_dist
    #     )
    # )

    expon_hamming_distances = _make_expon_hamming_distance_matrix(
        P_1_ints, P_2_ints, num_qubits)

    # evaluate sum and normalise counts
    return np.dot(
        P_1_dist,
        np.dot(
            expon_hamming_distances,
            P_2_dist
        )
    )  # / (np.sum(P_1_counts) * np.sum(P_2_counts))


def _vectorised_auto_cross_correlation_single_u(
    P_1_strings, P_1_ints, P_1_counts
):
    """
    Carries out the inner loop purity calculation of arxiv:1909.01282 and
    arxiv:1801.00999, etc. In contrast to the two-source calculation above
    (_cross_correlation_fixed_u), in this case there is only one measured
    distribution of bit string probabilities so we need to take additional care
    to avoid estimator bias when computing cross-correlations.

    Vectorised calculation should be slightly faster.

    Parameters
    ----------
    P_1_strings : tuple[str]
        List of measurement strings in little-endian binary for distribution 1
    P_1_ints : tuple[int]
        List of int repr of measurement strings
    P_1_counts : np.ndarray
        Counts histogram for the measurments on qubit 1
        P^{(1)}_U(s_A) = Tr[ U_A rho_1 U^dagger_A |s_A rangle langle s_A| ]
        where U is a fixed, randomly chosen unitary, and s_A is the set of
        strings (in corresponding order) given by P_1_strings

    Returns
    -------
    float
        Evaluation of the inner sum of the cross-fidelity
    """
    num_qubits = len(P_1_strings[0])

    # normalise counts
    num_measurements = np.sum(P_1_counts)
    P_1_dist = P_1_counts / num_measurements

    expon_hamming_distances = _make_expon_hamming_distance_matrix(
        P_1_ints, P_1_ints, num_qubits)

    # evaluate sum and normalise counts
    vectorised_sum = np.dot(
        P_1_dist, np.dot(expon_hamming_distances, P_1_dist)
    )  # / num_measurements**2

    # full_distance_matrix = _make_full_expon_hamming_distance_matrix(num_qubits)
    # _slice_indexes = np.ix_(P_1_ints, P_1_ints)
    # vectorised_sum = np.dot(
    #     P_1_dist, np.dot(
    #         full_distance_matrix[_slice_indexes], P_1_dist
    #     )
    # )

    # correct bias
    return (vectorised_sum * num_measurements - 1) / (num_measurements - 1)
