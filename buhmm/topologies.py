"""
Automated topology discovery.

The idea is to use a suite of techniques for identifying possible topologies.
Then feed these topologies into the Bayesian inference code.

One reason this is helpful is that if you truly suspect your underlying HMM
has many states, there isn't a practical way to enumerate all possible
topologies of, say, a 30 state HMM.

"""
from __future__ import division

from collections import deque, defaultdict

import numpy as np

import dit

from buhmm.exceptions import buhmmException, InvalidTopology, UnexpectedSymbol
from buhmm.twosample import chisq_twosample, bayesian_twosample


class Counts(object):
    def __init__(self, data, hLength, fLength, marginals=True, alphabet=None):
        """
        Parameters
        ----------
        data : iterable
            The data used to calculate the symbolic derivative.

        hLength : int
            The history word length to consider, initially. Queries involving
            longer history word lengths will automatically be handled.

        fLength : int
            The future word length to consider.

        marginals : bool
            If `True`, then the counts for all words less than `hLength` are
            calculated as well.

        alphabet : iterable
            The alphabet for the data. If `None`, then an extra pass through
            the data is required in order to obtain it.

        """
        self.data = data
        self.alphabet = alphabet

        self.fLength = fLength
        self.hLengths = set([])
        self.histories = []
        self.counts = None

        # Mapping from histories to row index into self.counts.
        self.index = {}

        self.calculate_counts(hLength, marginals)
        self.dist_length = len(self.alphabet)**fLength

    def __getitem__(self, x):
        L = len(x)
        if L not in self.hLengths:
            # Calculate counts for all words of the new length. Do not
            # calculate marginals though, since we may have already done that.
            self.calculate_counts(L, marginals=False)

        try:
            idx = self.index[x]
        except KeyError:
            # History does not appear in the data
            counts = np.zeros(self.dist_length)
            idx = None
        else:
            counts = self.counts[idx]

        return idx, counts

    def calculate_counts(self, hLength, marginals=True):
        """
        Calculates and stores the conditional counts from the data.

        Parameters
        ----------
        hLength : int
            The maximum word length to consider. If `None`, then use word
            length equal to ``L = len(data) - 1``.

        marginals : bool
            If `True`, then the symbolic derivative for all words less than
            `length` are calculated as well. If you don't marginalize, then the
            counts at smaller lengths might not be true marginals of the longer
            word lengths---the counts might be off by a few counts, depending
            on how large hLength is compared to the initial hLength.

        """
        data = self.data
        alphabet = self.alphabet

        if hLength > len(data) - self.fLength:
            msg = "`hLength` is too large."
            raise Exception(msg)

        if marginals:
            hLengths = {L for L in range(hLength + 1)}
            # Discard everything from before
            self.counts = None
            self.histories = []
            self.index = {}
            self.hLengths = set([])
        else:
            hLengths = {hLength}
        self.hLengths.update(hLengths)

        kwargs = {
            'data': data,
            'hLength': hLength,
            'fLength': self.fLength,
            'marginals': marginals,
            'alphabet': alphabet
        }
        out = dit.inference.counts_from_data(**kwargs)
        histories, cCounts, hCounts, alphabet = out

        if self.counts is not None:
            self.counts = np.vstack([self.counts, cCounts])
        else:
            self.counts = cCounts

        # Assign alphabet in case it was None was passed in.
        self.alphabet = alphabet

        # Extend the list of histories and the dictionary mapping them to rows.
        prev = len(self.histories)
        row_ids = range(prev, prev + len(histories))
        self.index.update(zip(histories, row_ids))
        self.histories.extend(histories)

def chisq_equal(alpha):
    def func(countsX, countsY):
        reject, pval = chisq_twosample(countsX, countsY, alpha)
        equal = not reject
        return equal
    return func

def bayesian_equal(prior=None):
    def func(countsX, countsY):
        reject, chi = bayesian_twosample(countsX, countsY, prior)
        equal = not reject
        return equal
    return func

def bool_equal(countsX, countsY):
    """
    Returns True if the locations of nonzero counts are identical.

    """
    countsX = np.asarray(countsX)
    countsY = np.asarray(countsY)
    return np.allclose(countsX > 0, countsY > 0)

def infnorm_equal(eps):
    """
    Returns a function that compares counts by the infinity norm.

    This is used by Chattopadhyay.

    Choosing eps is difficult, especially if you increase fLength since it
    could be that every probability is always less than a large eps.

    """
    def func(countsX, countsY):
        countsX = np.asarray(countsX)
        countsY = np.asarray(countsY)
        freqX = countsX / countsX.sum()
        freqY = countsY / countsY.sum()
        dist = np.max(np.abs(freqX - freqY))
        return dist < eps
    return func


def topology_from_single_history(counts, initial_history, equal_func):
    """
    Builds a topology by extending a single history.

    Only the attracting component is returned. If there is no attracting
    component, then None is returned.

    Note, due to non-transitivity of the equality metrics, the outcome
    will certainly depend on the order in which we visit symbols. We are
    choosing alphabetical order arbitrarily.

    """
    # Rename it, so we can use "counts" internally.
    Counts = counts

    eqclasses = defaultdict(list)

    # Set up the initial state
    state = 0
    history = initial_history
    eqclasses[state].append(history)
    todo = deque([state])

    def find_transition(next_history):
        """
        Determine if the counts should be "equal" to one of the
        existing states. We go through every history assigned to
        each state and say "yes it is equal" if there is at least
        one match. Note that due to non-transitivity, we will not
        have all histories within a state being "equal". In general, it is
        also possible that a count might be equal to more than one state.
        So this will only pick up the first such state.

        """
        idx, counts1 = Counts[next_history]

        # If idx is None, then it means the history wasn't observed. In that
        # case, counts1 is a zero vector.
        #
        #   For chisq_twosample, a zero vector will compare as equal to any
        #   other vector. This is because one of the rescaling parameters will
        #   be NaN.
        #
        #   For bayesian_twosample, the log ratio will always be zero if there
        #   is at least one zero vector. This means that the zero vector will
        #   compare as equal to any other vector.
        #
        #   For the infnorm, we can't create a frequency distribution. So it
        #   can't handle this either.
        #
        #   The bool_equal function can handle this just fine.
        #
        # Either way, we will treat this specially as a forbidden transition.
        # This is certainly not always the case---you may have not seen a word.
        # But in the limit as data length goes to infinity, this is a safe
        # assumption to make.
        #
        if idx is None:
            return -1

        for state2, histories in eqclasses.items():
            for history2 in histories:
                idx2, counts2 = Counts[history2]
                #print next_history, counts1
                #print history2, counts2
                if equal_func(counts1, counts2):
                    #print "Equal!"
                    return state2
                #else:
                #    print "Not equal"

        else:
            # New state
            return len(eqclasses)


    delta = []
    while todo:
        # Note, this will terminate eventually since the amount of data is
        # fixed. So eventually the distributions will all look alike.

        # Grab the state and mark it as done.
        state = todo.popleft()
        # Grab a representative history. Again, expect different outcomes
        # if we had chosen a different representative.
        history = eqclasses[state][0]
        #print "state", state
        transitions = []
        for symbol in Counts.alphabet:
            next_history = history + (symbol,)
            #print "Finding transition for:", state, history
            #print "   on symbol:", symbol
            next_state = find_transition(next_history)
            #print next_history, next_state
            transitions.append(next_state)

            if next_state != -1:
                if next_state not in eqclasses:
                    todo.append(next_state)

                # Now add the history to the eqclass of the state.
                eqclasses[next_state].append(next_history)

        delta.append(transitions)

    return np.asarray(delta)

def topology_identifier(delta, alphabet):
    """
    Returns the delta matrix for the largest attracting component and its id.

    """
    # Note that this delta is not a valid icdfa ordering since it wasn't
    # constructed via a traversal from an initial node. To get a proper one,
    # we need to use the icdfa_id and ICDFA.int_to_icdfa().

    # Also the delta will always have as many columns are there were symbols
    # in the data. But due to the how the transition matrix was constructed,
    # the resulting topology might not have the same size alphabet. We need
    # to flag these. Even if the topology has the correct alphabet, it could
    # be that the recurrent component does not have the correct alphabet
    # (e.g. due to dangling states or whatever).

    import cmpy
    from cmpy.orderlygen import ICDFA

    m = cmpy.machines.from_matrices((delta, None), style='deltaemission')

    # Take the largest attacting component
    acs = m.attracting_components()
    acs.sort(key=len, reverse=True)

    ac = acs[0]
    delta_ac, nodes, symbols = m.delta_matrix(ac)

    n = len(ac)
    k = len(symbols)

    if delta_ac.shape[1] < delta.shape[1]:
        # The effective alphabet of topology even before finding attracting
        # was smaller than the desired alphabet.
        icdfa = None
    elif np.any( (delta_ac == -1).sum(axis=0) == n ):
        # The attracting component didn't preserve the alphabet of the topology.
        icdfa = None
    else:
        icdfas = ICDFA.icdfas_from_delta(delta_ac.ravel(), n, k)
        icdfas = sorted(icdfas.tolist())
        # Use the first one as canonical.
        icdfa = tuple(icdfas[0])

    return n, k, icdfa

def generate_history_topologies(counts, hLength, equal_func):
    """
    Do not set hLength at the point when you don't think you will have good
    statistics. It specifies the length at which we begin traversing for the
    topology.

    """
    topologies = {}
    for history in counts.histories:
        if len(history) == hLength:
            delta = topology_from_single_history(counts, history, equal_func)
            identifier = topology_identifier(delta, counts.alphabet)
            if identifier[2] is not None:
                # Topology preserves alphabet.
                topologies[history] = identifier

    return topologies

def find_topologies(data, hLength, fLength):
    """
    Finds topologies using a variety of "distance" functions.

    """
    equal_funcs = [
        chisq_equal(.1),
        #chisq_equal(.2),
        chisq_equal(.3),
        #chisq_equal(.4),
        chisq_equal(.5),
        bool_equal,
        bayesian_equal(),
        #infnorm_equal(.5),
        #infnorm_equal(.4),
        infnorm_equal(.3),
        infnorm_equal(.2),
        infnorm_equal(.1),
    ]

    counts = Counts(data, hLength, fLength)
    all_tops = set([])
    for idx, equal_func in enumerate(equal_funcs):
        print "Equal func idx: {}".format(idx)
        tops = generate_history_topologies(counts, hLength, equal_func)
        all_tops.update(set(tops.values()))

    return all_tops, counts

def infer(data, hLength, fLength):
    """
    Returns a ModelComparison instance after discovering topologies.

    """
    from cmpy.orderlygen import ICDFA

    tops, counts = find_topologies(data, hLength, fLength)
    tops = sorted(tops)

    machs = [ICDFA.icdfa_to_machine(delta, n, k) for (n, k, delta) in tops]

    import buhmm
    infers = []
    for iden, m in zip(tops, machs):
        try:
            infer = buhmm.Infer(m, data, verify=True)
        except InvalidTopology:
            # The alphabet matches, but the topology is compatible.
            pass
        else:
            infers.append(infer)
    if len(infers) == 0:
        raise buhmmException("Could not find any valid topologies.")

    msg = "Found {} topologies, of which {} were valid."
    msg = msg.format(len(machs), len(infers))
    print(msg)
    mc = buhmm.ModelComparison(infers)
    return mc, counts

def main1():
    from cmpy import machines
    m = machines.ABC()
    m.prng.seed(0)
    d = m.symbols(1000)

    hLength = 5
    fLength = 2

    all_tops = find_topologies(d, hLength, fLength)

    icdfa = tuple(sorted(ICDFA.machine_to_icdfas_iter(m))[0])

    iden = (icdfa, len(m), len(m.alphabet()))
    print iden in all_tops

    return all_tops

def rrx_example():
    """
    Bayesian mc can't distinguish these machines...

    """
    import cmpy
    from cmpy.machines.algorithms.clonetransform import gaugify
    import buhmm

    m0 = cmpy.machines.RRX()
    m1 = cmpy.machines.from_string('A B 0; A C 1; B S1 0; C S2 1; B D 1; C D 0; S1 A 0; S2 A 0; D A 1;')
    m2 = cmpy.machines.from_string('A B 0; A C 1; B S 0; C S 1; B D1 1; C D2 0; D1 A 1; D2 A 1; S A 0;')
    m3 = cmpy.machines.from_string('A B 0; A C 1; B S1 0; C S2 1; B D1 1; C D2 0; D1 A 1; D2 A 1; S1 A 0; S2 A 0;')
    m4 = gaugify(m0, ['01|10'])
    m0.prng.seed(0)
    d = m0.symbols(1e6)
    infers = [buhmm.Infer(m, d) for m in [m0, m1, m2, m3, m4]]
    mc = buhmm.ModelComparison(infers)
    return mc, infers

if __name__ == '__main__':
    pass

