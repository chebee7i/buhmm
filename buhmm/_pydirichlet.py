# encoding: utf-8
# cython: profile=False
# cython: embedsignature=True
"""
Implementation of DirichletDistribution.

The Dirichlet distribution makes use of path counts from a DFA. Consider
two representations of the DFA for the golden mean process.

      0  1
    0 0  1
    1 0 -1

and

      0  1
    0 0  1
    1 0  2
    2 2  2

In the first, we have an incomplete DFA, whereas the second DFA is complete.
The first represents forbidden transitions with a special node, -1. There is no
explicit treatment of this node, and so, when counting paths, one must deal
with the reality that paths can terminate.  For the second DFA, node 2 is an
explicit node that receives all forbidden transitions. Self loops keep all
subsequent transitions at node 2. In this case, paths, even forbidden paths,
do not terminate prematurely. It should not be surprising that supporting
incomplete DFAs makes the code *more* complex. For example, the expression
for the likelihood that we use during inference is valid only if the path has
a nonzero probability, and so, the implementation must first check that the
path is valid.

Further complicating matters is that this implementation will infer transition
probabilities for every node that has more than one outgoing edge. Without
a mechanism for declaring which edges should be inferred and which should be
fixed, this means that doing inference on a complete DFA will yield undesirable
results---as forbidden edges will be inferred to have some nonzero probability.

What this means for the current implementation:

    If one is attempting to do inference on an HMM that does not have full
    support from each state, then one should pass in an incomplete DFA of its
    support, rather than a complete DFA. The expression for the likelihood
    (and thus, evidence) still only holds for words with nonzero probability.
    Edges to implicit, forbidden nodes will have probability 0, which is fixed
    and not inferred by the algorithm.

A goal for the future is to extend it so that one can declare which edges are
fixed and which should be inferred, and how the parameters might be related
to one another.

"""
from __future__ import absolute_import
from __future__ import division

import cython
#cimport cython

import numpy as np
#cimport numpy as np

from copy import deepcopy

from .counts import path_counts
from .exceptions import InvalidInitialNode

from itertools import product

BTYPE = np.bool
#ctypedef bint BTYPE_t

ITYPE = np.int64
#ctypedef np.int64_t ITYPE_t

__all__ = ['DirichletDistribution', 'Infer']

import dit

class DirichletDistribution(object):
    """
    A barebones representation of a product of Dirichlet distributions.

    """
    ### Public

    nodes = None
    symbols = None

    final_node = None
    valid_initial_nodes = None
    node_paths = None

    nNodes = None
    nSymbols = None
    nInitial = None
    nEdges = None

    prng = None

    ### Private

    tmatrix = None
    edges = None

    edge_alphas = None
    edge_counts = None

    node_alphas = None
    node_counts = None

    _temp = None

    def __init__(self, tmatrix, data=None, node_path=False, prng=None, out_arrays=None):
                 #np.ndarray[ITYPE_t, ndim=2, mode="c"] tmatrix,
                 #np.ndarray[ITYPE_t, ndim=1, mode="c"] data,
                 #BTYPE_t node_path=False,
                 #out_arrays=None):

        # In the follow, we use the following variables:
        #   n        : number of nodes
        #   k        : number of symbols
        #   L        : length of data
        #   nInitial : number of valid initial nodes
        #   nEdges   : number of edges

        if prng is None:
            prng = np.random.RandomState()
        self.prng = prng

        if data is None:
            data = np.array((), dtype=int)

        # shape: (n, k)
        # axes: (node, symbol)
        # Each element is the next node.
        self.tmatrix = tmatrix

        # shape: (nEdges, 2)
        # axes: (edge index, (node, symbol))
        self.edges = np.dstack(np.nonzero(tmatrix != -1))[0]

        # shape : (n,)
        # axes: (node,)
        # Not strictly necessary since the nodes are integers from zero.
        self.nodes = np.arange(tmatrix.shape[0])

        # shape: (k,)
        # axes: (symbol,)
        self.symbols = np.arange(tmatrix.shape[1])

        counts, final, node_paths = path_counts(tmatrix, data,
                                                node_path, out_arrays)

        # shape: (n, n, k)
        # axes: (initialNode, node, symbol)
        # Use float to support average counts.
        self.edge_counts = counts.astype(float)

        # shape: (n, n, k)
        # axes: (initialNode, node, symbol)
        # Start with uniform prior.
        self.edge_alphas = np.zeros(counts.shape, dtype=float) + 1

        self._update_node_alphacounts()

        # shape: (n,)
        # axes: (initial node,)
        # Each element is the final node.
        self.final_node = final

        # shape: (nInitial,)
        # axes: (initial node,)
        # Each element is a valid initial node.
        self.valid_initial_nodes = np.array(np.nonzero(final != -1)[0])

        # Eventually, we will need to determine which nodes have edges that
        # are to be inferred. Presently, this is every node since we cannot
        # have fixed edges with this algorithm. This will affect self.temp.

        # shape: (nNodes, L+1)
        # axes: (initialNode, time)
        self.node_paths = node_paths

        # The first row is for numerator terms
        # The second row is for denominator terms
        shape = (2, len(self.edges) + len(self.nodes))
        self._temp = np.empty(shape, dtype=float)

        self.nNodes = tmatrix.shape[0]
        self.nSymbols = tmatrix.shape[1]
        self.nInitial = len(self.valid_initial_nodes)
        self.nEdges = self.edges.shape[0]

    def _update_node_alphacounts(self, alpha=True, counts=True):
        """
        Recalculates `node_alpha` and `node_counts`.

        This must be called any time `edge_alpha` or `edge_counts` is updated.
        They are used to calculate the evidence.

        Practically, the node counts are the number of times each node was
        visited by some symbol. Effectively:

            node_count(initial_node, node)
                = \sum_{symbol} edge_count(initialNode, node, symbol)

        """
        # axes: (initialNode, node)
        # Each element is the count/alpha value.

        # Recall edge_counts and edge_alphas have:
        # shape: (n, n, k)
        # axes: (initialNode, node, symbol)

        # For the counts, if an edge was not traversed, then its count is
        # zero and will not affect the sum along axis=2. When we consider
        # the alphas, we must make sure that the alphas corresponding to
        # nonedges (assuming incomplete DFAs) do not effect the node alpha,
        # that is, the sum along axis=2. So we exclude nonedges from the sum.
        # This means the minimum node alpha for every (initial node, node) pair
        # is 1, even for nodes which have no edges that need to be inferred.
        # However, this is not a problem since algorithms, like the evidence,
        # will not query for those alpha values (since they use self.edges).
        #
        # The reason it is done this way is to simplify the data structure.
        # Technically, you only need priors for edges that are to be inferred.
        # As of now, the implementation is that these arrays will have fixed
        # size, no matter how many edges need to be inferred. An alternative
        # way to do this is to make axis=1 sparse and with size equal to the
        # number of edges to be inferred. We would then need to use a lookup to
        # match indexes along axis=1 to the edges.
        if alpha:
            condition = self.tmatrix != -1
            self.node_alphas = np.where(condition, self.edge_alphas, 0).sum(axis=2)
        if counts:
            self.node_counts = self.edge_counts.sum(axis=2)

    def add_counts_from(self, data):
        """
        Adds additional counts from `data`.

        """
        # For each symbol, add the count and update the final node.
        for symbol in data:
            for initial_node in self.valid_initial_nodes:
                final_node = self.final_node[initial_node]
                self.final_node[initial_node] = self.tmatrix[final_node, symbol]
                self.edge_counts[initial_node, final_node, symbol] += 1
                self.valid_initial_nodes = np.array(np.nonzero(self.final_node != -1)[0])

        self._update_node_alphacounts()

    def log_evidence(self, initial_node):
        """
        Returns the log evidence of the data using `node` as the initial node.

        Parameters
        ----------
        initial_node : int
            An initial node.

        Returns
        -------
        log_evid : float
            The base-2 log evidence of the data given the initial node. When
            its value is -inf, then it is not possible to generate the given
            data from the initial node. When its value is 0, then the given
            data is the only possible data that could be generated from the
            initial node.

        """
        if self.final_node[initial_node] == -1:
            # Then the data cannot be generated by this node.
            #
            # The form we use for the likelihood is valid only if the
            # probability of the data is nonzero. The reason is that it
            # requires edge counts for every edge, and we only obtain counts on
            # allowed edges. We could, alternatively, work with complete DFAs,
            # and then we *would* have counts for transitions following a
            # forbidden transition. In this case, the transition matrix would
            # have zero entries equal to -1 and one of the states would be the
            # garbage state. But this doesn't work for other reasons. See the
            # module docstring.
            log_evid = -np.inf

        else:
            from scipy.special import gammaln

            # shape: (2, nEdges + nNodes)
            temp = self._temp

            # It is no problem to iterate through nodes which only have
            # one edge, since the edge and node counts/alphas will cancel out.
            # Once we allow nodes with fixed probs, we will need to iterate
            # only through inferred edges and nodes with inferred edges.

            # Now iterate through every edge (u, x)
            edges = self.edges
            nEdges = self.nEdges
            ealphas = self.edge_alphas
            ecounts = self.edge_counts
            for i in range(nEdges):
                u = edges[i, 0]
                x = edges[i, 1]
                temp[0, i] = ealphas[initial_node, u, x] + \
                             ecounts[initial_node, u, x]
                temp[1, i] = ealphas[initial_node, u, x]

            # Similarly, iterate through every node (u, *)
            nalphas = self.node_alphas
            ncounts = self.node_counts
            for i in range(self.nNodes):
                temp[0, i + nEdges] = nalphas[initial_node, i]
                temp[1, i + nEdges] = nalphas[initial_node, i] + \
                                      ncounts[initial_node, i]

            gammaln(temp, temp)
            temp[1] *= -1
            log_evid = temp.sum()

        # Return base-2 logarithms.

        return log_evid / np.log(2)

    def log_evidence_array(self):
        """
        Returns an array of the log evidence of each node.

        """
        nNodes = self.nNodes
        log_evid = np.empty(nNodes)
        for i in range(nNodes):
            log_evid[i] = self.log_evidence(i)
        return log_evid

    def sample_uhmm(self, initial_node, size=None, prng=None):
        """
        Returns a uHMM sampled from the posterior.

        Parameters
        ----------
        initial_node : int
            The initial node.
        size : int
            The number of uHMMs to return.
        prng : np.RandomState
            A pseudorandom number generator, compatible with NumPy RandomState.

        Returns
        -------
        trans : NumPy array
            The transition probabilities of the uHMM. If `size` is None, then
            return a single transition matrix, shape (n, k). Otherwise, return
            `size` transition matrices in an array of shape (`size`, n, k).

        Raises
        ------
        Exception
            If `initial_node` is not a valid initial node.

        Notes
        -----
        The final node can be obtained from self.final_node[initial_node].

        """
        if prng is None:
            prng = self.prng

        final_node = self.final_node[initial_node]

        if final_node == -1:
            raise InvalidInitialNode(initial_node)

        post = self.edge_alphas[initial_node] + self.edge_counts[initial_node]
        condition = self.tmatrix != -1

        if size is None:
            shape = (1,) + self.tmatrix.shape
        else:
            shape = (size,) + self.tmatrix.shape

        trans = np.zeros(shape, dtype=float)
        for n in range(shape[0]):
            for i in range(shape[1]):
                cond = condition[i]
                trans[n, i, cond] = prng.dirichlet(post[i, cond])

        if size is None:
            trans = trans[0]

        return trans

    def pm_uhmm(self, initial_node):
        """
        Returns the posterior mean uHMM for the specified the initial node.

        Parameters
        ----------
        initial_node : int
            The initial node.

        Returns
        -------
        trans : NumPy array
            The transition probabilities of the uHMM.

        Raises
        ------
        InvalidInitialNode
            If `initial_node` is not a valid initial node.

        Notes
        -----
        The final node can be obtained from self.final_node[initial_node].

        """
        final_node = self.final_node[initial_node]

        if final_node == -1:
            raise InvalidInitialNode(initial_node)

        # This is a vectorized version of pm_edge_probability().

        # An edge is a node and symbol: s, x
        # alpha(s, x|s_i) + counts(s, x|s_i)
        trans = self.edge_alphas[initial_node] + \
                self.edge_counts[initial_node]

        # Now, we divide each row of trans by its normalization constant:
        #
        #   \sum_x (alpha(s, x | s_i) + counts(s, x | s_i))
        #
        # The node_* arrays have row/cols (initial_node, node). So we need
        # to associate their columns to the rows of trans. This is achieved
        # by dividing trans by a column vector. Before the [:, np.newaxis],
        # we have arrays of shape (n,). Afterwards, we have shape (n,1)
        trans /= (self.node_alphas[initial_node] + \
                  self.node_counts[initial_node])[:, np.newaxis]

        # It is necessary to explicitly mark forbidden transitions as having
        # zero probability since the alphas are nonzero for all transitions.
        condition = self.tmatrix == -1
        trans[condition] = 0

        return trans

    def pm_uhmm_array(self):
        """
        Returns an array of the posterior mean uHMMs.

        """
        uhmms = np.zeros((self.nInitial, self.nNodes, self.nSymbols))
        for i, initial_node in enumerate(self.valid_initial_nodes):
            uhmms[i] = self.pm_uhmm(initial_node)

        return uhmms

    def _ntm(self, trans):
        n = trans.shape[0]
        ntm = np.zeros((n,n), dtype=float)
        edges = self.edges
        tmatrix = self.tmatrix
        for i in range(len(edges)):
            u = edges[i, 0]
            x = edges[i, 1]
            v = tmatrix[u, x]
            ntm[u, v] += trans[u, x]

        return ntm

    def get_updated_prior(self):
        """
        Returns a new DirichletDistribution that incorporates observed counts.

        """
        new = deepcopy(self)

        # Transfer edge counts to edge alphas.
        new.edge_alphas += new.edge_counts
        new.edge_counts *= 0
        new._update_node_alphacounts()

        # To sample from the posterior, P( \theta | D, \sigma) we must keep the
        # same valid_initial_nodes. Note that the edge counts are zero in the
        # updated posterior. This suggests that the final_nodes should be
        # equal to the valid_initial_nodes since there is no data (e.g. no
        # edge counts). But doing this will not allow us to properly add new
        # since we *must* know the final state from all data seen (even if
        # the counts in the updated prior are now zero).

        return new

    def predictive_probability(self, x, initial_node):
        """
        Returns the mean predictive probability of `x` given `initial_node`.

        That is, we calculate::

            \Pr(x | D, \sigma) = \int \Pr( x | D, \theta, \sigma)
                                      \Pr( \theta | D, \sigma) d \theta

        This is a calculation from the posterior predictive distribution.

        Parameters
        ----------
        x : iterable
            The new data used to calculate the predictive probability.
        initial_node : int
            The initial node.

        Returns
        -------
        p : float
            The base-e logarithm of the mean predictive probability of `x`.

        Raises
        ------
        InvalidInitialNode
            If `initial_node` is not a valid initial node.

        """
        new = self.get_updated_prior()
        new.add_counts_from(x)
        return new.log_evidence(initial_node)

class DirichletDistributionCP(DirichletDistribution):
    """
    A Dirichlet distribution for Cartesian product inference.

    Importantly, the node/edge alpha and counts are not directly used to
    determine the posterior without first transforming them into the
    constituent parts of the Cartesian product.

    """
    ### Public

    nodes = None
    symbols = None

    final_node = None
    valid_initial_nodes = None
    node_paths = None

    nMatrices = None
    nNodes = None
    nSymbols = None
    nInitial = None
    nEdges = None

    prng = None

    ### Private

    tmatrices = None
    tmatrix = None
    edges = None

    _temp = None

    def __init__(self, tmatrices, data=None, node_path=False, prng=None, out_arrays=None):
        tmatrix = self._build_tmatrix(tmatrices, data)

        base = super(DirichletDistributionCP, self)
        base.__init__(tmatrix, data, node_path, prng, out_arrays)

    def _build_tmatrix(self, tmatrices, data):
        # list of length m
        # elements are arrays of shape: (n_i, k_i)
        # axes: (node, symbol) for the ith tmatrix.
        # Each element is the next node.
        self.tmatrices = tmatrices
        self.nMatrices = len(tmatrices)

        self.nNodes_array = np.array([tmatrix.shape[0] for tmatrix in tmatrices])
        nNodes = np.prod(self.nNodes_array)
        self.nodes = np.arange(nNodes)
        self.node_tuples = list(product(*[range(n) for n in self.nNodes_array]))
        self.node_tuples_index = dict(zip(self.node_tuples, self.nodes))

        self.nSymbols_array = np.array([tmatrix.shape[1] for tmatrix in tmatrices])
        nSymbols = np.prod(self.nSymbols_array)
        self.symbols = np.arange(nSymbols)
        self.symbol_tuples = list(product(*[range(n) for n in self.nSymbols_array]))
        self.symbol_tuples_index = dict(zip(self.symbol_tuples, self.symbols))

        shape = np.array([m.shape for m in self.tmatrices]).prod(axis=0)
        tmatrix = np.zeros(shape, dtype=int) - 1

        # Quick hack for now...generate the data for each tmatrix.
        # This requires a scan of the data for each tmatrix. Slow.
        # In principle, we can generate the counts/alphas with one scan,
        # and then propagate these values through summations to the counts
        # and alphas for each individual tmatrix.
        self.dd = []
        symbols = self.symbol_tuples
        for i,m in enumerate(tmatrices):
            if data is not None:
                data_ = np.array([symbols[sym][i] for sym in data])
            else:
                data_ = None
            self.dd.append( DirichletDistribution(m, data_) )

        for edges in product(*[dd.edges for dd in self.dd]):
            v = tuple(self.tmatrices[i][u, x] for i, (u, x) in enumerate(edges))
            u, x = zip(*edges)
            uu = self.node_tuples_index[u]
            vv = self.node_tuples_index[v]
            xx = self.symbol_tuples_index[x]
            tmatrix[uu, xx] = vv

        return tmatrix

    def log_evidence(self, initial_node):
        """
        Returns the log evidence of the data using `node` as the initial node.

        Parameters
        ----------
        initial_node : int
            An initial node.

        Returns
        -------
        log_evid : float
            The base-2 log evidence of the data given the initial node. When
            its value is -inf, then it is not possible to generate the given
            data from the initial node. When its value is 0, then the given
            data is the only possible data that could be generated from the
            initial node.

        """
        base = 2
        ops = dit.math.get_ops(base)

        node = self.node_tuples[initial_node]
        log_evids = np.array([self.dd[i].log_evidence(node[i])
                              for i in range(self.nMatrices)])
        log_evid = ops.mult_reduce(log_evids)

        return log_evid

    def sample_uhmm(self, initial_node, size=None, prng=None):
        """
        Returns a uHMM sampled from the posterior.

        Parameters
        ----------
        initial_node : int
            The initial node.
        size : int
            The number of uHMMs to return.
        prng : np.RandomState
            A pseudorandom number generator, compatible with NumPy RandomState.

        Returns
        -------
        trans : NumPy array
            The transition probabilities of the uHMM. If `size` is None, then
            return a single transition matrix, shape (n, k). Otherwise, return
            `size` transition matrices in an array of shape (`size`, n, k).

        Raises
        ------
        Exception
            If `initial_node` is not a valid initial node.

        Notes
        -----
        The final node can be obtained from self.final_node[initial_node].

        """
        if prng is None:
            prng = self.prng

        final_node = self.final_node[initial_node]
        if final_node == -1:
            raise InvalidInitialNode(initial_node)

        inodes = self.node_tuples[initial_node]
        uhmms = [self.dd[i].sample_uhmm(inodes[i], prng=prng)
                 for i in range(self.nMatrices)]

        trans = uhmms[0]
        for uhmm in uhmms[1:]:
            trans = np.kron(trans, uhmm)

        return trans

    def pm_uhmm(self, initial_node):
        """
        Returns the posterior mean uHMM for the specified the initial node.

        Parameters
        ----------
        initial_node : int
            The initial node.

        Returns
        -------
        trans : NumPy array
            The transition probabilities of the uHMM.

        Raises
        ------
        Exception
            If `initial_node` is not a valid initial node.

        Notes
        -----
        The final node can be obtained from self.final[initial_node].

        """
        final_node = self.final_node[initial_node]
        if final_node == -1:
            raise InvalidInitialNode(initial_node)

        inodes = self.node_tuples[initial_node]
        pm_uhmms = [self.dd[i].pm_uhmm(inodes[i])
                    for i in range(self.nMatrices)]

        trans = pm_uhmms[0]
        for pm_uhmm in pm_uhmms[1:]:
            trans = np.kron(trans, pm_uhmm)

        return trans

class Infer(object):
    """
    New methods are those which require a distribution over start nodes.

    """
    prng = None

    posterior = None
    inode_prior = None
    inode_posterior = None

    # The final node distribution is a deterministic function of the initial
    # node posterior distribution. It is a "posterior". For the prior, the
    # fnode_prior would be equal to inode_prior, and so, we do not include it
    # here.
    fnode_dist = None

    _nodedist_class = dit.ScalarDistribution
    _symboldist_class = dit.ScalarDistribution
    _posterior_class = DirichletDistribution

    def __init__(self, tmatrix, data=None, inode_prior=None, node_path=False, prng=None, out_arrays=None, options=None):
        """
        inode_prior is the initial node prior distribution.


        """
        # Allow the user to customize the classes used internally.
        if options is not None:
            attrs = ['nodedist_class', 'symboldist_class', 'posterior_class']
            for attr in attrs:
                _attr = '_' + attr
                setattr(self, _attr, options.get(attr, getattr(self, _attr)))

        if prng is None:
            prng = np.random.RandomState()
        self.prng = prng

        self.posterior = self._posterior_class(
            tmatrix, data, node_path, self.prng, out_arrays
        )

        self._inode_init(inode_prior)
        self._fnode_init()


    def _inode_init(self, inode_prior):
        #
        # Set up initial node prior distribution.
        #
        if inode_prior is None:
            outcomes = self.posterior.nodes
            n = self.posterior.nNodes
            pmf = [1 / n] * n
            inode_prior = self._nodedist_class(outcomes, pmf)
        else:
            # Assumes:
            #   1) the distribution is normalized
            #   2) sample space is range(n)

            if inode_prior.is_log():
                inode_prior.set_base('linear')

            # If the initial node dist does not assign positive probability to
            # any of the valid initial nodes, then the evidence (averaged over
            # the prior of nodes) will be 0, and the posterior over nodes is
            # not defined.  So we make sure that some probability is assigned
            # to at least one valid initial node.
            zero = inode_prior.ops.zero
            for node in self.posterior.valid_initial_nodes:
                if inode_prior[node] > zero:
                    break
            else:
                msg = "`inode_prior` does not assign probability to a valid node."
                raise Exception(msg)

        # There is no reason to make it sparse, except to match the posterior.
        inode_prior.make_sparse()
        self.inode_prior = inode_prior

        #
        # Calculate initial node posterior distribution. For state s and data x,
        #
        #   p(s|x) = p(x|s) p(s) / p(x)
        #
        # where p(x) = \sum_s p(x|s) p(s)
        #
        base = 2
        ops = dit.math.get_ops(base)

        p_xgs = self.posterior.log_evidence_array()
        # Need to use dense version of the prior's pmf
        p_s = dit.copypmf(inode_prior, base=base, mode='dense')
        p_sgx = ops.mult(p_xgs, p_s)
        p_x = ops.add_reduce(p_sgx)
        ops.mult_inplace(p_sgx, ops.invert(p_x))

        # No need to sort since the prior was already sorted.
        nodes = self.posterior.nodes
        d = self._nodedist_class(nodes, p_sgx, base=base, sort=False)
        d.set_base('linear')
        d.make_sparse()
        self.inode_posterior = d

    def _fnode_init(self):
        # This averages over initial nodes. Recall, due to unifilarity, for any
        # given initial node, there is exact one final node.
        #
        #   p(s_f | x) = \sum_{s_i} p(s_f | x, s_i) p(s_i | x)
        #
        # so p(s_f | x, s_i) is equal to 1.
        #
        ops = dit.math.LogOperations('e')
        pmf = np.zeros(self.posterior.nNodes, dtype=float)

        for initial_node in self.posterior.valid_initial_nodes:
            p = self.inode_posterior[initial_node]
            final_node = self.posterior.final_node[initial_node]
            pmf[final_node] += p

        nodes = self.posterior.nodes
        d = self._nodedist_class(nodes, pmf, base='linear', validate=False)
        d.make_sparse()
        self.fnode_dist = d

    def add_counts_from(self, data):
        """
        Adds additional counts from `data`.

        """
        self.posterior.add_counts_from(data)
        self._inode_init(self.inode_prior)
        self._fnode_init()

    def get_updated_prior(self):
        """
        Returns a new Infer that incorporates observed counts.

        """
        posterior = self.posterior
        try:
            self.posterior = None
            new = deepcopy(self)
        finally:
            self.posterior = posterior

        new.posterior = posterior.get_updated_prior()

        # The difference here is that we must use the inode_posterior as our
        # new initial distribution.
        new._inode_init(self.inode_posterior)
        # There is no need to reinit the fnode_dist since
        # new.posterior.valid_initial_nodes and new.posterior.final_node are
        # the same as in `self.posterior`.

        return new

    def pm_next_symbol_dist(self):
        # This averages over initial nodes.
        #
        #   p(x | D) = \sum_{s_i} p( x | D, s_i) p(s_i | D)
        #
        # where
        #
        #   p(x | D, s_i) = \int dtheta p(x | theta, D, s_i) p( theta | D, s_i)
        #
        #   p(x | theta, D, s_i) = \sum_{s_f} p( x, s_f | theta, D, s_i)
        #                        = p( x | theta, delta(s_i, D) )
        #
        # but note, this last equation is not really necessary for unifilar
        # HMMs because the symbol x uniquely identifies the next state s_f.
        # So we have:
        #
        #  p(x | D, s_i) = \int dtheta p(x | theta, delta(s_i, D)) p(theta | D, s_i)
        #
        # Thus,
        #
        #   p(x | D, s_i) = posterior mean of edge (x, delta(s_i, D))
        #
        # So for each valid initial node, we grab the row from the posterior
        # mean uHMM corresponding to its final node. These are the mean
        # probabilities of each symbol. This gives us a matrix of shape
        # (number of valid initial nodes, symbols). We construct a column
        # vector of the probability of each valid initial node and multiply
        # it elementwise on the rows (with broadcasting) to the mean
        # probabilities. Then, we sum the rows to get the final p(x | D).

        shape = (self.posterior.nInitial, self.posterior.nSymbols)
        probs = np.zeros(shape, dtype=float)

        # We must work with valid initial nodes since we are indexing with
        # the final node.
        for i, initial_node in enumerate(self.posterior.valid_initial_nodes):
            pm_uhmm = self.posterior.pm_uhmm(initial_node)
            final_node = self.posterior.final_node[initial_node]
            probs[i] = pm_uhmm[final_node]

        weights = dit.copypmf(self.inode_posterior, 'linear', 'sparse')
        weights = np.array([weights]).transpose()

        probs *= weights
        pmf = probs.sum(axis=0)

        d = self._symboldist_class(self.posterior.symbols, pmf)
        d.make_sparse()
        return d

    def log_evidence(self, initial_node=None):
        """
        Returns the log evidence of the data using `node` as the initial node.

            p(D | s)                if initial_node is not None

            \sum_s p(D|s) p(s)      if initial_node is None

        Parameters
        ----------
        initial_node : int, None
            An initial node. If `None`, then the expected log evidence is
            returned, where the expectation is over the initial node prior
            distribution.

        Returns
        -------
        log_evid : float
            The base-2 log evidence of the data.

        """
        if initial_node is not None:
            return self.posterior.log_evidence(initial_node)

        base = 2
        ops = dit.math.get_ops(base)

        p_s = dit.copypmf(self.inode_prior, base=base, mode='dense')
        evidences = self.posterior.log_evidence_array()
        log_evid = ops.add_reduce(ops.mult(evidences, p_s))

        return log_evid

    def sample_uhmm(self, initial_node=None, size=None, prng=None):
        """
        Returns uHMM transition matrices sampled from the posterior.

        Parameters
        ----------
        initial_node : int
            The initial node. If `None`, then the initial node is sampled from
            the initial node posterior distribution.
        size : int
            The number of uHMMs to return.
        prng : np.RandomState
            A pseudorandom number generator, compatible with NumPy RandomState.

        Returns
        -------
        inodes : int or NumPy array
            The initial nodes. If `size` is None, then return the integer
            corresponding to the sampled initial node. Otherwise, a NumPy array
            of shape (`size`,) containing the sampled initial nodes.
        trans : NumPy array
            The transition probabilities of the uHMM. If `size` is None, then
            return a single transition matrix, shape (n, k). Otherwise, return
            `size` transition matrices in an array of shape (`size`, n, k).

        Raises
        ------
        Exception
            If `initial_node` is not a valid initial node.

        """
        if prng is None:
            prng = self.prng

        single = False
        if size is None:
            size = 1
            single = True

        n, k = self.posterior.nNodes, self.posterior.nSymbols
        uhmms = np.zeros((size, n, k))
        if initial_node is None:
            inodes = self.inode_posterior.rand(size, prng=prng)
        else:
            inodes = [initial_node] * size

        for i, inode in enumerate(inodes):
            uhmms[i] = self.posterior.sample_uhmm(inode, prng=prng)

        if single:
            inodes = inodes[0]
            uhmms = uhmms[0]

        return inodes, uhmms

    # Depends on CMPy

    def sample_stationary_distributions(self, n=None, prng=None):
        from cmpy.math import stationary_distribution

        if prng is None:
            prng = self.prng

        if n is None:
            single = True
            n = 1
        else:
            single = False

        inodes = self.inode_posterior.rand(n, prng=prng)

        # n is likely to be small...let's build it up.
        pis = []
        posterior = self.posterior
        for initial_node in inodes:
            tmatrix = posterior.sample_uhmm(initial_node, prng=prng)
            ntm = posterior._ntm(tmatrix)
            pi = stationary_distribution(ntm, logs=False)
            pis.append(pi)

        if single:
            sdists = pis[0]
        else:
            sdists = np.array(pis)

        return sdists

    def predictive_probability(self, x, initial_node=None):
        """
        Returns the mean predictive probability of `x` given `initial_node`.

        That is, we calculate:

            \Pr(x | D, \sigma) = \int \Pr( x | D, \theta, \sigma)
                                      \Pr( \theta | D, \sigma) d \theta

        This is a calculation from the posterior predictive distribution. When
        `initial_node` is `None`, then we calculate:

            \Pr(x | D) = \sum \Pr(x | D, \sigma) \Pr( \sigma | D)

        Parameters
        ----------
        x : iterable
            The new data used to calculate the predictive probability.
        initial_node : int or None
            The initial node. If `None`, then the predictive probability is
            averaged over the initial node posterior distribution.

        Returns
        -------
        p : float
            The base-e logarithm of the mean predictive probability of `x`.

        Raises
        ------
        InvalidInitialNode
            If `initial_node` is not a valid initial node.

        """
        new = self.get_updated_prior()
        new.add_counts_from(x)
        return new.log_evidence(initial_node)

class InferCP(Infer):
    _posterior_class = DirichletDistributionCP
