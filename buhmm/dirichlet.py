# encoding: utf-8
"""
User-friendly implementations of DirichletDistribution and Infer.

The big difference is that we interface to CMPy, and do not require that
the user work with standardized data or transition matrices.

"""
from __future__ import division

from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import product

from . import _dirichlet
from .exceptions import NonunifilarException, InvalidInitialNode, InvalidNode
from .canonical import tmatrix

import cmpy
import dit
from dit.inference import standardize_data
import numpy as np
import networkx as nx

__all__ = [
    'DirichletDistribution',
    'DirichletDistributionCP',
    'Infer',
    'InferCP',
    'ModelComparison',
    'ModelComparisonPredictive',
]

class DirichletDistribution(object):
    """
    The reason we do not subclass has to do with the fact that we are
    fundamentally changing assumptions that the Cython implemented class
    _dirichlet.DirichletDistribution makes. As an example, to get a fast
    cython loop, we write for loops over integers as in:

        for i in range(self.nNodes):

    If we had done:

        for i in self.nodes:

    Then it would be possible to subclass, but then the loop in the Cython
    class would not be fast.

    Note that we are not exposing the node paths. They exist in standardized
    form on the Cython distribution only.

    """

    ### Public
    nodes = None
    symbols = None
    machine = None
    valid_initial_nodes = None
    final_node = None

    nNodes = None
    nSymbols = None
    nInitial = None
    nEdges = None

    ### Private

    # The Cython implementation of Dirichlet distribution.
    _dd = None

    # dicts mapping nodes to integers.
    _node_index = None
    _symbol_index = None

    def __init__(self, machine, data=None, node_path=False, prng=None, out_arrays=None, verify=True):

        if not machine.is_unifilar():
            raise NonunifilarException()

        if data is None:
            data = []

        # Canonicalize the machine and data.
        symbols = sorted(machine.alphabet())
        seen = sorted(set(data))
        if verify and not set(data).issubset(set(symbols)):
            msg = "Data contains symbols not in the alphabet of the machine!\n"
            msg += "\n\t {0: <18} {1}".format("Machine Alphabet:", symbols)
            msg += "\n\t {0: <18} {1}".format("Data Alphabet:", seen)
            raise Exception(msg)

        data, _ = standardize_data(data, symbols)

        delta, nodes, _ = tmatrix(machine, symbol_order=symbols)

        self.nodes = nodes
        self.symbols = symbols
        self.machine = machine

        self._node_index = dict(zip(nodes, range(len(nodes))))
        self._symbol_index = dict(zip(symbols, range(len(nodes))))

        dd = _dirichlet.DirichletDistribution
        self._dd = dd(delta, data, node_path, prng, out_arrays)

        self._post_init(self)

    @staticmethod
    def _post_init(dirichlet):
        """
        Post initialization once the Cython Dirichlet distribution is
        updated, replaced, or new.

        """
        vin = [dirichlet.nodes[i] for i in dirichlet._dd.valid_initial_nodes]
        dirichlet.valid_initial_nodes = vin

        # We store the final nodes as a dictionary, instead of a list, so that
        # we can handle arbitrary initial nodes. The Cython version uses a list
        # and indexes with integers.
        final = [dirichlet.nodes[i] if i != -1 else None
                 for i in dirichlet._dd.final_node]
        dirichlet.final_node = OrderedDict(zip(dirichlet.nodes, final))

        # These are only needed when dirichlet._dd is new or replaced, but not
        # when updated. But its quick either way, so we include it.
        dirichlet.prng = dirichlet._dd.prng
        dirichlet.nNodes = dirichlet._dd.nNodes
        dirichlet.nSymbols = dirichlet._dd.nSymbols
        dirichlet.nInitial = dirichlet._dd.nInitial
        dirichlet.nEdges = dirichlet._dd.nEdges

    def _get_node_index(self, node):
        try:
            inode = self._node_index[node]
        except KeyError:
            raise InvalidNode(node)
        else:
            return inode

    def _require_valid_initial_node(self, node):
        """
        Returns the node index and verifies that it is a valid initial node.

        """
        inode = self._get_node_index(node)
        if self._dd.final_node[inode] == -1:
            raise InvalidInitialNode(node)
        else:
            return inode

    def add_counts_from(self, data, verify=False):
        """
        Update the posterior by adding counts from new data.

        If node paths are stored, they will not be updated.

        """
        data, data_symbols = standardize_data(data, self.symbols)
        if verify:
            if not set(data_symbols).issubset(set(self.symbols)):
                msg = "Data contains symbols not in the alphabet of the machine!\n"
                msg += "\n\t {0: <18} {1}".format("Machine Alphabet:", self.symbols)
                msg += "\n\t {0: <18} {1}".format("Data Alphabet:", data_symbols)
                raise Exception(msg)

        self._dd.add_counts_from(data)
        self._post_init(self)

    def log_evidence(self, initial_node):
        inode = self._get_node_index(initial_node)
        return self._dd.log_evidence(inode)

    def log_evidence_array(self):
        return self._dd.log_evidence_array()

    def sample_uhmm(self, initial_node, size=None, prng=None):
        inode = self._require_valid_initial_node(initial_node)
        return self._dd.sample_uhmm(inode, size=size, prng=prng)

    def pm_uhmm(self, initial_node):
        inode = self._require_valid_initial_node(initial_node)
        return self._dd.pm_uhmm(inode)

    def pm_uhmm_array(self):
        return self._dd.pm_uhmm_array()

    def _ntm(self, trans):
        return self._dd._ntm(trans)

    def get_updated_prior(self):
        """
        Returns a new DirichletDistribution that incorporates observed counts.

        """
        # We use deepcopy so that this works seamlessly with subclasses.

        # Pull out the things we don't want deepcopied. Deepcopy and put back.
        dd = self._dd
        machine = self.machine
        try:
            self._dd = None
            self.machine = None
            new = deepcopy(self)
        finally:
            self._dd = dd
            self.machine = machine

        # Now update the new Dirichlet
        new._dd = dd.get_updated_prior()
        new.machine = machine
        self._post_init(new)

        return new

    # New methods

    def sample_machine(self, initial_node, trim=False, prng=None):
        dd = self._dd
        inode = self._require_valid_initial_node(initial_node)
        trans = dd.sample_uhmm(inode, prng)
        m = self._build_machine(trans, trim=trim)
        m.graph['initial_node'] = initial_node
        fnode = dd.final_node[inode]
        m.graph['final_node'] = self.nodes[fnode]
        # If we trimmed, get rid of any node not reachable from the final node.
        if trim:
            reachable = nx.shortest_path_length(m, self.nodes[fnode])
            not_reachable = set(m.nodes()) - set(reachable)
            m.remove_nodes_from(not_reachable)
        return m

    def pm_machine(self, initial_node, trim=False):
        dd = self._dd
        inode = self._require_valid_initial_node(initial_node)
        trans = dd.pm_uhmm(inode)
        m = self._build_machine(trans, trim=trim)
        m.graph['initial_node'] = initial_node
        fnode = dd.final_node[inode]
        m.graph['final_node'] = self.nodes[fnode]
        # If we trimmed, get rid of any node not reachable from the final node.
        if trim:
            reachable = nx.shortest_path_length(m, self.nodes[fnode])
            not_reachable = set(m.nodes()) - set(reachable)
            m.remove_nodes_from(not_reachable)
        return m

    def _build_machine(self, trans, nodes=None, symbols=None, trim=False):
        from cmpy.machines import MealyHMM

        if trim is True:
            raise Exception("trim must be a float")

        if nodes is None:
            nodes = self.nodes
        if symbols is None:
            symbols = self.symbols

        m = MealyHMM()
        edges = self._dd.edges
        tmatrix = self._dd.tmatrix
        for i in range(len(edges)):
            u = edges[i, 0]
            x = edges[i, 1]
            v = tmatrix[u,x]
            p = trans[u,x]
            if trim and p < trim:
                continue
            uu = nodes[u]
            vv = nodes[v]
            xx = symbols[x]
            m.add_edge(uu, vv, o=xx, p=p)

        if trim:
            m.renormalize_edges()

        return m

class DirichletDistributionCP(DirichletDistribution):
    """
    Need to fix:   If data is ['00', '01', '10']
    and machines have alphabets like ['0', '1'], then
    the alphabet of the Cartesian product is more like [('0', '1'), ...]
    and so they are incompatible. Need to make this more flexible.

    """
    def __init__(self, machines, data=None, node_path=False, prng=None, out_arrays=None):

        tmatrices = []
        symbols = []
        nodes = []
        for machine in machines:
            if not machine.is_unifilar():
                raise NonunifilarException()

            symbols.append(sorted(machine.alphabet()))
            delta, nodes_, _ = tmatrix(machine, symbol_order=symbols[-1])
            tmatrices.append(delta)
            nodes.append(nodes_)

        self.nodes = list(product(*nodes))
        self.symbols = list(product(*symbols))
        self.machines = machines

        self._node_index = dict(zip(self.nodes, range(len(self.nodes))))
        self._symbol_index = dict(zip(self.symbols, range(len(self.symbols))))

        data, data_symbols = standardize_data(data)

        # Lets help detect mistakes in the machine and data.
        if not set(data_symbols).issubset(set(self.symbols)):
            msg = "Data contains symbols not in the alphabet of the machine!\n"
            msg += "\n\t {0: <18} {1}".format("Machine Alphabet:", self.symbols)
            msg += "\n\t {0: <18} {1}".format("Data Alphabet:", data_symbols)
            raise Exception(msg)

        dd = _dirichlet.DirichletDistributionCP
        self._dd = dd(tmatrices, data, node_path, prng, out_arrays)
        self.prng = self._dd.prng

        vin = [self.nodes[i] for i in self._dd.valid_initial_nodes]
        self.valid_initial_nodes = vin

        # We store the final nodes as a dictionary, instead of a list, so that
        # we can handle arbitrary initial nodes. The Cython version uses a list
        # and indexes with integers.
        final = [self.nodes[i] if i != -1 else None for i in self._dd.final_node]
        self.final_node = OrderedDict(zip(self.nodes, final))

        self.nNodes = self._dd.nNodes
        self.nSymbols = self._dd.nSymbols
        self.nInitial = self._dd.nInitial
        self.nEdges = self._dd.nEdges

    @classmethod
    def get_cp_machine(cls, machines):
        return cmpy.machines.cartesian_product_gg(machines)

class Infer(_dirichlet.Infer):
    """
    New methods are those which require a distribution over start nodes.

    """
    _posterior_class = DirichletDistribution

    def __init__(self, machine, data=None, inode_prior=None, node_path=False, prng=None, out_arrays=None, options=None):

        if data is None:
            data = []
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            # Each row is meant to be a symbol, but we need data[i] to be
            # hashable and right now, they are ndarrays.
            data = map(tuple, data)

        cls = super(Infer, self)
        cls.__init__(machine, data, inode_prior, node_path, prng, out_arrays, options)

        # Update edge counts with typical counts.

        # Update the node counts.
        #self.posterior._update_node_alphacounts(alpha=False, counts=True)


    # The following methods use the nodes to index into arrays. So we have
    # to redefine them to look up the node's index. They will not benefit
    # from Cython as much, since they remain untyped.

    def _fnode_init(self):
        # This averages over initial nodes. Recall, due to unifilarity, for any
        # given initial node, there is exact one final node.
        #
        # p(s_f | x) = \sum_{s_i} p(s_f | x, s_i) p(s_i | x)
        #
        ops = dit.math.LogOperations('e')
        pmf = np.zeros(self.posterior.nNodes, dtype=float)
        for initial_node in self.posterior.valid_initial_nodes:
            p = self.inode_posterior[initial_node]
            final_node = self.posterior.final_node[initial_node]
            fnode_index = self.posterior._get_node_index(final_node)
            pmf[fnode_index] += p

        nodes = self.posterior.nodes
        d = self._nodedist_class(nodes, pmf, base='linear', validate=False)
        d.make_sparse()
        self.fnode_dist = d

    def pm_next_symbol_dist(self):
        shape = (self.posterior.nInitial, self.posterior.nSymbols)
        probs = np.zeros(shape, dtype=float)

        # We must work with valid initial nodes since we are indexing with
        # the final node.
        for i, initial_node in enumerate(self.posterior.valid_initial_nodes):
            pm_uhmm = self.posterior.pm_uhmm(initial_node)
            final_node = self.posterior.final_node[initial_node]
            fnode_index = self.posterior._get_node_index(final_node)
            probs[i] = pm_uhmm[fnode_index]

        weights = dit.copypmf(self.inode_posterior, 'linear', 'sparse')
        weights = np.array([weights]).transpose()

        probs *= weights
        pmf = probs.sum(axis=0)

        d = self._symboldist_class(self.posterior.symbols, pmf)
        d.make_sparse()
        return d

    def pm_next_word_dists(self, lengths, base=None):
        """
        Calculate next word distributions of the specified lengths.

        The posterior mean transition probabilities and final state of each
        hidden Markov model are used to calculate the word probabilities.

        """
        if base is None:
            base = dit.ditParams['base']
        if base != 'linear':
            logs = True
        else:
            logs = False

        dists = defaultdict(list)
        for i, initial_node in enumerate(self.posterior.valid_initial_nodes):
            pm_machine = self.posterior.pm_machine(initial_node)
            final_node = self.posterior.final_node[initial_node]
            kwds = {'logs': logs, 'sort': 'dit', 'ndist': final_node}
            d = pm_machine.probabilities(lengths, **kwds)
            for L, dist in zip(lengths, d):
                if logs and base != 2:
                    dist = dist.set_base(base)
                dists[L].append(dist)

        weights = dit.copypmf(self.inode_posterior, base, 'sparse')
        avg_dists = []
        for L in lengths:
            avg_dists.append( dit.mixture_distribution(dists[L], weights) )

        return avg_dists

    def sample_machine(self, initial_node=None, trim=False, prng=None):
        if initial_node is None:
            inode = self.inode_posterior.rand(prng=prng)
        else:
            inode = initial_node
        return self.posterior.sample_machine(inode, trim=trim, prng=prng)

    def pm_machine(self, initial_node=None, trim=False, prng=None):
        if initial_node is None:
            inode = self.inode_posterior.rand(prng=prng)
        else:
            inode = initial_node
        return self.posterior.pm_machine(inode, trim=trim)

class InferCP(Infer):
    _nodedist_class = dit.Distribution
    _symboldist_class = dit.Distribution
    _posterior_class = DirichletDistributionCP

class BaseModelComparison(object):
    infers = None
    model_dist = None

    def sample_machine(self):
        i = self.model_dist.rand()
        return self.infers[i].sample_machine()

class ModelComparison(BaseModelComparison):
    """
    Generally, this should only be used to compare Infer instances with the
    same amount of data, since evidences go more and more negative with length.

    """
    log_evids = None

    def __init__(self, infers):
        self.infers = infers

        self.log_evids = np.array([infer.log_evidence() for infer in infers])

        base = 2
        ops = dit.math.get_ops(base)

        logevid = ops.add_reduce(self.log_evids)
        norm = ops.invert(logevid)
        pmf = np.array([ops.mult(evid, norm) for evid in self.log_evids])
        d = dit.ScalarDistribution(pmf, base=base)
        d.set_base('linear') # Is this wise?
        self.model_dist = d


class ModelComparisonDistance(ModelComparison):
    """

    """
    distances = None

    def __init__(self, infers, distribution, distance=None):
        """
        distribution is the true model's distribution
        distance is a measure of how close the inferred prediction is to the
            true model's distribution.

        """
        if distance is None:
            # JSD is normalized to 1. Use 1 - JSD.
            distance = dit.divergences.jensen_shannon_divergence

        self.infers = infers

        L = distribution.outcome_length()
        if L > 0:
            distances = []
            for infer in infers:
                d = infer.pm_next_word_dists([L], base='linear')[0]
                dist = distance([d, distribution])
                distances.append(dist)

        else:
            distances = np.zeros(len(infers))

        self.distances = distances = np.asarray(distances)


        # Anything that is zero will share the probability uniformly.
        # Otherwise, everything is will be inverted and normalized.
        pmf = distances.copy()
        is_zero = pmf == 0
        if np.any(is_zero):
            pmf *= 0
            pmf[is_zero] = 1 / is_zero.sum()
        else:
            pmf = 1 / pmf
            pmf /= pmf.sum()

        d = dit.ScalarDistribution(pmf, base='linear')
        self.model_dist = d

class ModelComparisonPredictive(ModelComparison):
    """
    Model Comparison using the posterior predictive distribution.

    """
    predictive_probabilities = None

    def __init__(self, infers, test=None):
        """
        Parameters
        ----------
        infers : list
            A list of Infer instances.
        test : iterable
            The posterior predictive test for each Infer instance.

        """
        self.infers = infers
        if test is not None:
            self.compare(test)

    def compare(self, test):
        base = 2
        ops = dit.math.get_ops(base)

        pmf = np.array([i.predictive_probability(test) for i in self.infers])
        predictive_probabilities = pmf.copy()
        norm = ops.add_reduce(pmf)
        if np.isinf(norm):
            # If every model assigns probability zero to `test`, then we
            # will treat all models as equally likely.
            pmf = ops.log(np.array([1] * len(self.infers), dtype=float))
        else:
            ops.mult_inplace(pmf, ops.invert(norm))

        d = dit.ScalarDistribution(pmf, base=base)
        d.set_base('linear')
        d.make_dense()

        self.model_dist = d
        self.predictive_probabilities = predictive_probabilities
        return d, predictive_probabilities
