import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from bruhat_tits import BruhatTitsTree
from connection import WeilGraphConnection
from data_loader import RiemannZerosLoader

class TestBruhatTitsTree(unittest.TestCase):
    def test_tree_construction_p2_d3(self):
        tree = BruhatTitsTree(p=2, depth=2)
        self.assertEqual(len(tree.graph.nodes), 10)

    def test_tree_degrees(self):
        # All internal nodes should have degree p+1.
        p = 2
        tree = BruhatTitsTree(p=p, depth=3)
        graph = tree.graph
        
        # Root has degree 3
        self.assertEqual(graph.degree(0), p + 1)
        
        # Nodes at level 1 should have degree 3 (1 parent, 2 children)
        for node, level in tree.node_level.items():
            if level == 1:
                self.assertEqual(graph.degree(node), p + 1)

    def test_no_forced_negative_weights(self):
        loader = RiemannZerosLoader()
        gammas = loader.load_odlyzko(num_zeros=120, dps=50)
        result = WeilGraphConnection.experiment_graph_weight_assignment(gammas, sigma=1.0, num_primes=40)

        resonance = result["resonance_prime"]
        w_broken = result["prime_data"][resonance]["w_broken"]
        neg_fraction = result["broken_negative_edge_fraction"]

        # If the computed broken contribution is positive, tree should not be force-flipped.
        if w_broken > 0:
            self.assertEqual(neg_fraction, 0.0)

if __name__ == '__main__':
    unittest.main()
