import networkx as nx
import numpy as np
from typing import Dict, Tuple

class BruhatTitsTree:
    """
    Constructs a finite piece of the Bruhat-Tits tree for field Q_p.
    Provides visualization assuming the Weil functional hypothesis connects unit weights to signs.
    CONJECTURAL: Assignment of negative edge weights to model 'broken' unitarity.
    """
    
    def __init__(self, p: int = 2, depth: int = 5):
        """
        Args:
            p: prime number (each node except root will have degree p+1)
            depth: max tree depth from root
        """
        self.p = p
        self.depth = depth
        self.graph = nx.Graph()
        self.node_level = {}
        self.build()
        
    def build(self):
        """Builds a regular tree where every node has degree p+1 (root has p+1 children, others have p)."""
        node_id_counter = 0
        
        # Root node
        self.graph.add_node(node_id_counter)
        self.node_level[node_id_counter] = 0
        node_id_counter += 1
        
        # Queue for BFS: (parent_node_id, level)
        queue = [(0, 0)]
        
        while queue:
            parent_id, level = queue.pop(0)
            
            if level >= self.depth:
                continue
                
            # Root has p+1 children, internal nodes have p children (degree = p+1)
            num_children = self.p + 1 if parent_id == 0 else self.p
            
            for _ in range(num_children):
                child_id = node_id_counter
                node_id_counter += 1
                
                self.graph.add_node(child_id)
                self.node_level[child_id] = level + 1
                self.graph.add_edge(parent_id, child_id)
                queue.append((child_id, level + 1))
                
        print(f"✓ Built Bruhat-Tits tree: p={self.p}, depth={self.depth}")
        print(f"  Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")

    def assign_edge_weights_from_weil(self, p_contribution: float):
        """
        Assigns weights to edges based on the local Weil Functional contribution W_p.
        Our Toy Hypothesis (v3.0): 
        If W_p remains sufficiently positive -> physical tree, positive weights.
        If W_p becomes negative (due to resonance with a shifted zero) -> unitarity break, negative weights.
        """
        for u, v in self.graph.edges():
            base_weight = np.log(self.p)
            
            if p_contribution < 0:
                # Broken state: edge weight becomes negative proportional to the fracture
                weight = base_weight * p_contribution 
            else:
                # Healthy state
                weight = base_weight
                
            self.graph[u][v]['weight'] = weight

    def measure_unitarity_violation(self) -> float:
        """
        Returns fraction of negative edges.
        """
        negative_count = sum(1 for u, v, w in self.graph.edges(data='weight', default=0) if w < 0)
        total_edges = self.graph.number_of_edges()
        if total_edges == 0:
            return 0.0
        return negative_count / total_edges

    def visualize(self, filename: str = "tree.html"):
        """
        Visualizes the Bruhat-Tits tree using Plotly.
        Edge colors depend on weights (Red if < 0).
        """
        import plotly.graph_objects as go
        
        pos = self._compute_layout()
        
        edge_x, edge_y, edge_colors = [], [], []
        
        for u, v in self.graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = self.graph[u][v].get('weight', 0)
            color = 'red' if weight < 0 else 'gray'
            edge_colors.append(color)
            
        node_x = [pos[n][0] for n in self.graph.nodes()]
        node_y = [pos[n][1] for n in self.graph.nodes()]
        node_colors = [self.node_level.get(n, 0) for n in self.graph.nodes()]
        
        fig = go.Figure()
        
        # We add traces per edge individually to color them, or just use one color if not supported
        # Plotly doesn't easily support multi-color Line traces seamlessly without splitting.
        # So we'll iterate edges and add them to figure, or split by color
        
        edges_pos = [(u, v) for u, v in self.graph.edges() if self.graph[u][v].get('weight', 0) >= 0]
        edges_neg = [(u, v) for u, v in self.graph.edges() if self.graph[u][v].get('weight', 0) < 0]

        def get_lines(edge_list):
            ex, ey = [], []
            for u, v in edge_list:
                ex.extend([pos[u][0], pos[v][0], None])
                ey.extend([pos[u][1], pos[v][1], None])
            return ex, ey
            
        x_pos, y_pos = get_lines(edges_pos)
        x_neg, y_neg = get_lines(edges_neg)
        
        if x_pos:
            fig.add_trace(go.Scatter(x=x_pos, y=y_pos, mode='lines', line=dict(width=1, color='gray'), hoverinfo='none', name='Positive (Healthy)'))
        if x_neg:
            fig.add_trace(go.Scatter(x=x_neg, y=y_neg, mode='lines', line=dict(width=2, color='red'), hoverinfo='none', name='Negative (Broken)'))

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(size=8, color=node_colors, colorscale='Viridis', showscale=True, colorbar=dict(title="Level")),
            hovertext=[f"Node {n}, Level {self.node_level[n]}" for n in self.graph.nodes()], hoverinfo='text', name='Nodes'
        ))
        
        fig.update_layout(title=f"Bruhat-Tits Tree (p={self.p}, depth={self.depth})", hovermode='closest', showlegend=True)
        fig.write_html(filename)
        print(f"✓ Saved tree visualization to {filename}")

    def _compute_layout(self) -> Dict[int, Tuple[float, float]]:
        """A simple hierarchical layout heuristic."""
        try:
            # Requires Graphviz/PyGraphviz for actual tree layout in nx,
            # But we can write a manual BFS-based grid layout or use Kamada-Kawai if graph is small
            return nx.kamada_kawai_layout(self.graph)
        except:
            return nx.spring_layout(self.graph)
