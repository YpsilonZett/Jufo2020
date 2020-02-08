from math import sqrt, inf
import numpy as np 
from scipy.optimize import linear_sum_assignment, linprog


class RUP_Solver:  
	"""implements solution to Roboter-Umpositionierungsproblem"""
	def __init__(self, start_positions, target_positions): 
		"""expects two arrays of 2-tuples (x,y) with same length"""
		self.__start_positions = start_positions
		self.__target_positions = target_positions
		self.__check_data()

	def __check_data(self):  
		"""verifies that input has correct format"""
		if len(self.__start_positions) != len(self.__target_positions):
			raise ValueError("Anzahl der Start- und Zielpositionen muss übereinstimmen")
		if len(self.__start_positions) != len(set(self.__start_positions)) or \
			len(self.__target_positions) != len(set(self.__target_positions)):
			raise ValueError("Start- bzw. Zielpositionen dürfen nicht mehrfach vorkommen")

	def __create_cost_matrix(self): 
		"""builds adjacency matrix of graph defined by positions (vertices) and distances 
		   between them (edges)"""
		self.__start_positions = np.array(self.__start_positions)
		self.__target_positions = np.array(self.__target_positions)
		cost = np.zeros((self.__start_positions.shape[0], self.__target_positions.shape[0]))
		for x in range(self.__start_positions.shape[0]):
			for y in range(self.__target_positions.shape[0]):
				cost[x, y] = np.linalg.norm(self.__start_positions[x] - self.__target_positions[y])
		return cost

	def solve(self):
		"""uses scipy to solve the corresponding assignment problem, returns two arrays describing 
		   the start and end position of the ith element respectively"""
		cost = self.__create_cost_matrix()
		row_ind, col_ind = linear_sum_assignment(cost)
		return [self.__start_positions[row_ind].tolist(), self.__target_positions[col_ind].tolist()]
	

class GRP_Vertex:  
	"""datastructure for GRP_Network"""
	def __init__(self, name, start_value, target_value):  
		"""in- and outgoing edges are later set by GRP_Network"""
		self.name = name
		self.start_value = start_value
		self.target_value = target_value
		self.delta = self.start_value - self.target_value
		self.outgoing_edges = []  # datatype GRP_Edge
		self.ingoing_edges = []  # datatype GRP_Edge 

	def __eq__(self, other):
		return self.name == other.name 

	def __repr__(self):
		return "<{} ({}, {})>".format(self.name, self.start_value, self.target_value)


class GRP_Edge:  
	"""datastructure for GRP_Network"""
	def __init__(self, vertexU, vertexV):  
		"""vertices have to be of GRP_Vertex datatype"""
		self.vertexU = vertexU 
		self.vertexV = vertexV

	def __eq__(self, other):
		return self.vertexV == other.vertexV and self.vertexU == other.vertexU

	def __repr__(self):
		return "({}, {})".format(self.vertexU, self.vertexV)


class GRP_Network: 
	"""implements basic graph datastructure used by GRP_Solver"""
	def __init__(self):
		"""use datatypes GRP_Vertex and GRP_Edge for this"""
		self.vertices = []  
		self.edges = []  

	@classmethod 
	def __manhattan_dist(cls, str1, str2): 
		"""strings have "x,y" format x,y are integers"""
		x1, y1 = str1.split(",")
		x2, y2 = str2.split(",")
		return abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))

	@classmethod 
	def from_grid_graph(cls, start_values, end_values):  
		"""generates rectangular grid represented by directed graph;
		   both parameters have to be 2D np arrays with same shape"""
		if start_values.shape != end_values.shape and len(start_values.shape) == 2:
			raise ValueError
		V, E = [], []
		# build vertices
		n, m = start_values.shape
		for i in range(n):
			for j in range(m):
				V.append(GRP_Vertex("{},{}".format(i, j), start_values[i, j], end_values[i, j]))
		# build edges in grid form
		for v1 in V:
			for v2 in V:
				if cls.__manhattan_dist(v1.name, v2.name) == 1:
					E.append(GRP_Edge(v1, v2))
		network = GRP_Network()
		network.vertices = V
		network.edges = E
		return network 

	@classmethod
	def from_graph_dict(cls, graph_dict, values):  
		"""generates graph from adjacency dictionary and value array (of tuples (start height, end height));
		   dict keys have to be 0, 1, ..., n because they are used for indexing the vertices array"""
		network = GRP_Network()
		vertices, edges = [], []
		for u in graph_dict.keys():
			vertices.append(GRP_Vertex(u, values[u][0], values[u][1]))
		for u in graph_dict.keys():
			for v in graph_dict[u]:
				e = GRP_Edge(vertices[u], vertices[v])
				edges.append(e)
				vertices[u].outgoing_edges.append(e)
				vertices[v].ingoing_edges.append(e)
		network.vertices = vertices
		network.edges = edges 
		return network 


class GRP_Solver:
	"""implements solutions to the Graph Redistribution Problems"""
	@classmethod
	def __graph_to_equations(cls, grp_network):  
		"""represents graph as system of linear equations describing the mass flow (and constraints) in it"""
		nv = len(grp_network.vertices)
		ne = len(grp_network.edges)  
		M = sum(map(lambda v: v.start_value, grp_network.vertices))  # total "mass" 
		bounds = [(0, M) for _ in range(ne)]  # \forall e \in |E|: -M \leq f(e) \leq M
		c = np.ones(ne)  
		A_eq = np.zeros((nv, ne))  
		vertex_pos = {v.name: i for i, v in enumerate(grp_network.vertices)}
		for i in range(ne):  # add equations for flow conservation
			A_eq[vertex_pos[grp_network.edges[i].vertexU.name], i] = -1
			A_eq[vertex_pos[grp_network.edges[i].vertexV.name], i] = 1
		b_eq = [grp_network.vertices[i].target_value - grp_network.vertices[i].start_value
				for i in range(nv)]  
		return c, A_eq, b_eq, bounds

	@classmethod
	def solve_MUP_tree(cls, adj_dict):
		"""implements the tree algorithms to solve the special case of the MUP"""
		return  # TODO: implement and test 

	@classmethod
	def solve_MUP(cls, grp_network, method="interior-point"): 
		"""solves MUP (Masse-Umverteilungsproblem) for the given network using 
		   method "interior-point", "revised simplex" or "simplex" """
		c, A, b, bnds = cls.__graph_to_equations(grp_network) 
		res = linprog(c, A_eq=A, b_eq=b, bounds=bnds, method=method)
		flows = np.round(res.x, 2)
		ambivalence = res.x - flows 
		if np.max(np.abs(ambivalence)) > 0.4:
			raise ValueError("Lösung ist zu ungenau")
		if not res.success:
			raise ValueError("Gleichungssystem konnte nicht gelöst werden")	
		return flows

	@classmethod
	def solve_ZMUP(cls, grp_network, capacities, method="interior-point"):  
		"""solves ZMUP (zeitoptimales Masse-Umverteilungsproblem) by first solving MUP and using 
		   computes flows to solve ZMUP; note: capacities must be indexed like grp_network.edges"""
		flows = cls.solve_MUP(grp_network, method)
		q_min = inf
		for i in range(len(grp_network.edges)):
			if flows[i] != 0:
				q_e = capacities[i] / flows[i]
				if q_e < q_min:
					q_min = q_e 
		if q_min < 1:
			raise ValueError("Problemstellung ist nicht lösbar (Kapazitäten zu gering)!")
		new_flows = [f * q_min for f in flows]
		t_end = 1 / q_min 
		return new_flows, t_end
		
	@classmethod
	def solve_SZUP(cls, start_values, target_values, k_max, method="interior-point"):
		"""solves SZUP (Schaufel-Zellgitter-Umverteilungsproblem) where start_values, end_values are 2d np arrays 
		   (int) of same shape and k_max is maximum robot flow constant for grid"""
		if k_max != round(k_max) or k_max <= 0:
			raise ValueError("k_max muss eine natürliche Zahl sein")
		graph = GRP_Network.from_grid_graph(start_values, target_values)
		flows = cls.solve_MUP(graph, method)
		state = start_values.copy()
		process = [start_values.copy()]
		if flows != np.round(flows):
			print("WARNUNG: Lösung ist möglicherweise falsch!")
		while max(flows) > 0:
			for i in range(len(graph.edges)):
				x1, y1 = graph.edges[i].vertexU.name.split(",")
				x2, y2 = graph.edges[i].vertexV.name.split(",")
				if flows[i] >= k_max:
					f = k_max
				else:
					f = flows[i]
				state[int(x1), int(y1)] -= f 
				state[int(x2), int(y2)] += f
				flows[i] -= f
			process.append(state.copy())  
		return process 
