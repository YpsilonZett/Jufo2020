from math import sqrt, inf
import numpy as np 
from scipy.optimize import linear_sum_assignment, linprog


class RearrengementProblem: 
	def __init__(self, particle_positions, target_positions):  # TODO: documentation
		self.__positions = particle_positions
		self.__targets = target_positions
		self.__check_data()

	def __check_data(self):
		if len(self.__positions) != len(self.__targets):
			raise ValueError("Anzahl der Start- und Zielpositionen muss übereinstimmen")
		if len(self.__positions) != len(set(self.__positions)) or \
			len(self.__targets) != len(set(self.__targets)):
			raise ValueError("Start- bzw. Zielpositionen dürfen nicht mehrfach vorkommen")

	def __create_cost_matrix(self):
		self.__positions = np.array(self.__positions)
		self.__targets = np.array(self.__targets)
		cost = np.zeros((self.__positions.shape[0], self.__targets.shape[0]))
		for x in range(self.__positions.shape[0]):
			for y in range(self.__targets.shape[0]):
				cost[x, y] = np.linalg.norm(self.__positions[x] - self.__targets[y])
		return cost

	def solve(self):
		cost = self.__create_cost_matrix()
		row_ind, col_ind = linear_sum_assignment(cost)
		return [self.__positions[row_ind].tolist(), self.__targets[col_ind].tolist()]
	

class GRP_Vertex:  
	def __init__(self, name, start_value, target_value):  
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
	def __init__(self, vertexU, vertexV):  # vertices have to be GRP_Vertex datatype
		self.vertexU = vertexU 
		self.vertexV = vertexV

	def __eq__(self, other):
		return self.vertexV == other.vertexV and self.vertexU == other.vertexU

	def __repr__(self):
		return "({}, {})".format(self.vertexU, self.vertexV)


class GRP_Network:
	def __init__(self):
		self.vertices = []  # datatype GRP_Vertex
		self.edges = []  # datatype GRP_Edge

	@staticmethod 
	def __manhattan_dist(str1, str2): # strings have "x,y" format x,y are integers
		x1, y1 = str1.split(",")
		x2, y2 = str2.split(",")
		return abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))

	@staticmethod 
	def from_grid_graph(start_values, end_values):  # both are 2D np arrays with same shape
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
				if GRP_Network.__manhattan_dist(v1.name, v2.name) == 1:
					E.append(GRP_Edge(v1, v2))
		network = GRP_Network()
		network.vertices = V
		network.edges = E
		return network 

	@staticmethod
	def from_graph_dict(graph_dict, values):  
		# works with adjacency dictionary and value array (of tuples (start height, end height))
		# only with dict keys 0, 1, ..., n (because they are used to index the value array and name the vertices)
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
	def __init__(self, grp_network):  # has to be grp_network data type
		self.__graph = grp_network

	def __graph_to_equations(self):  # represents graph as lineq system (for linprog)
		nv = len(self.__graph.vertices)
		ne = len(self.__graph.edges)  
		M = sum(map(lambda v: v.start_value, self.__graph.vertices))  # total "mass" 
		bounds = [(0, M) for _ in range(ne)]  # \forall e \in |E|: -M \leq f(e) \leq M
		c = np.ones(ne)  
		A_eq = np.zeros((nv, ne))  
		vertex_pos = {v.name: i for i, v in enumerate(self.__graph.vertices)}
		for i in range(ne):  # add equations for flow conservation
			A_eq[vertex_pos[self.__graph.edges[i].vertexU.name], i] = -1
			A_eq[vertex_pos[self.__graph.edges[i].vertexV.name], i] = 1
		b_eq = [self.__graph.vertices[i].target_value - self.__graph.vertices[i].start_value
				for i in range(nv)]  
		return c, A_eq, b_eq, bounds

	def solve_MUP(self): 
		c, A, b, bnds = self.__graph_to_equations() 
		res = linprog(c, A_eq=A, b_eq=b, bounds=bnds)
		#print(res)
		self.__flows = np.round(res.x, 2)
		ambivalence = res.x - self.__flows 
		if np.max(np.abs(ambivalence)) > 0.4:
			raise ValueError("Lösung ist zu ungenau")
		if not res.success:
			raise ValueError("Gleichungssystem konnte nicht gelöst werden")	
		return self.__flows

	def solve_ZMUP(self, capacities):  # capacities has to be array indexed like flows array (like edges array)
		# NOTE: at the moment this isn't too pretty (because ZMUP was added later to the class)
		self.solve_MUP()
		q_min = inf
		for i in range(len(self.__graph.edges)):
			try:
				q_e = capacities[i] / self.__flows[i]
				if q_e < q_min:
					q_min = q_e 
			except:
				continue 
		if q_min < 1:
			raise ValueError("Problemstellung ist nicht lösbar!")
		new_flows = [f * q_min for f in self.__flows]
		t_end = 1 / q_min 
		return new_flows, t_end
		

if __name__ == "__main__":
	# TEST 1
	#G = {0:[1], 1:[0,2,3,4], 2:[1,3], 3:[1,2,4,5], 4:[1,3], 5:[3]}
	#val = [(10,5), (10,5), (15,35), (50,10), (25,50), (5,10)]
	#network = GRP_Network.from_graph_dict(G, val)
	#print(network.vertices, network.edges)
	# TEST 2
	#solver = GRP_Solver(network)
	#print(solver.solve_MUP())
	# TEST 3
	g = GRP_Network.from_grid_graph(np.array([[1,2,3],[0,0,0]]), np.array([[0,0,1],[1,2,2]]))
	print(g.vertices, g.edges)
	# TEST 4
	s = GRP_Solver(g)
	print(s.solve_MUP())
	print(s.solve_ZMUP([3 for _ in range(len(g.edges))]))