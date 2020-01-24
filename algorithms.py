from math import sqrt
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
	

class GRP_Vertex:  # TODO: documentate!
	def __init__(self, name, start_value, target_value):
		self.name = name
		self.start_value = start_value
		self.target_value = target_value
		self.delta = self.start_value - self.target_value
		self.outgoing_edges = []
		self.ingoing_edges = []

	def __eq__(self, other):
		return self.name == other.name 

	def __repr__(self):
		return "<{} ({}, {})>".format(self.name, self.start_value, self.target_value)


class GRP_Edge:
	def __init__(self, vertexU, vertexV):
		self.vertexU = vertexU 
		self.vertexV = vertexV

	def __eq__(self, other):
		return self.vertexV == other.vertexV and self.vertexU == other.vertexU

	def __repr__(self):
		return "({}, {})".format(self.vertexU, self.vertexV)


class GRP_Network:
	def __init__(self):
		self.vertices = []
		self.edges = []

	def init_from_dict(self, graph_dict, values):
		for u in graph_dict.keys():
			self.vertices.append(GRP_Vertex(u, values[u][0], values[u][1]))
		for u in graph_dict.keys():
			for v in graph_dict[u]:
				e = GRP_Edge(self.vertices[u], self.vertices[v])
				self.edges.append(e)
				self.vertices[u].outgoing_edges.append(e)
				self.vertices[v].ingoing_edges.append(e)


class GRP_Solver:
	def __init__(self, grp_network):  # TODO: documentation
		self.__graph = grp_network

	def __graph_to_equations(self):
		nv = len(self.__graph.vertices)
		ne = len(self.__graph.edges)  
		M = sum(map(lambda v: v.start_value, self.__graph.vertices))  # total "mass" 
		bounds = [(0, M) for _ in range(ne)]  # \forall e \in |E|: -M \leq f(e) \leq M
		c = np.ones(ne)  
		A_eq = np.zeros((nv, ne))  
		for i in range(ne):
			A_eq[self.__graph.edges[i].vertexU.name, i] = -1
			A_eq[self.__graph.edges[i].vertexV.name, i] = 1
		b_eq = [self.__graph.vertices[i].target_value - self.__graph.vertices[i].start_value
				for i in range(nv)]
		return c, A_eq, b_eq, bounds

	def solve(self):
		c, A, b, bnds = self.__graph_to_equations() 
		res = linprog(c, A_eq=A, b_eq=b, bounds=bnds)
		#print(res)
		flows = np.round(res.x, 2)
		ambivalence = res.x - flows 
		if np.max(np.abs(ambivalence)) > 0.4:
			raise ValueError("Lösung ist zu ungenau")
		if not res.success:
			raise ValueError("Gleichungssystem konnte nicht gelöst werden")	
		return flows


if __name__ == "__main__":
	G = {0:[1], 1:[0,2,3,4], 2:[1,3], 3:[1,2,4,5], 4:[1,3], 5:[3]}
	val = [(10,5), (10,5), (15,35), (50,10), (25,50), (5,10)]
	network = GRP_Network()
	network.init_from_dict(G, val)
	solver = GRP_Solver(network)
	solver.solve()