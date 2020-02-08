import unittest
import algorithms


class TestAlgorithms(unittest.TestCase):
	def test_RUP_check_data(self):
		with self.assertRaises(ValueError):
			solver = algorithms.RUP_Solver([(1,2)], [(1,1), (2,3)])
		with self.assertRaises(ValueError):
			solver = algorithms.RUP_Solver([(1,1), (1,1)], [(1,2), (2,3)])

	def test_RUP_solve(self):
		solver = algorithms.RUP_Solver([(0,0), (0,1)], [(5,1), (1,0)])
		self.assertEqual([[[0,0], [0,1]], [[1,0], [5,1]]], solver.solve())

	def test_GRP_Network_from_graph_dict(self):
		G = {0:[1], 1:[0,2,3,4], 2:[1,3], 3:[1,2,4,5], 4:[1,3], 5:[3]}
		val = [(10,5), (10,5), (15,35), (50,10), (25,50), (5,10)]
		network = algorithms.GRP_Network.from_graph_dict(G, val)
		grp_vertices_str = \
			"[<0 (10, 5)>, <1 (10, 5)>, <2 (15, 35)>, <3 (50, 10)>, <4 (25, 50)>, <5 (5, 10)>]"
		grp_edges_str = "[(<0 (10, 5)>, <1 (10, 5)>), (<1 (10, 5)>, <0 (10, 5)>), (<1 (10, 5)>, "+\
			"<2 (15, 35)>), (<1 (10, 5)>, <3 (50, 10)>), (<1 (10, 5)>, <4 (25, 50)>), (<2 (15, 35)>, "+\
			"<1 (10, 5)>), (<2 (15, 35)>, <3 (50, 10)>), (<3 (50, 10)>, <1 (10, 5)>), (<3 (50, 10)>, "+\
			"<2 (15, 35)>), (<3 (50, 10)>, <4 (25, 50)>), (<3 (50, 10)>, <5 (5, 10)>), (<4 (25, 50)>, "+\
			"<1 (10, 5)>), (<4 (25, 50)>, <3 (50, 10)>), (<5 (5, 10)>, <3 (50, 10)>)]"
		self.assertEqual(grp_vertices_str, str(network.vertices))
		self.assertEqual(grp_edges_str, str(network.edges))

	def test_GRP_Network_from_grid_graph(self):
		g = algorithms.GRP_Network.from_grid_graph(algorithms.np.array([[1,2,3],[0,0,0]]), 
			algorithms.np.array([[0,0,1],[1,2,2]]))
		vertices_str = "[<0,0 (1, 0)>, <0,1 (2, 0)>, <0,2 (3, 1)>, <1,0 (0, 1)>, <1,1 (0, 2)>, <1,2 (0, 2)>]"
		edges_str = "[(<0,0 (1, 0)>, <0,1 (2, 0)>), (<0,0 (1, 0)>, <1,0 (0, 1)>), (<0,1 (2, 0)>, <0,0 (1, 0)>), "+\
			"(<0,1 (2, 0)>, <0,2 (3, 1)>), (<0,1 (2, 0)>, <1,1 (0, 2)>), (<0,2 (3, 1)>, <0,1 (2, 0)>), (<0,2 (3, 1)>, "+\
			"<1,2 (0, 2)>), (<1,0 (0, 1)>, <0,0 (1, 0)>), (<1,0 (0, 1)>, <1,1 (0, 2)>), (<1,1 (0, 2)>, <0,1 (2, 0)>), "+\
			"(<1,1 (0, 2)>, <1,0 (0, 1)>), (<1,1 (0, 2)>, <1,2 (0, 2)>), (<1,2 (0, 2)>, <0,2 (3, 1)>), (<1,2 (0, 2)>, "+\
			"<1,1 (0, 2)>)]"
		self.assertEqual(vertices_str, str(g.vertices))
		self.assertEqual(edges_str, str(g.edges))

	def test_GRP_Solver_solve_MUP(self):
		g = algorithms.GRP_Network.from_grid_graph(algorithms.np.array([[1,2,3],[0,0,0]]), 
			algorithms.np.array([[0,0,1],[1,2,2]]))
		self.assertEqual("[0. 1. 0. 0. 2. 0. 2. 0. 0. 0. 0. 0. 0. 0.]", str(algorithms.GRP_Solver.solve_MUP(g)))

	def test_GRP_Solver_solve_ZMUP(self):
		g = algorithms.GRP_Network.from_grid_graph(algorithms.np.array([[1,2,3],[0,0,0]]), 
			algorithms.np.array([[0,0,1],[1,2,2]]))
		capacities = [3 for _ in range(len(g.edges))]
		test_str = "([0.0, 1.5, 0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.6666666666666666)"
		self.assertEqual(test_str, str(algorithms.GRP_Solver.solve_ZMUP(g, capacities)))

	def test_GRP_Solver_solve_SZUP(self): 
		a1 = algorithms.np.array([[11,0,9],[20,0,0],[0,7,13]])
		a2 = algorithms.np.array([[0,20,0],[0,20,0],[0,20,0]])
		s = algorithms.GRP_Solver.solve_SZUP(a1, a2, 6)
		res_str = """[array([[11,  0,  9],
       [20,  0,  0],
       [ 0,  7, 13]]), array([[ 5, 12,  3],
       [14,  6,  0],
       [ 0, 13,  7]]), array([[ 0, 20,  0],
       [ 8, 12,  0],
       [ 0, 19,  1]]), array([[ 0, 20,  0],
       [ 2, 18,  0],
       [ 0, 20,  0]]), array([[ 0, 20,  0],
       [ 0, 20,  0],
       [ 0, 20,  0]])]"""
		self.assertEqual(str(s), res_str)


if __name__ == '__main__':
	unittest.main()