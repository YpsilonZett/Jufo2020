import algorithms

import sys
import traceback

import wx
import wx.lib.mixins.inspection as WIT

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas 
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

import networkx as nx


class ParticleRearrengementPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.add_canvas_with_toolbar()
        self.add_controls()

        self.SetSizer(self.sizer)
        self.Fit()

        #self.animator = animation.FuncAnimation(self.figure, self.anim, interval=1000)
        self.particle_positions = []  # TODO: set
        self.target_positions = []
        self.matching = []

    def anim(self, a):
        if(len(self.dataSet) == 0):
            return 0
        i = a % len(self.dataSet)

        self.subplot.clear()
        obj = self.subplot.scatter([0,1,2],[randint(1,10),1,1])
        return obj

    def add_canvas_with_toolbar(self):
        self.plot_panel = wx.Panel(self)
        self.plot_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.figure = plt.Figure(figsize=(8,8))
        self.subplot = self.figure.add_subplot(111)
        plt.title("Title")
        self.canvas = FigureCanvas(self.plot_panel, -1, self.figure)  
        self.plot_panel_sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.sizer.Add(self.plot_panel, 1, wx.EXPAND, 5)

        #toolbar = NavigationToolbar2Wx(self.canvas)
        #toolbar.Realize()
        #self.plot_panel_sizer.Add(toolbar, 1, wx.EXPAND, 5)
        #toolbar.update() 

    def add_controls(self): 
        panel = wx.Panel(self) 
        # particle positions
        l1 = wx.StaticText(panel, -1, "Neuer Partikel an \"x,y\"") 
        self.pos_input = wx.TextCtrl(panel) 
        create_btn = wx.Button(panel, wx.ID_ANY, label="Hinzufügen")
        create_btn.Bind(wx.EVT_BUTTON, self.create_particle)
        undo_btn = wx.Button(panel, wx.ID_ANY, label="Rückgängig")
        undo_btn.Bind(wx.EVT_BUTTON, self.undo_particle)
        # target positions 
        l2 = wx.StaticText(panel, -1, "Neue Zielposition an \"x,y\"") 
        self.pos_input2 = wx.TextCtrl(panel) 
        create_btn2 = wx.Button(panel, wx.ID_ANY, label="Hinzufügen")
        create_btn2.Bind(wx.EVT_BUTTON, self.create_target_position)
        undo_btn2 = wx.Button(panel, wx.ID_ANY, label="Rückgängig")
        undo_btn2.Bind(wx.EVT_BUTTON, self.undo_target_position)
        # control buttons
        calculate_btn = wx.Button(panel, wx.ID_ANY, label="Berechnen")
        calculate_btn.Bind(wx.EVT_BUTTON, self.calculate)
        simulate_btn = wx.Button(panel, wx.ID_ANY, label="Simulieren")
        simulate_btn.Bind(wx.EVT_BUTTON, self.simulate)
        # add to sizer
        vbox = wx.BoxSizer(wx.VERTICAL)  
        vbox.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(self.pos_input, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(create_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(undo_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.AddSpacer(4)
        vbox.Add(l2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(self.pos_input2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(create_btn2, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(undo_btn2, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.AddSpacer(4)
        vbox.Add(calculate_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(simulate_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        panel.SetSizer(vbox) 
        self.sizer.Add(panel)

    def create_particle(self, event):
        try:
            data = tuple(map(float, self.pos_input.GetValue().split(',')))
        except:
            wx.MessageBox("Fehler: Position muss das Format \"x,y\" haben", "Error", wx.OK | wx.ICON_ERROR)
            return 
        if data not in self.particle_positions:
            self.particle_positions.append(data)
        self.update_pos_plot()

    def undo_particle(self, event):
        if len(self.particle_positions) > 0:
            self.particle_positions.pop()
            self.update_pos_plot()

    def create_target_position(self, event):
        try:
            data = tuple(map(float, self.pos_input2.GetValue().split(',')))
        except:
            wx.MessageBox("Fehler: Position muss das Format \"x,y\" haben", "Error", wx.OK | wx.ICON_ERROR)
            return 
        if data not in self.target_positions:
            self.target_positions.append(data)
        self.update_pos_plot()

    def undo_target_position(self, event):
        if len(self.target_positions) > 0:
            self.target_positions.pop()
            self.update_pos_plot() 

    def calculate(self, event):
        try:
            rp = algorithms.RUP_Solver(self.particle_positions, self.target_positions)
            self.matching = rp.solve()
            self.update_line_plot()
        except ValueError as e:
            wx.MessageBox("Fehler: " + str(e), "Error", wx.OK | wx.ICON_ERROR) 

    def simulate(self, event):
        return 

    def update_pos_plot(self):
        self.figure.set_canvas(self.canvas)
        self.subplot.clear()
        data = tuple(zip(*self.particle_positions))
        if len(data) == 0:
            data = ([], [])
        self.subplot.scatter(list(data[0]), list(data[1]))
        data2 = tuple(zip(*self.target_positions))
        if len(data2) == 0:
            data2 = ([], [])
        self.subplot.scatter(list(data2[0]), list(data2[1]))
        self.canvas.draw()

    def update_line_plot(self):
        for i in range(len(self.matching[0])):
            data = tuple(zip(self.matching[0][i], self.matching[1][i]))
            self.subplot.plot(data[0], data[1], "g-")
        self.canvas.draw()


class GraphRedistributePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.add_canvas_with_toolbar()
        self.add_controls()

        self.SetSizer(self.sizer)
        self.Fit()

        self.vertices = []
        self.edges = []
        self.capacity = 0

    def add_canvas_with_toolbar(self):
        self.plot_panel = wx.Panel(self)
        self.plot_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.figure = plt.Figure(figsize=(8, 8))
        self.subplot = self.figure.add_subplot(111)
        plt.title("Title")
        self.canvas = FigureCanvas(self.plot_panel, -1, self.figure)  
        self.plot_panel_sizer.Add(self.canvas, 1, wx.EXPAND, 5)
        self.plot_panel.SetSizer(self.plot_panel_sizer)
        #toolbar = NavigationToolbar2Wx(self.canvas)
        #toolbar.Realize()
        #self.plot_panel_sizer.Add(toolbar, 1, wx.SHAPED, 5)
        #toolbar.update() 
        self.sizer.Add(self.plot_panel, 1, wx.EXPAND, 5)

    def add_controls(self): 
        panel = wx.Panel(self) 
        # node inputs
        l1 = wx.StaticText(panel, -1, "Neuer Knoten \"s,t\"") 
        self.vertex_input = wx.TextCtrl(panel) 
        create_btn = wx.Button(panel, wx.ID_ANY, label="Hinzufügen")
        create_btn.Bind(wx.EVT_BUTTON, self.create_vertex)
        undo_btn = wx.Button(panel, wx.ID_ANY, label="Rückgängig")
        undo_btn.Bind(wx.EVT_BUTTON, self.undo_vertex)
        # edge inputs
        l2 = wx.StaticText(panel, -1, "Neue Kante \"u,v\"") 
        self.edge_input = wx.TextCtrl(panel) 
        create_btn2 = wx.Button(panel, wx.ID_ANY, label="Hinzufügen")
        create_btn2.Bind(wx.EVT_BUTTON, self.create_edge)
        undo_btn2 = wx.Button(panel, wx.ID_ANY, label="Rückgängig")
        undo_btn2.Bind(wx.EVT_BUTTON, self.undo_edge)
        # control buttons
        calculate_btn = wx.Button(panel, wx.ID_ANY, label="MUP Lösen")
        calculate_btn.Bind(wx.EVT_BUTTON, self.calculate)
        height_btn = wx.Button(panel, wx.ID_ANY, label="h(t) anzeigen")
        height_btn.Bind(wx.EVT_BUTTON, self.show_ht_funcs)
        # zmup extension 
        l3 = wx.StaticText(panel, -1, "Kantenkapazität \"c\"") # TODO capacity for every edge
        self.c_input = wx.TextCtrl(panel) 
        create_btn3 = wx.Button(panel, wx.ID_ANY, label="Setzen")
        create_btn3.Bind(wx.EVT_BUTTON, self.set_capacity)
        height_btn2 = wx.Button(panel, wx.ID_ANY, label="ZMUP lösen")
        height_btn2.Bind(wx.EVT_BUTTON, self.show_ht_func_zmup)
        # add to sizer
        vbox = wx.BoxSizer(wx.VERTICAL)  
        vbox.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(self.vertex_input, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(create_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(undo_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.AddSpacer(4)
        vbox.Add(l2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(self.edge_input, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(create_btn2, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(undo_btn2, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.AddSpacer(4)
        vbox.Add(calculate_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(height_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.AddSpacer(4)
        vbox.Add(l3, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(self.c_input, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(create_btn3, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(height_btn2, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        panel.SetSizer(vbox) 
        self.sizer.Add(panel)

    def create_vertex(self, event):  
        try:
            data = tuple(map(float, self.vertex_input.GetValue().split(',')))
            vertex = algorithms.GRP_Vertex(len(self.vertices), *data)  # name is its own position
            self.vertices.append(vertex)
        except:
            wx.MessageBox("Fehler: Eingabe muss das Format \"s,t\" haben", "Error", wx.OK | wx.ICON_ERROR)
            return 
        self.update_graph_plot()

    def undo_vertex(self, event):
        if len(self.vertices) > 0:
            self.vertices.pop()
        self.update_graph_plot() # TOdO: don't remove vertex while edges still exist

    def create_edge(self, event): 
        try:
            data = tuple(map(int, self.edge_input.GetValue().split(',')))
            edge = algorithms.GRP_Edge(self.vertices[data[0]], self.vertices[data[1]])
            if edge not in self.edges:
                self.edges.append(edge)        
        except:
            wx.MessageBox("Fehler: Eingabe muss das Format \"u,v\" haben " + \
                "und Knoten müssen existieren", "Error", wx.OK | wx.ICON_ERROR)
            return   
        self.update_graph_plot() 

    def undo_edge(self, event):
        if len(self.edges) > 0:
            self.edges.pop()
        self.update_graph_plot()

    def set_capacity(self, event):
        try: 
            self.capacity = float(self.c_input.GetValue())
        except:
            wx.MessageBox("Fehler: Eingabe muss das Format \"c\" (float) haben", "Error", wx.OK | wx.ICON_ERROR)
            return  

    def calculate(self, event):
        self.network = algorithms.GRP_Network() 
        self.network.vertices = self.vertices
        self.network.edges = self.edges
        self.flows = [0.0 for _ in range(len(self.edges))]
        try:
            self.flows = algorithms.GRP_Solver.solve_MUP(self.network)
        except ValueError as e:
            wx.MessageBox("Fehler: " + str(e), "Error", wx.OK | wx.ICON_ERROR) 
        l = {}
        for i in range(len(self.edges)):
            l[(str(self.edges[i].vertexU), str(self.edges[i].vertexV))] = str(self.flows[i])
        nx.draw_networkx_edge_labels(self.nx_graph, pos=self.graph_layout, ax=self.subplot, 
            edge_labels=l, font_size=8, label_pos=0.25)    
        self.canvas.draw()

    def show_ht_funcs(self, event): # call this after "calculate"
        frame = HtFunctionPlotFrame(self.flows, self.network) 

    def show_ht_func_zmup(self, event): 
        zmup_flows, end = algorithms.GRP_Solver.solve_ZMUP(self.network, 
            [self.capacity for _ in range(len(self.edges))])
        frame = HtFunctionPlotFrame(zmup_flows, self.network, t_end=end)

    def update_graph_plot(self):
        self.subplot.clear()
        self.nx_graph = nx.DiGraph()
        V = [str(v) for v in self.vertices]
        self.nx_graph.add_nodes_from(V)
        E = [(str(e.vertexU), str(e.vertexV)) for e in self.edges]
        self.nx_graph.add_edges_from(E)
        self.graph_layout = nx.spring_layout(self.nx_graph, scale=1)
        self.subplot.set_xlim([-1.25,1.25])
        self.subplot.set_ylim([-1.25,1.25])
        nx.draw(self.nx_graph, pos=self.graph_layout, ax=self.subplot, node_size=150)
        nx.draw_networkx_labels(self.nx_graph, pos=self.graph_layout, ax=self.subplot, 
            labels={v: v for v in V}, font_size=10, font_weight="bold")
        self.canvas.draw()
     

class HtFunctionPlotFrame(wx.Frame):
    def __init__(self, flows, network, parent=None, t_end=1):
        wx.Frame.__init__(self, parent=parent, title="h_v(t) Plots")
        self.Show()
        self.flows = flows
        self.network = network
        self.t_end = t_end
        self.setup_plot()
        self.draw_plot()

    def setup_plot(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.figure = plt.Figure(figsize=(4, 4))
        self.subplot = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)  
        self.sizer.Add(self.canvas, 1, wx.EXPAND, 5)
    
    def draw_plot(self):
        self.subplot.clear()
        starting_values = algorithms.np.array([v.start_value for v in self.network.vertices])
        flow_delta_vertices = algorithms.np.zeros(len(self.network.vertices))
        for i in range(len(self.network.edges)):
            flow_delta_vertices[self.network.edges[i].vertexU.name] -= self.flows[i]
            flow_delta_vertices[self.network.edges[i].vertexV.name] += self.flows[i]
        time = algorithms.np.linspace(0, self.t_end, 100)
        for i in range(len(self.network.vertices)):
            ht_v = time * flow_delta_vertices[i] + starting_values[i]
            self.subplot.plot(time, ht_v, label="h_{}(t)".format(i))
        self.subplot.set_xlim((0,1))
        self.subplot.legend()
        self.canvas.draw()


class GridGraphPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.add_canvas_with_toolbar()
        self.add_controls()

        self.SetSizer(self.sizer)
        self.Fit()

        self.capacity = 0
        self.start_values = []
        self.current_values = []
        self.target_values = []
        self.process = []

    def add_canvas_with_toolbar(self):
        self.plot_panel = wx.Panel(self)
        self.plot_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.figure = plt.Figure(figsize=(8, 8))
        self.subplot = self.figure.add_subplot(111)
        plt.title("Title")
        self.canvas = FigureCanvas(self.plot_panel, -1, self.figure)  
        self.plot_panel_sizer.Add(self.canvas, 1, wx.EXPAND, 5)
        self.plot_panel.SetSizer(self.plot_panel_sizer)
        #toolbar = NavigationToolbar2Wx(self.canvas)
        #toolbar.Realize()
        #self.plot_panel_sizer.Add(toolbar, 1, wx.SHAPED, 5)
        #toolbar.update() 
        self.sizer.Add(self.plot_panel, 1, wx.EXPAND, 5)

    def add_controls(self): 
        panel = wx.Panel(self) 

        l1 = wx.StaticText(panel, -1, "Gitter-Abmessungen \"N,M\"") 
        self.dimension_input = wx.TextCtrl(panel) 
        dimension_btn = wx.Button(panel, wx.ID_ANY, label="Setzen")
        dimension_btn.Bind(wx.EVT_BUTTON, self.init_grid)

        l2 = wx.StaticText(panel, -1, "Zellwert \"x,y,start,ziel\"") 
        self.value_input = wx.TextCtrl(panel) 
        value_btn = wx.Button(panel, wx.ID_ANY, label="Setzen")
        value_btn.Bind(wx.EVT_BUTTON, self.set_value)

        l3 = wx.StaticText(panel, -1, "k_max:") 
        self.capacity_input = wx.TextCtrl(panel) 
        capacity_btn = wx.Button(panel, wx.ID_ANY, label="Setzen")
        capacity_btn.Bind(wx.EVT_BUTTON, self.set_capacity)

        step_simulation_btn = wx.Button(panel, wx.ID_ANY, label="SZMUP simulieren")
        step_simulation_btn.Bind(wx.EVT_BUTTON, self.step_simulation)
        next_step_btn = wx.Button(panel, wx.ID_ANY, label="Nächster Schritt")
        next_step_btn.Bind(wx.EVT_BUTTON, self.next_step)
        continuous_btn = wx.Button(panel, wx.ID_ANY, label="MUP stetig Simulieren")
        continuous_btn.Bind(wx.EVT_BUTTON, self.continuous_simulation)

        # add to sizer
        vbox = wx.BoxSizer(wx.VERTICAL)  
        vbox.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(self.dimension_input, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(dimension_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(l2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(self.value_input, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(value_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.AddSpacer(4)
        vbox.Add(l3, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        vbox.Add(self.capacity_input, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5) 
        vbox.Add(capacity_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(step_simulation_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.Add(next_step_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        vbox.AddSpacer(4)
        vbox.Add(continuous_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        panel.SetSizer(vbox) 
        self.sizer.Add(panel)

    def init_grid(self, event):
        try:
            N, M = self.dimension_input.GetValue().split(",")
            self.start_values = algorithms.np.zeros((int(N), int(M)))
            self.target_values = algorithms.np.zeros((int(N), int(M)))
        except:
            wx.MessageBox("Fehler: Eingabe muss das Format \"N,M\" haben", "Error", wx.OK | wx.ICON_ERROR)
            return 
        self.current_values = self.start_values
        self.update_grid_plot()

    def set_value(self, event):
        try:
            x, y, start, target = self.value_input.GetValue().split(",")
            self.start_values[int(y),int(x)] = float(start)
            self.target_values[int(y),int(x)] = float(target)
        except:
            wx.MessageBox("Fehler: Eingabe muss das Format \"x,y,start,ziel\" haben (Index startet bei 0)", 
                "Error", wx.OK | wx.ICON_ERROR)
            return 
        self.current_values = self.start_values
        self.update_grid_plot()

    def set_capacity(self, event):
        try:
            self.capacity = int(self.capacity_input.GetValue())
        except:
            wx.MessageBox("Fehler: Kapazität muss eine natürliche Zahl größer 0 sein", "Error", 
                wx.OK | wx.ICON_ERROR)
            return 
        self.current_values = self.start_values
        self.update_grid_plot()

    def step_simulation(self, event):
        self.process = algorithms.GRP_Solver.solve_SZUP(self.start_values, 
            self.target_values, self.capacity)
        self.current_values = self.process.pop(0)

    def next_step(self, event):
        if len(self.process) == 0:
            return 
        self.current_values = self.process.pop(0)
        self.update_grid_plot()

    def continuous_simulation(self, event):
        return 

    def update_grid_plot(self):
        self.subplot.clear()
        im = self.subplot.imshow(self.current_values)
        for y in range(self.current_values.shape[0]):
            for x in range(self.current_values.shape[1]):
                text = self.subplot.text(x, y, "{} / {}".format(self.current_values[y,x], self.target_values[y,x]), 
                    ha="center", va="center", color="w")
        self.canvas.draw()


class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'JuFo 2020', size=(800, 800))
        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.create_menu()
        # startup panel is rearrengement panel
        self.particle_rearregement_panel = ParticleRearrengementPanel(self)
        self.current_panel = self.particle_rearregement_panel
        self.main_sizer.Add(self.current_panel)

        self.SetSizer(self.main_sizer)
        self.Centre()
        self.Fit()

    def create_menu(self):
        menu_sizer = wx.BoxSizer(wx.VERTICAL)
        btn_prp = wx.Button(self, wx.ID_ANY, label="Roboter Umpositionierung")
        btn_prp.Bind(wx.EVT_BUTTON, self.on_btn_prp)
        menu_sizer.Add(btn_prp, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)

        btn_ggp = wx.Button(self, wx.ID_ANY, label="Partikel im Zellgitter")
        btn_ggp.Bind(wx.EVT_BUTTON, self.on_btn_ggp)
        menu_sizer.Add(btn_ggp, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)

        btn_grp = wx.Button(self, wx.ID_ANY, label="Graphen-MUP und -ZMUP")
        btn_grp.Bind(wx.EVT_BUTTON, self.on_btn_grp)
        menu_sizer.Add(btn_grp, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        self.main_sizer.Add(menu_sizer)

    def on_btn_prp(self, event):
        prp = ParticleRearrengementPanel(self)
        self.current_panel.Hide()
        self.main_sizer.Remove(self.main_sizer.GetItemCount() - 1)
        self.current_panel = prp
        self.main_sizer.Add(self.current_panel)
        self.Layout()

    def on_btn_grp(self, event):
        grp = GraphRedistributePanel(self)
        self.current_panel.Hide()
        self.main_sizer.Remove(self.main_sizer.GetItemCount() - 1)
        self.current_panel = grp
        self.main_sizer.Add(self.current_panel)
        self.Layout()

    def on_btn_ggp(self, event):
        ggp = GridGraphPanel(self)
        self.current_panel.Hide()
        self.main_sizer.Remove(self.main_sizer.GetItemCount() - 1)
        self.current_panel = ggp
        self.main_sizer.Add(self.current_panel)
        self.Layout() 


def exception_handler(etype, value, trace):
    frame = wx.GetApp().GetTopWindow()
    tmp = traceback.format_exception(etype, value, trace)
    exception = "".join(tmp)
    wx.MessageBox(exception, "Fehler!", wx.OK | wx.ICON_ERROR)

    
class App(WIT.InspectableApp):
    def OnInit(self):
        self.Init()
        frame = MainFrame()
        sys.excepthook = exception_handler
        frame.Show(True)
        return True