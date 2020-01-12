from algorithms import *

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
            rp = RearrengementProblem(self.particle_positions, self.target_positions)
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
        calculate_btn = wx.Button(panel, wx.ID_ANY, label="Berechnen")
        calculate_btn.Bind(wx.EVT_BUTTON, self.calculate)
        simulate_btn = wx.Button(panel, wx.ID_ANY, label="Simulieren")
        simulate_btn.Bind(wx.EVT_BUTTON, self.simulate)
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
        vbox.Add(simulate_btn, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        panel.SetSizer(vbox) 
        self.sizer.Add(panel)

    def create_vertex(self, event):  
        try:
            data = tuple(map(float, self.vertex_input.GetValue().split(',')))
            vertex = GRP_Vertex(len(self.vertices), *data)  # name is its own position
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
            edge = GRP_Edge(self.vertices[data[0]], self.vertices[data[1]])
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

    def calculate(self, event):
        network = GRP_Network() 
        network.vertices = self.vertices
        network.edges = self.edges
        flows = GRP_Solver(network).solve()
        l = {e: str(flows[i]) for i, e in enumerate(self.nx_graph.edges())}
        nx.draw_networkx_edge_labels(self.nx_graph, pos=self.graph_layout, ax=self.subplot, 
            edge_labels=l, font_size=8, label_pos=0.75)    
        self.canvas.draw()

    def simulate(self, event):
        return 

    def update_graph_plot(self):
        self.subplot.clear()
        self.nx_graph = nx.DiGraph()
        V = [str(v) for v in self.vertices]
        self.nx_graph.add_nodes_from(V)
        E = [(str(e.vertexU), str(e.vertexV)) for e in self.edges]
        self.nx_graph.add_edges_from(E)
        self.graph_layout = nx.kamada_kawai_layout(self.nx_graph, scale=1)
        self.subplot.set_xlim([-1.25,1.25])
        self.subplot.set_ylim([-1.25,1.25])
        nx.draw(self.nx_graph, pos=self.graph_layout, ax=self.subplot, node_size=150)
        nx.draw_networkx_labels(self.nx_graph, pos=self.graph_layout, ax=self.subplot, 
            labels={v: v for v in V}, font_size=10, font_weight="bold")
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
        btn_prp = wx.Button(self, wx.ID_ANY, label="Partikel-Umordnungsproblem")
        btn_prp.Bind(wx.EVT_BUTTON, self.on_btn_prp)
        menu_sizer.Add(btn_prp, 1, wx.EXPAND | wx.ALIGN_LEFT, 5)
        btn_grp = wx.Button(self, wx.ID_ANY, label="Graphen-Umverteilungsproblem")
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

    
class App(WIT.InspectableApp):
    def OnInit(self):
        self.Init()
        frame = MainFrame()
        frame.Show(True)
        return True