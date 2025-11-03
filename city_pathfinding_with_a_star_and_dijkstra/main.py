import heapq
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import threading

class CityMap:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.nodes = {}
        self.edges = []
        self.obstacles = []
        self.generate_map()
    
    def generate_map(self):
        """Generate a grid-based city map with roads and obstacles"""
        # Create nodes in a grid pattern
        node_id = 0
        for x in range(self.width):
            for y in range(self.height):
                self.nodes[node_id] = (x, y)
                node_id += 1
        
        # Create edges (roads) between adjacent nodes
        for node_id, (x, y) in self.nodes.items():
            # Connect to right neighbor
            if x < self.width - 1:
                right_id = node_id + 1
                distance = 1.0
                # Add some random variation to distances
                distance += np.random.uniform(0, 0.3)
                self.edges.append((node_id, right_id, distance))
            
            # Connect to bottom neighbor
            if y < self.height - 1:
                bottom_id = node_id + self.width
                distance = 1.0
                distance += np.random.uniform(0, 0.3)
                self.edges.append((node_id, bottom_id, distance))
        
        # Add some random obstacles (buildings, parks, etc.)
        num_obstacles = int(self.width * self.height * 0.1)  # 10% of map
        for _ in range(num_obstacles):
            obs_x = np.random.randint(0, self.width)
            obs_y = np.random.randint(0, self.height)
            self.obstacles.append((obs_x, obs_y))
            
            # Remove edges that connect to obstacle nodes
            obstacle_id = obs_y * self.width + obs_x
            self.edges = [edge for edge in self.edges 
                         if obstacle_id not in (edge[0], edge[1])]
    
    def get_neighbors(self, node_id):
        """Get all neighbors of a node with their distances"""
        neighbors = []
        for edge in self.edges:
            if edge[0] == node_id:
                neighbors.append((edge[1], edge[2]))
            elif edge[1] == node_id:
                neighbors.append((edge[0], edge[2]))
        return neighbors
    
    def euclidean_distance(self, node1_id, node2_id):
        """Calculate Euclidean distance between two nodes"""
        x1, y1 = self.nodes[node1_id]
        x2, y2 = self.nodes[node2_id]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def manhattan_distance(self, node1_id, node2_id):
        """Calculate Manhattan distance between two nodes"""
        x1, y1 = self.nodes[node1_id]
        x2, y2 = self.nodes[node2_id]
        return abs(x2 - x1) + abs(y2 - y1)
    
    def clear_paths(self):
        """Clear any existing paths (for GUI)"""
        # This method is for future extension if we want to store paths in the map
        pass

class Pathfinder:
    def __init__(self, city_map):
        self.city_map = city_map
    
    def dijkstra(self, start_id, goal_id):
        """Implement Dijkstra's algorithm for shortest path"""
        start_time = time.time()
        
        # Initialize distances and predecessors
        distances = {node_id: float('inf') for node_id in self.city_map.nodes}
        predecessors = {node_id: None for node_id in self.city_map.nodes}
        distances[start_id] = 0
        
        # Priority queue: (distance, node_id)
        priority_queue = [(0, start_id)]
        nodes_explored = 0
        
        while priority_queue:
            current_distance, current_id = heapq.heappop(priority_queue)
            nodes_explored += 1
            
            # Early exit if we reached the goal
            if current_id == goal_id:
                break
            
            # Skip if we found a better path to this node already
            if current_distance > distances[current_id]:
                continue
            
            # Explore neighbors
            for neighbor_id, edge_distance in self.city_map.get_neighbors(current_id):
                distance = current_distance + edge_distance
                
                # If we found a shorter path to the neighbor
                if distance < distances[neighbor_id]:
                    distances[neighbor_id] = distance
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(priority_queue, (distance, neighbor_id))
        
        # Reconstruct path
        path = []
        if distances[goal_id] < float('inf'):  # Path exists
            current_id = goal_id
            while current_id is not None:
                path.append(current_id)
                current_id = predecessors[current_id]
            path.reverse()
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        return path, distances[goal_id], nodes_explored, computation_time
    
    def a_star(self, start_id, goal_id, heuristic='euclidean'):
        """Implement A* algorithm for shortest path with heuristic"""
        start_time = time.time()
        
        # Initialize distances and predecessors
        g_scores = {node_id: float('inf') for node_id in self.city_map.nodes}
        predecessors = {node_id: None for node_id in self.city_map.nodes}
        g_scores[start_id] = 0
        
        # Choose heuristic function
        if heuristic == 'euclidean':
            heuristic_func = self.city_map.euclidean_distance
        else:
            heuristic_func = self.city_map.manhattan_distance
        
        # Priority queue: (f_score, node_id)
        f_scores = {node_id: float('inf') for node_id in self.city_map.nodes}
        f_scores[start_id] = heuristic_func(start_id, goal_id)
        
        priority_queue = [(f_scores[start_id], start_id)]
        nodes_explored = 0
        
        while priority_queue:
            current_f_score, current_id = heapq.heappop(priority_queue)
            nodes_explored += 1
            
            # Early exit if we reached the goal
            if current_id == goal_id:
                break
            
            # Skip if we found a better path to this node already
            if current_f_score > f_scores[current_id]:
                continue
            
            # Explore neighbors
            for neighbor_id, edge_distance in self.city_map.get_neighbors(current_id):
                tentative_g_score = g_scores[current_id] + edge_distance
                
                # If we found a shorter path to the neighbor
                if tentative_g_score < g_scores[neighbor_id]:
                    g_scores[neighbor_id] = tentative_g_score
                    f_scores[neighbor_id] = tentative_g_score + heuristic_func(neighbor_id, goal_id)
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(priority_queue, (f_scores[neighbor_id], neighbor_id))
        
        # Reconstruct path
        path = []
        if g_scores[goal_id] < float('inf'):  # Path exists
            current_id = goal_id
            while current_id is not None:
                path.append(current_id)
                current_id = predecessors[current_id]
            path.reverse()
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        return path, g_scores[goal_id], nodes_explored, computation_time

class CityPathfindingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("City Pathfinding - A* vs Dijkstra")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.city_map = None
        self.pathfinder = None
        self.start_id = None
        self.goal_id = None
        self.path_dijkstra = None
        self.path_astar = None
        self.fig = None
        self.ax = None
        self.canvas = None
        
        self.setup_gui()
        self.generate_map()
    
    def setup_gui(self):
        """Setup the GUI elements"""
        # Create main frames
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        plot_frame = ttk.Frame(self.root, padding="10")
        plot_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Control frame widgets
        ttk.Label(control_frame, text="City Pathfinding", font=('Arial', 16, 'bold')).grid(row=0, column=0, columnspan=4, pady=(0, 10))
        
        # Map settings
        ttk.Label(control_frame, text="Map Width:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.width_var = tk.StringVar(value="20")
        ttk.Entry(control_frame, textvariable=self.width_var, width=5).grid(row=1, column=1, sticky=tk.W, padx=(0, 10))
        
        ttk.Label(control_frame, text="Map Height:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5))
        self.height_var = tk.StringVar(value="20")
        ttk.Entry(control_frame, textvariable=self.height_var, width=5).grid(row=1, column=3, sticky=tk.W, padx=(0, 10))
        
        ttk.Button(control_frame, text="Generate Map", command=self.generate_map).grid(row=1, column=4, padx=(10, 0))
        
        # Start and goal selection
        ttk.Label(control_frame, text="Start Node ID:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        self.start_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.start_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        
        ttk.Label(control_frame, text="Goal Node ID:").grid(row=2, column=2, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        self.goal_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.goal_var, width=5).grid(row=2, column=3, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        
        ttk.Button(control_frame, text="Set Start/Goal", command=self.set_start_goal).grid(row=2, column=4, padx=(10, 0), pady=(10, 0))
        ttk.Button(control_frame, text="Random Points", command=self.random_points).grid(row=2, column=5, padx=(10, 0), pady=(10, 0))
        
        # Algorithm controls
        ttk.Label(control_frame, text="Heuristic:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        self.heuristic_var = tk.StringVar(value="euclidean")
        heuristic_combo = ttk.Combobox(control_frame, textvariable=self.heuristic_var, 
                                      values=["euclidean", "manhattan"], state="readonly", width=10)
        heuristic_combo.grid(row=3, column=1, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        
        ttk.Button(control_frame, text="Run Dijkstra", command=lambda: self.run_algorithm("dijkstra")).grid(row=3, column=2, padx=(0, 5), pady=(10, 0))
        ttk.Button(control_frame, text="Run A*", command=lambda: self.run_algorithm("astar")).grid(row=3, column=3, padx=(0, 5), pady=(10, 0))
        ttk.Button(control_frame, text="Compare Both", command=lambda: self.run_algorithm("both")).grid(row=3, column=4, padx=(0, 5), pady=(10, 0))
        ttk.Button(control_frame, text="Clear Paths", command=self.clear_paths).grid(row=3, column=5, padx=(0, 5), pady=(10, 0))
        
        # Results display
        self.results_text = tk.Text(control_frame, height=8, width=80)
        self.results_text.grid(row=4, column=0, columnspan=6, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Create scrollbar for results text
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=4, column=6, pady=(10, 0), sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Plot frame - for matplotlib figure
        self.setup_plot(plot_frame)
    
    def setup_plot(self, parent):
        """Setup the matplotlib plot in the GUI"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Connect click event for selecting nodes
        self.canvas.mpl_connect('button_press_event', self.on_click)
    
    def on_click(self, event):
        """Handle mouse clicks on the plot to select start and goal"""
        if event.inaxes != self.ax:
            return
        
        # Find the closest node to the click
        min_dist = float('inf')
        closest_node = None
        
        for node_id, (x, y) in self.city_map.nodes.items():
            dist = math.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node_id
        
        if closest_node is not None:
            # Check if it's an obstacle
            node_coords = self.city_map.nodes[closest_node]
            if node_coords in self.city_map.obstacles:
                messagebox.showwarning("Obstacle", "Cannot select an obstacle node!")
                return
            
            # Set as start or goal based on current selection
            if self.start_id is None:
                self.start_id = closest_node
                self.start_var.set(str(closest_node))
                messagebox.showinfo("Start Set", f"Start node set to {closest_node} at {node_coords}")
            elif self.goal_id is None:
                self.goal_id = closest_node
                self.goal_var.set(str(closest_node))
                messagebox.showinfo("Goal Set", f"Goal node set to {closest_node} at {node_coords}")
            else:
                # Ask user which to replace
                choice = messagebox.askquestion("Replace Node", 
                                              "Both start and goal are set. Replace which one?",
                                              icon='question')
                if choice == 'yes':
                    self.start_id = closest_node
                    self.start_var.set(str(closest_node))
                    messagebox.showinfo("Start Set", f"Start node set to {closest_node} at {node_coords}")
                else:
                    self.goal_id = closest_node
                    self.goal_var.set(str(closest_node))
                    messagebox.showinfo("Goal Set", f"Goal node set to {closest_node} at {node_coords}")
            
            self.visualize_map()
    
    def generate_map(self):
        """Generate a new city map"""
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            
            if width < 5 or height < 5:
                messagebox.showerror("Invalid Size", "Map dimensions must be at least 5x5")
                return
            if width > 50 or height > 50:
                messagebox.showerror("Invalid Size", "Map dimensions cannot exceed 50x50")
                return
                
            self.city_map = CityMap(width, height)
            self.pathfinder = Pathfinder(self.city_map)
            self.start_id = None
            self.goal_id = None
            self.path_dijkstra = None
            self.path_astar = None
            self.start_var.set("")
            self.goal_var.set("")
            
            self.visualize_map()
            self.add_to_results("New map generated successfully.\n")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integers for map dimensions")
    
    def set_start_goal(self):
        """Set start and goal from text inputs"""
        try:
            start_id = int(self.start_var.get())
            goal_id = int(self.goal_var.get())
            
            if start_id not in self.city_map.nodes:
                messagebox.showerror("Invalid Node", f"Start node {start_id} does not exist")
                return
            if goal_id not in self.city_map.nodes:
                messagebox.showerror("Invalid Node", f"Goal node {goal_id} does not exist")
                return
            
            # Check if nodes are obstacles
            start_coords = self.city_map.nodes[start_id]
            goal_coords = self.city_map.nodes[goal_id]
            
            if start_coords in self.city_map.obstacles:
                messagebox.showerror("Invalid Node", f"Start node {start_id} is an obstacle")
                return
            if goal_coords in self.city_map.obstacles:
                messagebox.showerror("Invalid Node", f"Goal node {goal_id} is an obstacle")
                return
            
            self.start_id = start_id
            self.goal_id = goal_id
            
            self.visualize_map()
            self.add_to_results(f"Start set to node {start_id}, Goal set to node {goal_id}\n")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integers for node IDs")
    
    def random_points(self):
        """Set random start and goal points"""
        if not self.city_map:
            messagebox.showerror("No Map", "Please generate a map first")
            return
        
        # Get all non-obstacle nodes
        valid_nodes = [node_id for node_id, coords in self.city_map.nodes.items() 
                      if coords not in self.city_map.obstacles]
        
        if len(valid_nodes) < 2:
            messagebox.showerror("No Valid Points", "Not enough valid nodes for start and goal")
            return
        
        # Select random start and goal
        np.random.shuffle(valid_nodes)
        self.start_id = valid_nodes[0]
        self.goal_id = valid_nodes[1]
        
        self.start_var.set(str(self.start_id))
        self.goal_var.set(str(self.goal_id))
        
        self.visualize_map()
        self.add_to_results(f"Random points set: Start={self.start_id}, Goal={self.goal_id}\n")
    
    def run_algorithm(self, algorithm):
        """Run the selected pathfinding algorithm"""
        if not self.city_map:
            messagebox.showerror("No Map", "Please generate a map first")
            return
        
        if self.start_id is None or self.goal_id is None:
            messagebox.showerror("Missing Points", "Please set start and goal points first")
            return
        
        # Disable buttons during computation
        self.toggle_buttons(False)
        
        # Run in a separate thread to avoid GUI freezing
        thread = threading.Thread(target=self._run_algorithm_thread, args=(algorithm,))
        thread.daemon = True
        thread.start()
    
    def _run_algorithm_thread(self, algorithm):
        """Thread function for running algorithms"""
        try:
            if algorithm == "dijkstra":
                self.path_dijkstra, dist_dijkstra, nodes_dijkstra, time_dijkstra = \
                    self.pathfinder.dijkstra(self.start_id, self.goal_id)
                self.path_astar = None
                
                # Update GUI in main thread
                self.root.after(0, lambda: self._display_results(
                    "Dijkstra", dist_dijkstra, nodes_dijkstra, time_dijkstra))
                
            elif algorithm == "astar":
                heuristic = self.heuristic_var.get()
                self.path_astar, dist_astar, nodes_astar, time_astar = \
                    self.pathfinder.a_star(self.start_id, self.goal_id, heuristic)
                self.path_dijkstra = None
                
                # Update GUI in main thread
                self.root.after(0, lambda: self._display_results(
                    f"A* ({heuristic})", dist_astar, nodes_astar, time_astar))
                
            elif algorithm == "both":
                # Run both algorithms
                self.path_dijkstra, dist_dijkstra, nodes_dijkstra, time_dijkstra = \
                    self.pathfinder.dijkstra(self.start_id, self.goal_id)
                
                heuristic = self.heuristic_var.get()
                self.path_astar, dist_astar, nodes_astar, time_astar = \
                    self.pathfinder.a_star(self.start_id, self.goal_id, heuristic)
                
                # Update GUI in main thread
                self.root.after(0, lambda: self._display_comparison(
                    dist_dijkstra, nodes_dijkstra, time_dijkstra,
                    dist_astar, nodes_astar, time_astar, heuristic))
            
            # Update visualization in main thread
            self.root.after(0, self.visualize_map)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.toggle_buttons(True))
    
    def _display_results(self, algorithm, distance, nodes_explored, computation_time):
        """Display results for a single algorithm"""
        if distance == float('inf'):
            result_text = f"{algorithm}: No path found!\n"
        else:
            # Calculate travel time (assuming average speed of 30 units per time unit)
            avg_speed = 30
            travel_time = distance / avg_speed
            
            result_text = (
                f"{algorithm} Results:\n"
                f"  Path Length: {distance:.2f}\n"
                f"  Nodes Explored: {nodes_explored}\n"
                f"  Computation Time: {computation_time:.4f}s\n"
                f"  Estimated Travel Time: {travel_time:.2f} units\n\n"
            )
        
        self.add_to_results(result_text)
    
    def _display_comparison(self, dist_d, nodes_d, time_d, dist_a, nodes_a, time_a, heuristic):
        """Display comparison between Dijkstra and A*"""
        if dist_d == float('inf') or dist_a == float('inf'):
            result_text = "No path found by one or both algorithms!\n\n"
        else:
            # Calculate travel times
            avg_speed = 30
            travel_d = dist_d / avg_speed
            travel_a = dist_a / avg_speed
            
            result_text = (
                "Algorithm Comparison:\n"
                "Dijkstra:\n"
                f"  Path Length: {dist_d:.2f}\n"
                f"  Nodes Explored: {nodes_d}\n"
                f"  Computation Time: {time_d:.4f}s\n"
                f"  Estimated Travel Time: {travel_d:.2f} units\n\n"
                f"A* ({heuristic}):\n"
                f"  Path Length: {dist_a:.2f}\n"
                f"  Nodes Explored: {nodes_a}\n"
                f"  Computation Time: {time_a:.4f}s\n"
                f"  Estimated Travel Time: {travel_a:.2f} units\n\n"
            )
        
        self.add_to_results(result_text)
    
    def clear_paths(self):
        """Clear the current paths"""
        self.path_dijkstra = None
        self.path_astar = None
        self.visualize_map()
        self.add_to_results("Paths cleared.\n")
    
    def toggle_buttons(self, enabled):
        """Enable or disable control buttons"""
        # This would need to be implemented to disable buttons during computation
        # For simplicity, we'll just update the cursor
        if enabled:
            self.root.config(cursor="")
        else:
            self.root.config(cursor="watch")
        self.root.update()
    
    def add_to_results(self, text):
        """Add text to the results display"""
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def visualize_map(self):
        """Visualize the city map with current paths"""
        if not self.city_map:
            return
        
        self.ax.clear()
        
        # Draw the grid
        for node_id, (x, y) in self.city_map.nodes.items():
            self.ax.plot(x, y, 'ko', markersize=3, alpha=0.5)
        
        # Draw edges (roads)
        for edge in self.city_map.edges:
            node1_id, node2_id, _ = edge
            x1, y1 = self.city_map.nodes[node1_id]
            x2, y2 = self.city_map.nodes[node2_id]
            self.ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.7)
        
        # Draw obstacles
        for obs_x, obs_y in self.city_map.obstacles:
            self.ax.add_patch(patches.Rectangle((obs_x-0.5, obs_y-0.5), 1, 1, 
                                              facecolor='red', alpha=0.3))
        
        # Draw paths if they exist
        if self.path_dijkstra and len(self.path_dijkstra) > 0:
            x_coords = [self.city_map.nodes[node_id][0] for node_id in self.path_dijkstra]
            y_coords = [self.city_map.nodes[node_id][1] for node_id in self.path_dijkstra]
            self.ax.plot(x_coords, y_coords, 'b-', linewidth=3, label='Dijkstra Path')
        
        if self.path_astar and len(self.path_astar) > 0:
            x_coords = [self.city_map.nodes[node_id][0] for node_id in self.path_astar]
            y_coords = [self.city_map.nodes[node_id][1] for node_id in self.path_astar]
            self.ax.plot(x_coords, y_coords, 'g-', linewidth=2, label='A* Path')
        
        # Mark start and goal
        if self.start_id:
            start_x, start_y = self.city_map.nodes[self.start_id]
            self.ax.plot(start_x, start_y, 'go', markersize=10, label='Start')
        
        if self.goal_id:
            goal_x, goal_y = self.city_map.nodes[self.goal_id]
            self.ax.plot(goal_x, goal_y, 'ro', markersize=10, label='Goal')
        
        self.ax.set_xlim(-1, self.city_map.width)
        self.ax.set_ylim(-1, self.city_map.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('City Map with Pathfinding')
        self.ax.legend()
        
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = CityPathfindingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()