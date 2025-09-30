import gymnasium as gym
import numpy as np
import networkx as nx
from track_layout import create_railway_network

class Train:
    """
    Represents a single train in the simulation.
    ADDED: Metrics tracking attributes.
    """
    def __init__(self, train_id, initial_node, destination_node, speed=120):
        self.id = train_id
        self.speed_kph = speed
        self.current_node = initial_node
        self.destination_node = destination_node
        self.next_node = None
        self.distance_to_next_node = 0.0
        self.state = "stopped"  # Can be "stopped", "moving", "finished"
        
        # --- NEW METRICS ---
        self.start_time = 0
        self.end_time = -1 # -1 means not finished yet
        self.wait_time_steps = 0
        self.journey_time_steps = 0

    def __repr__(self):
        return (f"Train(id={self.id}, current={self.current_node}, "
                f"next={self.next_node}, dist_rem={self.distance_to_next_node:.1f}km, "
                f"state={self.state})")

class TrainTrafficEnv(gym.Env):
    """
    The main simulation environment for train traffic control.
    """
    def __init__(self):
        super(TrainTrafficEnv, self).__init__()
        self.rail_network = create_railway_network()
        self.time_step_seconds = 60
        self.train_configs = {
            "T01": {"start": "S_A", "dest": "S_B"},
            "T02": {"start": "S_B", "dest": "S_A"}
        }
        self.trains = {} # Will be populated in reset
        
        self.action_space = gym.spaces.MultiDiscrete([2] * len(self.train_configs))
        self.observation_space = gym.spaces.Box(low=0, high=len(self.rail_network.nodes), 
                                                shape=(len(self.train_configs),), dtype=np.int32)
        
        self.node_list = list(self.rail_network.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # --- MODIFIED: Train creation and metric reset ---
        self.trains = {
            tid: Train(tid, conf["start"], conf["dest"]) 
            for tid, conf in self.train_configs.items()
        }
        for train in self.trains.values():
            train.start_time = self.current_step
        
        # print("\n--- Environment Reset ---") # Optional: uncomment for verbose debugging
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """Gets the current observation of the environment state."""
        obs = [self.node_to_idx[train.current_node] for train in self.trains.values()]
        return np.array(obs, dtype=np.int32)

    def _get_info(self):
        """
        Gets auxiliary information, including our new metrics.
        """
        finished_trains = [t for t in self.trains.values() if t.state == "finished"]
        
        total_journey_time = 0
        if finished_trains:
            total_journey_time = sum(t.journey_time_steps for t in finished_trains)
        
        return {
            "train_details": {train.id: train.__dict__ for train in self.trains.values()},
            "metrics": {
                "throughput": len(finished_trains),
                "avg_journey_time": (total_journey_time / len(finished_trains)) if finished_trains else 0,
                "total_wait_time": sum(t.wait_time_steps for t in self.trains.values())
            }
        }

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        self.current_step += 1
        reward = 0
        terminated = False
        
        train_actions = {tid: act for tid, act in zip(self.trains.keys(), action)}

        for train_id, train in self.trains.items():
            if train.state == "finished":
                continue

            # --- METRIC UPDATE: Increment journey time ---
            train.journey_time_steps += 1

            if train_actions[train_id] == 0: # Action: Hold
                train.state = "stopped"
                # --- METRIC UPDATE: Increment wait time if stopped ---
                train.wait_time_steps += 1
            elif train_actions[train_id] == 1 and train.state == "stopped": # Action: Proceed
                try:
                    path = nx.shortest_path(self.rail_network, 
                                            source=train.current_node, 
                                            target=train.destination_node, 
                                            weight='weight')
                    if len(path) > 1:
                        train.next_node = path[1]
                        train.distance_to_next_node = self.rail_network[train.current_node][train.next_node]['weight']
                        train.state = "moving"
                except (nx.NetworkXNoPath, IndexError):
                    train.state = "stopped"
                    train.wait_time_steps += 1 # Also waiting if no path found
            
            if train.state == "moving":
                distance_covered = (train.speed_kph / 3600) * self.time_step_seconds
                train.distance_to_next_node -= distance_covered
                
                if train.distance_to_next_node <= 0:
                    train.current_node = train.next_node
                    train.next_node = None
                    train.state = "stopped"
                    
                    if train.current_node == train.destination_node:
                        train.state = "finished"
                        train.end_time = self.current_step
                        reward += 100 # Big reward for finishing
                        # print(f"SUCCESS: Train {train.id} reached destination!") # Optional
        
        if all(t.state == "finished" for t in self.trains.values()):
            terminated = True
        
        # --- UPDATED REWARD FUNCTION ---
        reward -= 1 # Small penalty for every time step to encourage speed
        total_wait_steps = sum(t.wait_time_steps for t in self.trains.values())
        reward -= 0.1 * total_wait_steps # Penalize inefficiency
        
        return self._get_obs(), reward, terminated, False, self._get_info()