import torch
import numpy as np
import eventlet
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import os
from transformers import pipeline

# Import our custom classes and functions
from train_env import TrainTrafficEnv
from dqn_agent import QNetwork

# --- NEW SETUP for Local AI Model ---
# This will download the model the first time you run the script.
print("Loading local text generation model (this may take a moment)...")
# Using a small, efficient model perfect for this task.
explainer = pipeline("text2text-generation", model="google/flan-t5-small")
print("Local model loaded successfully.")


def get_ai_explanation(train_info, action_array):
    """
    Generates a human-readable explanation using a locally run AI model.
    """
    # Adjusted for the new metrics structure in train_env.py
    prompt_context = "Current Situation:\n"
    trains_list = list(train_info['train_details'].values())
    for i, train in enumerate(trains_list):
        prompt_context += f"- Train '{train['id']}' is at node '{train['current_node']}'. Its current state is '{train['state']}'.\n"

    action_desc = "AI Recommended Action:\n"
    for i, train in enumerate(trains_list):
        action = "Proceed" if action_array[i] == 1 else "Hold"
        action_desc += f"- For Train '{train['id']}': {action}\n"
    
    # Create a prompt formatted for the local FLAN-T5 model
    prompt = f"Question: Based on the following situation, briefly explain the reasoning for the recommended action in one sentence.\n\nContext:\n{prompt_context}{action_desc}\n\nAnswer:"
    
    try:
        response = explainer(prompt, max_length=30, num_beams=2)
        return response[0]['generated_text']
    except Exception as e:
        print(f"Local LLM generation failed: {e}")
        return "Could not generate an explanation locally."

# --- Backend Setup ---
app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Global Variables ---
env = TrainTrafficEnv()
try:
    n_observations = len(env.observation_space.sample())
    # Use train_configs, which is available at the start.
    n_actions = 2 ** len(env.train_configs)
    
    policy_net = QNetwork(n_observations, n_actions)
    policy_net.load_state_dict(torch.load("dqn_policy_net.pth"))
    policy_net.eval()
    print("AI model loaded successfully.")
except FileNotFoundError:
    print("ERROR: dqn_policy_net.pth not found. Please run dqn_agent.py to train the model first.")
    exit()

simulation_thread = None
is_simulation_running = False

# --- Simulation Logic ---
def run_simulation():
    global is_simulation_running
    state, info = env.reset()
    
    while is_simulation_running:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_q_values = policy_net(state_tensor)
            action_index = action_q_values.max(1)[1].item()
        
        action_array = [int(x) for x in np.binary_repr(action_index, width=len(env.train_configs))]
        
        current_train_info = env._get_info()
        explanation = get_ai_explanation(current_train_info, action_array)
        
        current_state_data = {
            'nodes': list(env.rail_network.nodes(data=True)),
            'edges': list(env.rail_network.edges(data=True)),
            'trains': current_train_info['train_details'],
            'suggestion': {
                'action_desc': f"AI suggests action: {action_array} ({'Hold T01, ' if action_array[0]==0 else 'Proceed T01, '}{'Hold T02' if action_array[1]==0 else 'Proceed T02'})",
                'explanation': explanation
            }
        }
        socketio.emit('update_state', current_state_data)
        state, reward, terminated, _, info = env.step(action_array)
        
        if terminated:
            print("Simulation episode finished. Resetting.")
            state, info = env.reset()
            
        socketio.sleep(3)

# --- API and WebSocket Routes ---
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('start_simulation')
def handle_start__simulation():
    global is_simulation_running, simulation_thread
    if not is_simulation_running:
        is_simulation_running = True
        simulation_thread = socketio.start_background_task(target=run_simulation)
        print("Starting simulation...")

@socketio.on('disconnect')
def test_disconnect():
    global is_simulation_running
    is_simulation_running = False
    print('Client disconnected, stopping simulation.')

if __name__ == '__main__':
    print("Server starting... Please ensure your frontend is built and ready.")
    socketio.run(app, port=5000)