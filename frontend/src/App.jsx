import React, { useState, useEffect, useCallback } from 'react';
import io from 'socket.io-client';
import './App.css';
import TrackVisualizer from './TrackVisualizer';

// Connect to the backend server
const socket = io('http://localhost:5000');

function App() {
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [simulationState, setSimulationState] = useState(null);

  useEffect(() => {
    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));
    socket.on('update_state', (data) => {
      console.log("Received state update:", data);
      setSimulationState(data);
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('update_state');
    };
  }, []);

  const startSimulation = useCallback(() => {
    console.log("Requesting to start simulation...");
    socket.emit('start_simulation');
  }, []);
  
  return (
    <div className="App">
      <header>
        <h1>AI-Powered Train Traffic Control</h1>
        <p>Connection Status: {isConnected ? 'Connected ✅' : 'Disconnected ❌'}</p>
      </header>
      <main>
        <div className="visualizer-container">
          <TrackVisualizer simulationState={simulationState} />
        </div>
        <div className="control-panel-container">
          <h2>Control Panel</h2>
          <button onClick={startSimulation} disabled={!isConnected}>
            Start Simulation
          </button>
          {simulationState && simulationState.suggestion && (
            <div className="suggestion-box">
              <h3>AI Suggestion</h3>
              <p>{simulationState.suggestion.action_desc}</p>
              <h3>Explanation</h3>
              <p>{simulationState.suggestion.explanation}</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;