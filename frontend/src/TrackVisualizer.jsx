import React, { useMemo } from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';

const nodeLayoutPositions = {
    "S_A": { x: 50, y: 200 },
    "J_A": { x: 250, y: 200 },
    "ML": { x: 450, y: 150 },
    "SL": { x: 450, y: 250 },
    "J_B": { x: 650, y: 200 },
    "S_B": { x: 850, y: 200 }
};

const TrainNode = ({ data }) => (
    <div style={{
        background: '#ff6b6b',
        color: 'white',
        padding: '5px 10px',
        borderRadius: '5px',
        border: '2px solid #c94a4a'
    }}>
        <strong>🚂 {data.label}</strong>
    </div>
);

const nodeTypes = { train: TrainNode };

function TrackVisualizer({ simulationState }) {
    const { nodes, edges } = useMemo(() => {
        if (!simulationState || !simulationState.nodes) {
            return { nodes: [], edges: [] };
        }

        const trackNodes = simulationState.nodes.map(([nodeId, attrs]) => ({
            id: nodeId,
            data: { label: `${nodeId} (${attrs.type})` },
            position: nodeLayoutPositions[nodeId] || { x: 0, y: 0 },
            style: {
                background: attrs.type === 'station' ? '#a3d9ff' : '#90ee90',
                color: '#333',
                border: '1px solid #222',
                width: 120,
            }
        }));

        const trainNodes = Object.values(simulationState.trains || {}).map(train => ({
            id: train.id,
            type: 'train',
            data: { label: train.id },
            position: { 
                x: nodeLayoutPositions[train.current_node].x + 20,
                y: nodeLayoutPositions[train.current_node].y + 40,
            },
        }));

        const trackEdges = simulationState.edges.map(([u, v, attrs]) => ({
            id: `e-${u}-${v}`,
            source: u,
            target: v,
            label: `${attrs.weight}km`,
            animated: Object.values(simulationState.trains || {}).some(t => t.current_node === u && t.next_node === v),
        }));

        return { nodes: [...trackNodes, ...trainNodes], edges: trackEdges };
    }, [simulationState]);

    return (
        <div style={{ height: '100%', width: '100%' }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                fitView
            >
                <Background />
                <Controls />
            </ReactFlow>
        </div>
    );
}

export default TrackVisualizer;