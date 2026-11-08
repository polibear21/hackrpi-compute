import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// Real Darrin Communications Center (DCC) 200 Level Waypoints extracted directly from "2nd floor" ARKit walk
const DEFAULT_NODES = [
  { id: 'exit_west',       label: '218 West Lobby / Stair 2ST1', pos: [-5.5, 0.0, -4.8], is_exit: true,  blocked: false },
  { id: 'corr_232_w',      label: '218 West Lobby Doors',        pos: [-2.0, 0.0, -4.3], is_exit: false, blocked: false },
  { id: 'room_232',        label: 'Room 232 (Inside)',           pos: [0.1,  0.0,  0.5], is_exit: false, blocked: false },
  { id: 'door_232',        label: 'Room 232 Doorway',            pos: [-1.1, 0.0, -0.8], is_exit: false, blocked: false },
  { id: 'corr_232',        label: 'Corridor 219 (Outside 232)',  pos: [0.9,  0.0, -5.8], is_exit: false, blocked: false },
  { id: 'corr_234',        label: 'Corridor 219 (Outside 234)',  pos: [6.3,  0.0, -5.5], is_exit: false, blocked: false },
  { id: 'room_235_door',   label: 'Room 235 Alcove / Door',      pos: [9.9,  0.0, -3.7], is_exit: false, blocked: false },
  { id: 'corr_235',        label: 'Corridor 219 (Outside 235)',  pos: [10.6, 0.0, -5.3], is_exit: false, blocked: false },
  { id: 'corr_236',        label: 'Corridor 219 (Outside 236)',  pos: [15.0, 0.0, -5.5], is_exit: false, blocked: false },
  { id: 'corr_238',        label: 'Corridor 219 (Outside 238)',  pos: [21.3, 0.0, -4.7], is_exit: false, blocked: false },
  { id: 'corr_239',        label: 'Corridor 219 (Outside 239)',  pos: [27.0, 0.0, -5.0], is_exit: false, blocked: false },
  { id: 'corr_240',        label: 'Corridor 240 (Outside 240)',  pos: [31.6, 0.0, -4.6], is_exit: false, blocked: false },
  { id: 'exit_stair_2st6', label: 'Stair 2ST6 Exit',             pos: [37.9, 0.0, -4.8], is_exit: true,  blocked: false },
  { id: 'corr_east_end',   label: 'Corridor 219 East Jct',       pos: [42.1, 0.0, -6.3], is_exit: false, blocked: false },
  { id: 'exit_east_242',   label: '242 East Lobby Exit',         pos: [43.6, 0.0,  0.1], is_exit: true,  blocked: false },
];

const DEFAULT_EDGES = [
  ['room_232', 'door_232'],
  ['door_232', 'corr_232_w'],
  ['door_232', 'corr_232'],
  ['corr_232_w', 'exit_west'],
  ['corr_232_w', 'corr_232'],
  ['corr_232', 'corr_234'],
  ['corr_234', 'room_235_door'],
  ['room_235_door', 'corr_235'],
  ['corr_234', 'corr_235'],
  ['corr_235', 'corr_236'],
  ['corr_236', 'corr_238'],
  ['corr_238', 'corr_239'],
  ['corr_239', 'corr_240'],
  ['corr_240', 'exit_stair_2st6'],
  ['exit_stair_2st6', 'corr_east_end'],
  ['corr_east_end', 'exit_east_242'],
];

// Client-side Dijkstra fallback if WebSocket backend is offline
function computeLocalDijkstra(nodes, edges, ffPos) {
  const nodeMap = {};
  nodes.forEach((n) => { nodeMap[n.id] = n; });

  let startId = 'room_232';
  let bestDist = Infinity;
  nodes.forEach((n) => {
    if (n.blocked) return;
    const d = Math.hypot(n.pos[0] - ffPos[0], n.pos[1] - ffPos[1], n.pos[2] - ffPos[2]);
    if (d < bestDist) {
      bestDist = d;
      startId = n.id;
    }
  });

  const adj = {};
  nodes.forEach((n) => { adj[n.id] = []; });
  edges.forEach(([u, v]) => {
    if (nodeMap[u] && nodeMap[v]) {
      const pu = nodeMap[u].pos;
      const pv = nodeMap[v].pos;
      const w = Math.hypot(pu[0] - pv[0], pu[1] - pv[1], pu[2] - pv[2]);
      adj[u].push({ to: v, w });
      adj[v].push({ to: u, w });
    }
  });

  const dist = {};
  const prev = {};
  const visited = new Set();
  nodes.forEach((n) => { dist[n.id] = Infinity; prev[n.id] = null; });
  dist[startId] = 0;

  let chosenExit = null;
  while (true) {
    let u = null;
    let best = Infinity;
    for (const n of nodes) {
      if (!visited.has(n.id) && !n.blocked && dist[n.id] < best) {
        best = dist[n.id];
        u = n.id;
      }
    }
    if (!u) break;
    visited.add(u);
    if (u === 'exit_east_242' && !nodeMap[u].blocked) {
      chosenExit = u;
      break;
    }
    for (const edge of adj[u] || []) {
      if (nodeMap[edge.to].blocked || visited.has(edge.to)) continue;
      const alt = dist[u] + edge.w;
      if (alt < dist[edge.to]) {
        dist[edge.to] = alt;
        prev[edge.to] = u;
      }
    }
  }

  if (!chosenExit) {
    return {
      start_node: startId,
      start_label: nodeMap[startId]?.label || startId,
      target_exit: null,
      path_ids: [],
      path_coords: [],
      total_distance_m: null,
      reachable: false,
    };
  }

  const pathIds = [];
  let curr = chosenExit;
  while (curr) {
    pathIds.unshift(curr);
    curr = prev[curr];
  }
  return {
    start_node: startId,
    start_label: nodeMap[startId]?.label || startId,
    target_exit: chosenExit,
    path_ids: pathIds,
    path_coords: pathIds.map((id) => nodeMap[id].pos),
    total_distance_m: Number(dist[chosenExit].toFixed(2)),
    reachable: true,
  };
}

// ============================================================================
// 3D SCENE COMPONENTS
// ============================================================================

function PolycamGlbModel({ url }) {
  const [sceneObj, setSceneObj] = useState(null);

  useEffect(() => {
    if (!url) return;
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => setSceneObj(gltf.scene),
      undefined,
      () => setSceneObj(null)
    );
  }, [url]);

  if (!sceneObj) return null;
  return <primitive object={sceneObj} position={[0, 0, 0]} />;
}

function ProceduralBuildingShell() {
  const [floorTex, setFloorTex] = useState(null);

  useEffect(() => {
    const loader = new THREE.TextureLoader();
    loader.load('/dcc_200_floorplan.png', (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      setFloorTex(tex);
    });
  }, []);

  // DCC 200 Level Classrooms (232, 235, 236, 239, 240) + Corridor 219 aligned to "2nd floor" walk (46.3m span)
  const rooms = [
    { name: '232 CLASSROOM', center: [0.0, 1.1, -0.5], size: [5.5, 2.2, 5.2], color: '#38bdf8' },
    { name: '235 CLASSROOM', center: [10.2, 1.1, -0.5], size: [6.0, 2.2, 5.2], color: '#38bdf8' },
    { name: '236 CLASSROOM', center: [16.5, 1.1, -0.5], size: [6.0, 2.2, 5.2], color: '#38bdf8' },
    { name: '239 CLASSROOM', center: [26.5, 1.1, -0.5], size: [6.0, 2.2, 5.2], color: '#38bdf8' },
    { name: '240 CLASSROOM', center: [33.0, 1.1, -0.5], size: [6.0, 2.2, 5.2], color: '#38bdf8' },
    { name: '219 CORRIDOR', center: [19.5, 1.1, -5.2], size: [50.0, 2.2, 2.6], color: '#64748b' },
  ];

  return (
    <group>
      {/* Actual DCC 200 Level Blueprint Texture on 3D Ground Plane (54m x 20.1m) */}
      {floorTex && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[19.0, 0.01, -3.5]}>
          <planeGeometry args={[54.0, 20.1]} />
          <meshBasicMaterial map={floorTex} transparent opacity={0.85} />
        </mesh>
      )}
      {rooms.map((r, i) => (
        <group key={i} position={r.center}>
          <mesh>
            <boxGeometry args={r.size} />
            <meshStandardMaterial
              color={r.color}
              transparent
              opacity={0.07}
              depthWrite={false}
              side={THREE.DoubleSide}
            />
          </mesh>
          <lineSegments>
            <edgesGeometry args={[new THREE.BoxGeometry(...r.size)]} />
            <lineBasicMaterial color={r.color} transparent opacity={0.28} />
          </lineSegments>
        </group>
      ))}
    </group>
  );
}

function VpsPointCloud({ points }) {
  const geo = useMemo(() => {
    if (!points || points.length === 0) return null;
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      pos[i * 3] = p[0];
      pos[i * 3 + 1] = p[1];
      pos[i * 3 + 2] = p[2];
    });
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    return g;
  }, [points]);

  if (!geo) return null;
  return (
    <points geometry={geo}>
      <pointsMaterial color="#38bdf8" size={0.055} transparent opacity={0.75} />
    </points>
  );
}

function WaypointNode({ node, isOnActivePath, onToggleBlock }) {
  const meshRef = useRef();
  const ringRef = useRef();
  const [hovered, setHovered] = useState(false);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (ringRef.current) {
      const scale = 1.0 + 0.25 * Math.sin(t * 4.0);
      ringRef.current.scale.set(scale, scale, scale);
    }
    if (meshRef.current && hovered) {
      meshRef.current.position.y = 0.22 + 0.05 * Math.sin(t * 6.0);
    } else if (meshRef.current) {
      meshRef.current.position.y = 0.18;
    }
  });

  let color = '#64748b';
  if (node.blocked) color = '#ef4444';
  else if (node.is_exit) color = '#10b981';
  else if (isOnActivePath) color = '#22d3ee';

  return (
    <group position={[node.pos[0], node.pos[1], node.pos[2]]}>
      <mesh
        ref={meshRef}
        position={[0, 0.18, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onToggleBlock(node.id);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          setHovered(false);
          document.body.style.cursor = 'default';
        }}
      >
        <sphereGeometry args={[node.is_exit ? 0.26 : 0.2, 24, 24]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={node.blocked || isOnActivePath || node.is_exit ? 0.85 : 0.3}
        />
      </mesh>

      {/* Pulsing hazard or exit ring on floor */}
      {(node.blocked || node.is_exit || isOnActivePath) && (
        <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
          <ringGeometry args={[0.26, 0.36, 32]} />
          <meshBasicMaterial color={color} side={THREE.DoubleSide} transparent opacity={0.75} />
        </mesh>
      )}

      {/* Floating 3D Label */}
      <Html position={[0, 0.55, 0]} center distanceFactor={14}>
        <div
          onClick={() => onToggleBlock(node.id)}
          style={{
            background: node.blocked
              ? 'rgba(185, 28, 28, 0.92)'
              : node.is_exit
              ? 'rgba(6, 95, 70, 0.92)'
              : 'rgba(15, 23, 42, 0.88)',
            border: `1px solid ${color}`,
            color: '#fff',
            padding: '3px 8px',
            borderRadius: '6px',
            fontSize: '11px',
            fontWeight: 600,
            whiteSpace: 'nowrap',
            cursor: 'pointer',
            userSelect: 'none',
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          }}
        >
          {node.blocked ? '🔥 BLOCKED: ' : node.is_exit ? '🚪 ' : ''}
          {node.label}
        </div>
      </Html>
    </group>
  );
}

function FirefighterMarker({ position, mayday }) {
  const groupRef = useRef();

  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.position.y = position[1] + 0.35 + 0.06 * Math.sin(clock.getElapsedTime() * 5);
    }
  });

  const color = mayday ? '#ef4444' : '#f97316';

  return (
    <group position={[position[0], 0, position[2]]}>
      <mesh ref={groupRef}>
        <octahedronGeometry args={[0.25, 0]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.1} />
      </mesh>
      <Html position={[0, 0.85, 0]} center>
        <div
          style={{
            background: mayday ? '#dc2626' : '#ea580c',
            color: '#fff',
            padding: '4px 10px',
            borderRadius: '999px',
            fontSize: '11px',
            fontWeight: 800,
            letterSpacing: '0.5px',
            whiteSpace: 'nowrap',
            boxShadow: '0 0 16px rgba(249, 115, 22, 0.8)',
          }}
        >
          {mayday ? '🚨 MAYDAY // FIREFIGHTER 1' : '👨‍🚒 FIREFIGHTER 1 (LIVE VPS)'}
        </div>
      </Html>
    </group>
  );
}

// ============================================================================
// MAIN COMMANDER DASHBOARD COMPONENT
// ============================================================================
export default function App() {
  const [wsConnected, setWsConnected] = useState(false);
  const [nodes, setNodes] = useState(DEFAULT_NODES);
  const [edges, setEdges] = useState(DEFAULT_EDGES);
  const [pointcloud, setPointcloud] = useState([]);
  const [ffPos, setFfPos] = useState([0.1, 0.0, 0.5]);
  const [mayday, setMayday] = useState(false);
  const [audioCue, setAudioCue] = useState('system_online.mp3');
  const [glbUrl, setGlbUrl] = useState('/building.glb');
  const [showShell, setShowShell] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [latestFrameB64, setLatestFrameB64] = useState(null);

  const wsRef = useRef(null);

  const route = useMemo(
    () => computeLocalDijkstra(nodes, edges, ffPos),
    [nodes, edges, ffPos]
  );

  // Connect to FastAPI /ws/dashboard WebSocket
  useEffect(() => {
    const wsUrl = `ws://${window.location.hostname || 'localhost'}:8000/ws/dashboard`;
    let socket;
    let reconnectTimer;

    const connect = () => {
      socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => setWsConnected(true);
      socket.onclose = () => {
        setWsConnected(false);
        reconnectTimer = setTimeout(connect, 2500);
      };
      socket.onerror = () => socket.close();
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.nodes) setNodes(data.nodes);
          if (data.edges) setEdges(data.edges);
          if (data.pointcloud && data.pointcloud.length > 0) setPointcloud(data.pointcloud);
          if (data.firefighter) {
            setFfPos(data.firefighter.position);
            setMayday(Boolean(data.firefighter.mayday));
          }
          if (data.audio_cue) setAudioCue(data.audio_cue);
          if (data.latest_frame_b64) setLatestFrameB64(data.latest_frame_b64);
        } catch (err) {
          console.error('WS parse error:', err);
        }
      };
    };

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      if (socket) socket.close();
    };
  }, []);

  // Toggle "Blocked" state on a waypoint (syncs with FastAPI backend + updates Dijkstra immediately)
  const handleToggleBlock = useCallback((nodeId) => {
    setNodes((prev) =>
      prev.map((n) => (n.id === nodeId ? { ...n, blocked: !n.blocked } : n))
    );
    setAudioCue('hazard_ahead_rerouting.mp3');

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'toggle_block', node_id: nodeId }));
    }
  }, []);

  // Simulate firefighter walking along the active Dijkstra path
  useEffect(() => {
    if (!simulating) return;
    let step = 0;
    const timer = setInterval(() => {
      if (!route.path_coords || route.path_coords.length === 0) return;
      const target = route.path_coords[step % route.path_coords.length];
      setFfPos(target);
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'simulate_move', position: target }));
      }
      step++;
    }, 1100);
    return () => clearInterval(timer);
  }, [simulating, route.path_coords]);

  // Allow dragging and dropping a Polycam .glb file right onto the browser window
  const handleDropGlb = (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file && (file.name.endsWith('.glb') || file.name.endsWith('.gltf'))) {
      const objectUrl = URL.createObjectURL(file);
      setGlbUrl(objectUrl);
    }
  };

  const activeSet = useMemo(() => new Set(route.path_ids || []), [route.path_ids]);
  const nodeLookup = useMemo(() => {
    const m = {};
    nodes.forEach((n) => { m[n.id] = n; });
    return m;
  }, [nodes]);

  return (
    <div
      style={{ width: '100vw', height: '100vh', position: 'relative' }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDropGlb}
    >
      {/* MAYDAY EMERGENCY BANNER */}
      {mayday && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 50,
            background: 'linear-gradient(90deg, #991b1b, #ef4444, #991b1b)',
            color: '#fff',
            padding: '10px 24px',
            textAlign: 'center',
            fontWeight: 800,
            letterSpacing: '1px',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '16px',
          }}
        >
          <span>🚨 MAYDAY TRANSMITTED — FIREFIGHTER 1 AT {route.start_label.toUpperCase()}</span>
          <button
            onClick={() => {
              setMayday(false);
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'clear_mayday' }));
              }
            }}
            style={{
              background: '#fff',
              color: '#991b1b',
              border: 'none',
              padding: '4px 12px',
              borderRadius: '6px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Acknowledge Mayday
          </button>
        </div>
      )}

      {/* LEFT HUD: INCIDENT COMMAND TELEMETRY */}
      <div
        style={{
          position: 'absolute',
          top: mayday ? 56 : 16,
          left: 16,
          zIndex: 20,
          width: 350,
          background: 'rgba(15, 23, 42, 0.90)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(249, 115, 22, 0.4)',
          borderRadius: 14,
          padding: 18,
          boxShadow: '0 16px 40px rgba(0,0,0,0.6)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ margin: 0, fontSize: 18, color: '#f97316', letterSpacing: 0.8 }}>
            FIRESIGHT // COMMAND
          </h2>
          <span
            style={{
              fontSize: 11,
              padding: '3px 8px',
              borderRadius: 999,
              fontWeight: 700,
              background: wsConnected ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
              color: wsConnected ? '#34d399' : '#fbbf24',
            }}
          >
            {wsConnected ? '● BRAIN ONLINE' : '● LOCAL SIM MODE'}
          </span>
        </div>

        {/* Firefighter Current Location Card */}
        <div
          style={{
            marginTop: 14,
            padding: 12,
            background: 'rgba(30, 41, 59, 0.85)',
            borderRadius: 10,
            borderLeft: '4px solid #f97316',
          }}
        >
          <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>
            Firefighter 1 Located In
          </div>
          <div style={{ fontSize: 18, fontWeight: 800, color: '#fff', marginTop: 2 }}>
            {route.start_label}
          </div>
          <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 4, fontFamily: 'monospace' }}>
            XYZ: ({ffPos[0].toFixed(2)}m, {ffPos[1].toFixed(2)}m, {ffPos[2].toFixed(2)}m)
          </div>
        </div>

        {/* Dijkstra Route Status */}
        <div
          style={{
            marginTop: 10,
            padding: 12,
            background: 'rgba(30, 41, 59, 0.85)',
            borderRadius: 10,
            borderLeft: `4px solid ${route.reachable ? '#22d3ee' : '#ef4444'}`,
          }}
        >
          <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>
            Active Dijkstra Egress Route
          </div>
          {route.reachable ? (
            <>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#22d3ee', marginTop: 3 }}>
                → {nodeLookup[route.target_exit]?.label} ({route.total_distance_m}m)
              </div>
              <div style={{ fontSize: 11, color: '#cbd5e1', marginTop: 4 }}>
                Path: {route.path_ids.join(' → ')}
              </div>
            </>
          ) : (
            <div style={{ fontSize: 14, fontWeight: 700, color: '#ef4444', marginTop: 3 }}>
              ⚠️ ALL ROUTES BLOCKED — NO EGRESS!
            </div>
          )}
        </div>

        {/* Pre-rendered Audio Flag Dispatched to iOS */}
        <div style={{ marginTop: 10, fontSize: 12, color: '#94a3b8' }}>
          Last iOS Audio Trigger:{' '}
          <code style={{ color: '#fbbf24', background: '#1e293b', padding: '2px 6px', borderRadius: 4 }}>
            {audioCue || 'none'}
          </code>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: 8, marginTop: 14, flexWrap: 'wrap' }}>
          <button
            onClick={() => setSimulating((s) => !s)}
            style={{
              flex: 1,
              background: simulating ? '#dc2626' : '#f97316',
              color: '#fff',
              border: 'none',
              padding: '8px 12px',
              borderRadius: 8,
              fontWeight: 700,
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            {simulating ? '⏹ Stop Walk Sim' : '▶ Simulate Walk'}
          </button>
          <button
            onClick={() => setShowShell((s) => !s)}
            style={{
              background: '#1e293b',
              color: '#e2e8f0',
              border: '1px solid #475569',
              padding: '8px 10px',
              borderRadius: 8,
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            {showShell ? 'Hide 3D Walls' : 'Show 3D Walls'}
          </button>
        </div>

        <div style={{ marginTop: 10, fontSize: 11, color: '#64748b' }}>
          💡 <b>Tip:</b> Click any 3D waypoint node in the map (or below) to mark it{' '}
          <span style={{ color: '#ef4444', fontWeight: 700 }}>BLOCKED</span> and trigger live Dijkstra rerouting.
          Drag & drop any Polycam <code>.glb</code> file onto the window to load it.
        </div>

        {/* LIVE HELMET CAMERA FEED (PiP inside Commander HUD) */}
        {latestFrameB64 && (
          <div
            style={{
              marginTop: 12,
              background: '#090d16',
              border: '1px solid rgba(34, 211, 238, 0.45)',
              borderRadius: 10,
              padding: 8,
            }}
          >
            <div
              style={{
                fontSize: 10,
                fontWeight: 800,
                color: '#22d3ee',
                marginBottom: 6,
                letterSpacing: 0.6,
                display: 'flex',
                justifyContent: 'space-between',
              }}
            >
              <span>📹 LIVE HELMET CAM (AR STREAM)</span>
              <span style={{ color: '#10b981' }}>● LIVE</span>
            </div>
            <img
              src={`data:image/jpeg;base64,${latestFrameB64}`}
              alt="Firefighter Helmet Camera"
              style={{
                width: '100%',
                height: 150,
                objectFit: 'cover',
                borderRadius: 6,
                display: 'block',
              }}
            />
          </div>
        )}
      </div>

      {/* RIGHT HUD: WAYPOINT NODE LIST */}
      <div
        style={{
          position: 'absolute',
          top: mayday ? 56 : 16,
          right: 16,
          zIndex: 20,
          width: 270,
          maxHeight: '85vh',
          overflowY: 'auto',
          background: 'rgba(15, 23, 42, 0.88)',
          backdropFilter: 'blur(12px)',
          border: '1px solid #334155',
          borderRadius: 14,
          padding: 14,
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>
          WAYPOINT GRAPH ({nodes.length} NODES)
        </div>
        {nodes.map((n) => (
          <div
            key={n.id}
            onClick={() => handleToggleBlock(n.id)}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '7px 10px',
              marginBottom: 6,
              borderRadius: 8,
              cursor: 'pointer',
              background: n.blocked
                ? 'rgba(185, 28, 28, 0.28)'
                : activeSet.has(n.id)
                ? 'rgba(34, 211, 238, 0.15)'
                : 'rgba(30, 41, 59, 0.65)',
              border: `1px solid ${
                n.blocked ? '#ef4444' : activeSet.has(n.id) ? '#22d3ee' : 'transparent'
              }`,
            }}
          >
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#f8fafc' }}>{n.label}</div>
              <div style={{ fontSize: 10, color: '#94a3b8', fontFamily: 'monospace' }}>
                ({n.pos[0]}, {n.pos[1]}, {n.pos[2]})
              </div>
            </div>
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                padding: '2px 7px',
                borderRadius: 4,
                background: n.blocked ? '#ef4444' : n.is_exit ? '#10b981' : '#334155',
                color: '#fff',
              }}
            >
              {n.blocked ? 'BLOCKED' : n.is_exit ? 'EXIT' : 'CLEAR'}
            </span>
          </div>
        ))}
      </div>

      {/* 3D REACT THREE FIBER CANVAS */}
      <Canvas camera={{ position: [3.5, 9.5, 6.5], fov: 55 }}>
        <color attach="background" args={['#070b14']} />
        <ambientLight intensity={0.8} />
        <directionalLight position={[8, 15, 10]} intensity={1.2} />
        <OrbitControls target={[3.5, 0.5, -3.5]} makeDefault />

        <gridHelper args={[28, 28, '#1e293b', '#0f172a']} position={[3.5, 0, -3.5]} />

        {/* Polycam .glb mesh (if present) + Procedural 3D Corridor Walls */}
        <PolycamGlbModel url={glbUrl} />
        {showShell && <ProceduralBuildingShell />}

        {/* 3D SuperPoint Landmarks from final_model */}
        <VpsPointCloud points={pointcloud} />

        {/* All Graph Edges (thin gray lines) */}
        {edges.map(([u, v], idx) => {
          const nu = nodeLookup[u];
          const nv = nodeLookup[v];
          if (!nu || !nv) return null;
          return (
            <Line
              key={idx}
              points={[
                [nu.pos[0], nu.pos[1] + 0.05, nu.pos[2]],
                [nv.pos[0], nv.pos[1] + 0.05, nv.pos[2]],
              ]}
              color="#334155"
              lineWidth={1.5}
            />
          );
        })}

        {/* Active Dijkstra Path (bright glowing cyan line) */}
        {route.path_coords && route.path_coords.length > 1 && (
          <Line
            points={route.path_coords.map((p) => [p[0], p[1] + 0.14, p[2]])}
            color="#22d3ee"
            lineWidth={5}
          />
        )}

        {/* Interactive Waypoint Nodes */}
        {(route.path_coords || []).slice(1).flatMap((end, i) => {
          const start = route.path_coords[i];
          const direction = new THREE.Vector3(...end).sub(new THREE.Vector3(...start));
          const length = direction.length();
          direction.normalize();
          return Array.from({ length: Math.ceil(length / 2) }, (_, j) => {
            const origin = new THREE.Vector3(...start).addScaledVector(direction, (j + 0.5) * length / Math.ceil(length / 2));
            origin.y += 0.2;
            return <arrowHelper key={`${i}-${j}`} args={[direction, origin, Math.min(1, length * 0.8), '#22d3ee', 0.35, 0.3]} />;
          });
        })}
        {nodes.map((node) => (
          <WaypointNode
            key={node.id}
            node={node}
            isOnActivePath={activeSet.has(node.id)}
            onToggleBlock={handleToggleBlock}
          />
        ))}

        {/* Live Firefighter Marker */}
        <FirefighterMarker position={ffPos} mayday={mayday} />
      </Canvas>
    </div>
  );
}
