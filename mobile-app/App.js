import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  StyleSheet,
  Text,
  View,
  Image,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  SafeAreaView,
  Platform,
  Animated,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';
import navigation from './navigation';
const { nextRoutePoint, smoothTurn } = navigation;

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

// Default to your laptop's detected Wi-Fi IP on port 8000
const DEFAULT_WS_URL = 'ws://129.161.52.222:8000/ws/ar';

// Quick-pick DCC Floor 2 rooms for Firefighter Target-Room Navigation
const QUICK_DCC_ROOMS = [
  { id: 'F2-232', short: 'Rm 232' },
  { id: 'F2-234', short: 'Rm 234' },
  { id: 'F2-236', short: 'Rm 235/236' },
  { id: 'F2-238', short: 'Rm 238' },
  { id: 'F2-239', short: 'Rm 239' },
  { id: 'F2-240', short: 'Rm 240' },
  { id: 'F2-218', short: '218 Lobby' },
  { id: 'F2-2ST6', short: 'Stair 2ST6' },
  { id: 'F2-308L', short: '308 Lecture' },
];

// Audio cue voice lines (triggered by JSON flags from centralized backend)
const AUDIO_CUE_PHRASES = {
  'system_online.mp3': 'Firesight 3D D C C Navigation Online.',
  'hazard_ahead_rerouting.mp3': 'Warning! Fire hazard reported ahead. Rerouting path immediately.',
  'route_recalculated.mp3': 'Route updated with latest live fire data.',
  'proceed_to_next_waypoint.mp3': 'Continue along the highlighted 3D path.',
  'arrived_at_exit.mp3': 'You have reached your destination.',
  'mayday_transmitting.mp3': 'Mayday alert transmitted to Incident Command.',
  'mayday_acknowledged.mp3': 'Incident Commander acknowledged your Mayday.',
};

// Compute relative yaw angle (in degrees) from camera pose to next waypoint in ARKit world (X, Z)
function computeTurnAngleDeg(camPos, camQuat, nextWaypoint) {
  if (!camPos || !nextWaypoint) return 0;
  const dx = nextWaypoint[0] - camPos[0];
  const dz = nextWaypoint[2] - camPos[2];
  if (Math.hypot(dx, dz) < 0.25) return 0;

  const targetYaw = Math.atan2(dx, -dz);

  if (camQuat && camQuat.length === 4) {
    const [qx, qy, qz, qw] = camQuat;
    const lookX = -(2 * (qx * qz + qw * qy));
    const lookZ = -(1 - 2 * (qx * qx + qy * qy));
    const camYaw = Math.atan2(lookX, -lookZ);
    let diff = ((targetYaw - camYaw) * 180) / Math.PI;
    while (diff > 180) diff -= 360;
    while (diff < -180) diff += 360;
    return diff;
  }
  return 0;
}

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const [liveFrameUri, setLiveFrameUri] = useState(null);
  const [useStreamOverlay, setUseStreamOverlay] = useState(true);
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS_URL);
  const [connected, setConnected] = useState(false);
  const [localized, setLocalized] = useState(false);

  // Task 1 ('occupant' -> Escape to Closest Exit) vs Task 2 ('firefighter' -> Navigate to Target Room + Report Fire)
  const [role, setRole] = useState('occupant');
  const [targetRoomId, setTargetRoomId] = useState('F2-240');
  const [alertBanner, setAlertBanner] = useState(null);

  const [position, setPosition] = useState([0.0, 0.0, 0.0]);
  const [quaternion, setQuaternion] = useState([0, 0, 0, 1]);
  const [currentRoomId, setCurrentRoomId] = useState('F2-232');
  const [currentLabel, setCurrentLabel] = useState('232 CLASSROOM');
  const [inliers, setInliers] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [nodes, setNodes] = useState([]);
  const [route, setRoute] = useState({
    role: 'occupant',
    start_node: 'F2-232',
    start_label: '232 CLASSROOM',
    target_exit: '242 East Lobby Exit',
    path_coords: [
      [0.0, 0.0, 0.0],
      [-1.1, 0.0, -0.8],
      [0.9, 0.0, -5.8],
      [15.0, 0.0, -5.5],
      [31.6, 0.0, -4.6],
      [42.1, 0.0, -6.3],
      [44.19, 0.0, 3.29],
    ],
    total_distance_m: 59.1,
    instructions: ['Leave room 232 and turn east along corridor 219 toward the 242 East Lobby Exit.'],
    fire_rooms: [],
    smoke_rooms: [],
    reachable: true,
  });
  const [mayday, setMayday] = useState(false);
  const [pulsePhase, setPulsePhase] = useState(0);
  const [showUrlBar, setShowUrlBar] = useState(false);

  const cameraRef = useRef(null);
  const wsRef = useRef(null);
  const isCapturingRef = useRef(false);
  const roleRef = useRef(role);
  const targetRoomRef = useRef(targetRoomId);
  const filteredTurnRef = useRef(null);
  const displayedTurnRef = useRef(0);
  const arrowMotion = useRef(new Animated.Value(0)).current;
  const [turnDeg, setTurnDeg] = useState(0);

  useEffect(() => {
    roleRef.current = role;
  }, [role]);

  useEffect(() => {
    targetRoomRef.current = targetRoomId;
  }, [targetRoomId]);

  // Animate glowing floor arrows
  useEffect(() => {
    const id = setInterval(() => {
      setPulsePhase((p) => (p + 1) % 3);
    }, 350);
    return () => clearInterval(id);
  }, []);

  // Connect to Centralized FastAPI /ws/ar WebSocket
  const connectWebSocket = (urlToConnect) => {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (_) {}
    }
    try {
      const ws = new WebSocket(urlToConnect);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        ws.send(
          JSON.stringify({
            type: 'set_mode',
            role: roleRef.current,
            target_room_id: targetRoomRef.current,
          })
        );
      };
      ws.onclose = () => {
        setConnected(false);
      };
      ws.onerror = () => {
        setConnected(false);
      };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.nodes) setNodes(msg.nodes);
          if (msg.route) {
            setRoute(msg.route);
            if (msg.route.start_node) setCurrentRoomId(msg.route.start_node);
            if (msg.route.start_label) setCurrentLabel(msg.route.start_label);
          }
          if (msg.alert_banner) {
            setAlertBanner(msg.alert_banner);
          }
          if (msg.localized) {
            setLocalized(true);
            if (msg.position) setPosition(msg.position);
            if (msg.quaternion) setQuaternion(msg.quaternion);
            if (msg.num_inliers !== undefined) setInliers(msg.num_inliers);
            if (msg.latency_ms !== undefined) setLatencyMs(msg.latency_ms);
          } else if (msg.localized === false) {
            setLocalized(false);
            if (msg.latency_ms !== undefined) setLatencyMs(msg.latency_ms);
          }
          if (msg.audio_cue && AUDIO_CUE_PHRASES[msg.audio_cue]) {
            try {
              Speech.stop();
              Speech.speak(AUDIO_CUE_PHRASES[msg.audio_cue], { rate: 1.05, pitch: 0.95 });
            } catch (_) {}
          }
        } catch (_) {}
      };
    } catch (_) {
      setConnected(false);
    }
  };

  useEffect(() => {
    connectWebSocket(wsUrl);
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Switch between Task 1 (Occupant Escape) and Task 2 (Firefighter Target Room)
  const switchMode = (newRole, newTargetRoom = targetRoomId) => {
    setRole(newRole);
    if (newTargetRoom) setTargetRoomId(newTargetRoom);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'set_mode',
          role: newRole,
          target_room_id: newTargetRoom,
        })
      );
    }
  };

  // Report live fire or clear status from Firefighter phone -> updates Commander webpage + all phones
  const sendFireReport = (kind, customRoomId = null) => {
    const roomToReport = customRoomId || currentRoomId || 'F2-232';
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'report_fire',
          room_id: roomToReport,
          kind,
        })
      );
    }
  };

  // Periodically grab 640x480 frame from camera, update live screen feed, and send to centralized server
  useEffect(() => {
    const interval = setInterval(async () => {
      if (!cameraRef.current || isCapturingRef.current) {
        return;
      }
      isCapturingRef.current = true;
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.35,
          base64: true,
          shutterSound: false,
        });
        if (photo?.uri) {
          setLiveFrameUri(photo.uri);
        } else if (photo?.base64) {
          setLiveFrameUri(`data:image/jpeg;base64,${photo.base64}`);
        }
        if (photo?.base64 && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: 'frame',
              image_b64: photo.base64,
              role: roleRef.current,
              target_room_id: targetRoomRef.current,
            })
          );
        }
      } catch (_) {
        // Ignore transient camera frame busy errors
      } finally {
        isCapturingRef.current = false;
      }
    }, 450);

    return () => clearInterval(interval);
  }, [connected, cameraReady]);

  const sendMayday = (active) => {
    setMayday(active);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'mayday', active }));
    }
    if (active) {
      try {
        Speech.stop();
        Speech.speak(AUDIO_CUE_PHRASES['mayday_transmitting.mp3'], { rate: 1.08 });
      } catch (_) {}
    }
  };

  // Pick next waypoint along the 3D DCC route at least ~1.2m ahead
  const nextWaypoint = useMemo(() => {
    const coords = route?.path_coords || [];
    return nextRoutePoint(coords, position);
  }, [route, position]);

  useEffect(() => {
    if (!connected || !localized || route.arrived || !route.reachable || !nextWaypoint) return;
    const desired = computeTurnAngleDeg(position, quaternion, nextWaypoint);
    const filtered = filteredTurnRef.current == null ? desired : smoothTurn(filteredTurnRef.current, desired);
    filteredTurnRef.current = filtered;
    const wrapped = ((filtered + 180) % 360 + 360) % 360 - 180;
    const stable = Math.abs(wrapped) > 150 && displayedTurnRef.current !== 0
      ? Math.sign(displayedTurnRef.current) * Math.abs(wrapped) : wrapped;
    const displayTurn = Math.max(-65, Math.min(65, stable));
    displayedTurnRef.current = displayTurn;
    setTurnDeg(displayTurn);
    const animation = Animated.timing(arrowMotion, { toValue: displayTurn, duration: 300, useNativeDriver: true });
    animation.start();
    return () => animation.stop();
  }, [position, quaternion, nextWaypoint, connected, localized, route.arrived, route.reachable, arrowMotion]);

  const fireRoomIds = useMemo(() => {
    const s = new Set();
    (route?.fire_rooms || []).forEach((r) => s.add(r.id));
    return s;
  }, [route]);

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permissionContainer}>
        <Text style={styles.permTitle}>FIRESIGHT // 3D DCC NAVIGATION</Text>
        <Text style={styles.permSub}>
          Firesight uses your camera to localize inside the 3D model of Darrin Communications Center (DCC) and route around live fires.
        </Text>
        <TouchableOpacity style={styles.permButton} onPress={requestPermission}>
          <Text style={styles.permButtonText}>ENABLE AR CAMERA</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  const mapX = (x) => Math.max(4, Math.min(148, 22 + ((x + 12) / 56) * 124));
  const mapZ = (z) => Math.max(4, Math.min(64, 38 + ((z + 10) / 18) * 28));
  const primaryInstruction =
    route?.instructions && route.instructions.length > 0
      ? route.instructions[0]
      : Math.abs(turnDeg) < 15
      ? 'Proceed straight ahead'
      : turnDeg > 0
      ? `Turn right (${Math.round(turnDeg)}°)`
      : `Turn left (${Math.round(Math.abs(turnDeg))}°)`;

  return (
    <View style={styles.container}>
      <StatusBar style="light" />

      {/* 1A. LIVE FULLSCREEN BACK CAMERA (640x480 fast VGA preset) */}
      <CameraView
        ref={cameraRef}
        style={styles.cameraFull}
        facing="back"
        mode="picture"
        pictureSize="640x480"
        active={true}
        animateShutter={false}
        onCameraReady={() => setCameraReady(true)}
      />

      {/* 1B. GUARANTEED LIVE CAMERA FEED OVERLAY */}
      {useStreamOverlay && liveFrameUri && (
        <Image
          source={{ uri: liveFrameUri }}
          style={StyleSheet.absoluteFillObject}
          resizeMode="cover"
          fadeDuration={0}
        />
      )}

      {/* 2. 3D PERSPECTIVE AR FLOOR ARROWS OVERLAY */}
      <View style={styles.arFloorContainer} pointerEvents="none">
        {(!connected || !localized || !route.reachable || route.arrived ? [] : [2, 1, 0]).map((idx) => {
          const isLit = pulsePhase === idx;
          const scale = 1.0 - idx * 0.22;
          const baseColor = !route.reachable
            ? '#ef4444'
            : role === 'firefighter'
            ? isLit
              ? '#fb923c'
              : 'rgba(249, 115, 22, 0.45)'
            : isLit
            ? '#22d3ee'
            : 'rgba(6, 182, 212, 0.45)';

          return (
            <Animated.View
              key={idx}
              style={[
                styles.chevronRow,
                {
                  marginBottom: 18,
                  transform: [
                    { perspective: 420 },
                    { translateX: arrowMotion.interpolate({ inputRange: [-65, 65], outputRange: [-SCREEN_W * 0.25 * ((idx + 1) / 3), SCREEN_W * 0.25 * ((idx + 1) / 3)] }) },
                    { rotateX: '58deg' },
                    { rotateZ: arrowMotion.interpolate({ inputRange: [-65, 65], outputRange: ['-29.25deg', '29.25deg'] }) },
                    { scale },
                  ],
                },
              ]}
            >
              <View
                style={[
                  styles.chevronBox,
                  {
                    borderTopColor: baseColor,
                    borderRightColor: baseColor,
                    shadowColor: baseColor,
                    shadowOpacity: isLit ? 0.9 : 0.3,
                  },
                ]}
              />
            </Animated.View>
          );
        })}
      </View>

      {/* 3. TOP TACTICAL VISOR HUD */}
      <SafeAreaView style={styles.topHudWrapper}>
        <View
          style={[
            styles.hudCard,
            role === 'firefighter' && styles.hudCardFirefighter,
            mayday && styles.hudCardMayday,
          ]}
        >
          {/* MODE SWITCHER: Task 1 (Occupant Escape) vs Task 2 (Firefighter Target Room) */}
          <View style={styles.modeSwitchRow}>
            <TouchableOpacity
              style={[styles.modeBtn, role === 'occupant' && styles.modeBtnOccupantActive]}
              onPress={() => switchMode('occupant')}
            >
              <Text style={styles.modeBtnText}>🏃 EVACUATE TO EXIT</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.modeBtn, role === 'firefighter' && styles.modeBtnFirefighterActive]}
              onPress={() => switchMode('firefighter', targetRoomId)}
            >
              <Text style={styles.modeBtnText}>👨‍🚒 FIREFIGHTER MODE</Text>
            </TouchableOpacity>
          </View>

          {/* Connection & VPS Lock Header */}
          <View style={styles.hudHeaderRow}>
            <View style={styles.statusBadge}>
              <View
                style={[
                  styles.statusDot,
                  { backgroundColor: connected ? (localized ? '#10b981' : '#f59e0b') : '#ef4444' },
                ]}
              />
              <Text style={styles.statusText}>
                {connected
                  ? localized
                    ? `3D LOCKED (${inliers} PTS • ${Math.round(latencyMs)}ms)`
                    : 'SCANNING DCC...'
                  : 'SERVER OFFLINE'}
              </Text>
            </View>

            <View style={{ flexDirection: 'row', gap: 6 }}>
              <TouchableOpacity
                onPress={() => setUseStreamOverlay((v) => !v)}
                style={styles.cfgBtn}
              >
                <Text style={styles.cfgBtnText}>{useStreamOverlay ? 'Cam: Live' : 'Cam: Raw'}</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => setShowUrlBar((s) => !s)} style={styles.cfgBtn}>
                <Text style={styles.cfgBtnText}>{showUrlBar ? 'Hide IP' : 'Server IP'}</Text>
              </TouchableOpacity>
            </View>
          </View>

          {showUrlBar && (
            <View style={styles.urlRow}>
              <TextInput
                style={styles.urlInput}
                value={wsUrl}
                onChangeText={setWsUrl}
                autoCapitalize="none"
                autoCorrect={false}
                placeholder="ws://129.161.52.222:8000/ws/ar"
                placeholderTextColor="#64748b"
              />
              <TouchableOpacity
                style={styles.connectBtn}
                onPress={() => connectWebSocket(wsUrl)}
              >
                <Text style={styles.connectBtnText}>Connect</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* LIVE FIRE ALERT BANNER (Synced with Commander Webpage & Firefighter Reports) */}
          {(alertBanner || (route?.fire_rooms && route.fire_rooms.length > 0)) && (
            <View style={styles.fireBanner}>
              <Text style={styles.fireBannerText} numberOfLines={1}>
                {route?.fire_rooms && route.fire_rooms.length > 0
                  ? `🔥 ACTIVE FIRE: ${route.fire_rooms.map((r) => r.label).join(', ')}`
                  : alertBanner}
              </Text>
            </View>
          )}

          {/* Pinpointed Room / Hallway Location in 3D DCC Model */}
          <Text style={styles.locationLabel}>CURRENT LOCATION (DCC FLOOR 2)</Text>
          <Text style={styles.roomTitle}>{currentLabel.toUpperCase()}</Text>

          {/* TASK 2: FIREFIGHTER TARGET ROOM SELECTOR */}
          {role === 'firefighter' && (
            <View style={styles.targetSelectorWrap}>
              <Text style={styles.targetSelectorLabel}>SELECT TARGET ROOM TO NAVIGATE:</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.roomChipScroll}>
                {QUICK_DCC_ROOMS.map((rm) => {
                  const isSelected = targetRoomId === rm.id;
                  const isOnFire = fireRoomIds.has(rm.id);
                  return (
                    <TouchableOpacity
                      key={rm.id}
                      onPress={() => switchMode('firefighter', rm.id)}
                      style={[
                        styles.roomChip,
                        isSelected && styles.roomChipSelected,
                        isOnFire && styles.roomChipFire,
                      ]}
                    >
                      <Text style={styles.roomChipText}>
                        {isOnFire ? `🔥 ${rm.short}` : rm.short}
                      </Text>
                    </TouchableOpacity>
                  );
                })}
              </ScrollView>
            </View>
          )}

          {/* Active 3D Route Bar (Closest Safe Exit for Occupants, Target Room for Firefighters) */}
          <View style={styles.egressRow}>
            <View style={{ flex: 1, paddingRight: 8 }}>
              <Text
                style={[
                  styles.egressTarget,
                  role === 'firefighter' && { color: '#fb923c' },
                  !route.reachable && { color: '#ef4444' },
                ]}
              >
                {route.arrived
                  ? 'ARRIVED AT 242 EAST LOBBY EXIT'
                  : !route.reachable
                  ? `⚠️ ${route.reason || 'NO SAFE ROUTE'}`
                  : role === 'firefighter'
                  ? `🎯 TARGET ROOM: ${(route.target_exit || targetRoomId).toUpperCase()}`
                  : `🚪 EXIT: ${(route.target_exit || 'EXIT').toUpperCase()}`}
              </Text>
              <Text style={styles.turnInstruction} numberOfLines={2}>
                {primaryInstruction}
              </Text>
            </View>
            <Text
              style={[
                styles.distanceBadge,
                role === 'firefighter' && { backgroundColor: 'rgba(249, 115, 22, 0.25)' },
              ]}
            >
              {route.arrived ? 'ARRIVED' : route.total_distance_m != null ? `${route.total_distance_m}m` : '—'}
            </Text>
          </View>
        </View>
      </SafeAreaView>

      {/* 4. LIVE DCC 200 LEVEL MINI-MAP (Shows Rooms, Active Fires in Red, and User Dot) */}
      <View style={styles.minimapCard}>
        <Text style={styles.minimapTitle}>DCC 3D MAP (LIVE FIRE SYNC)</Text>
        <View style={styles.minimapCanvas}>
          {/* Corridor 219 spine */}
          <View
            style={{
              position: 'absolute',
              left: mapX(-6),
              top: mapZ(-5) - 2,
              width: Math.max(20, mapX(38) - mapX(-6)),
              height: 4,
              backgroundColor: '#334155',
              borderRadius: 2,
            }}
          />
          {/* DCC Floor 2 Room Dots */}
          {(route.path_coords || []).slice(1).map((end, i) => {
            const start = route.path_coords[i];
            const x1 = mapX(start[0]), y1 = mapZ(start[2]);
            const x2 = mapX(end[0]), y2 = mapZ(end[2]);
            const length = Math.hypot(x2 - x1, y2 - y1);
            if (length < 0.1) return null;
            return <View key={`route-${i}`} style={{
              position: 'absolute', left: (x1 + x2) / 2 - length / 2,
              top: (y1 + y2) / 2 - 1.5, width: length, height: 3,
              backgroundColor: '#22d3ee', borderRadius: 2,
              transform: [{ rotate: `${Math.atan2(y2 - y1, x2 - x1)}rad` }],
            }} />;
          })}
          {role === 'occupant' && <View style={{
            position: 'absolute', left: mapX(44.19) - 5, top: mapZ(3.29) - 5,
            width: 10, height: 10, borderRadius: 2, backgroundColor: '#10b981',
            borderColor: '#ffffff', borderWidth: 1,
          }} />}
          {nodes.map((n) => (
            <View
              key={n.id}
              style={{
                position: 'absolute',
                left: mapX(n.pos[0]) - 3,
                top: mapZ(n.pos[2]) - 3,
                width: n.blocked ? 9 : n.is_exit ? 7 : 5,
                height: n.blocked ? 9 : n.is_exit ? 7 : 5,
                borderRadius: 5,
                backgroundColor: n.blocked
                  ? '#ef4444'
                  : n.status === 'smoke'
                  ? '#f59e0b'
                  : n.is_exit
                  ? '#10b981'
                  : '#64748b',
              }}
            />
          ))}
          {/* Live User Position Dot */}
          <View
            style={{
              position: 'absolute',
              left: mapX(position[0]) - 5,
              top: mapZ(position[2]) - 5,
              width: 10,
              height: 10,
              borderRadius: 5,
              backgroundColor: role === 'firefighter' ? '#f97316' : '#22d3ee',
              borderWidth: 1.5,
              borderColor: '#ffffff',
            }}
          />
        </View>
        <Text style={{ color: '#22d3ee', fontSize: 10, marginTop: 4 }}>
          {route.arrived ? 'Arrived at 242' : 'Cyan: route · Green: exit 242'}
        </Text>
      </View>

      {/* 5. BOTTOM ACTION BAR: Firefighter Live Fire Reporting OR Occupant Trapped/Mayday */}
      <View style={styles.bottomBar}>
        {role === 'firefighter' ? (
          <View style={styles.ffActionRow}>
            <TouchableOpacity
              activeOpacity={0.85}
              onPress={() => sendFireReport('fire')}
              style={styles.reportFireBtn}
            >
              <Text style={styles.reportBtnText}>🔥 REPORT FIRE HERE</Text>
              <Text style={styles.reportBtnSub}>{currentLabel}</Text>
            </TouchableOpacity>

            <TouchableOpacity
              activeOpacity={0.85}
              onPress={() => sendFireReport('clear')}
              style={styles.reportClearBtn}
            >
              <Text style={styles.reportBtnText}>✅ ROOM CLEAR</Text>
              <Text style={styles.reportBtnSub}>Mark Searched</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <TouchableOpacity
            activeOpacity={0.85}
            onPressIn={() => sendMayday(true)}
            onPressOut={() => sendMayday(false)}
            style={[styles.maydayBtn, mayday && styles.maydayBtnActive]}
          >
            <Text style={styles.maydayText}>
              {mayday ? '🚨 SENDING TRAPPED ALERT TO COMMAND...' : '🆘 HOLD IF TRAPPED / NEED RESCUE'}
            </Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#050811' },
  cameraFull: {
    flex: 1,
    width: SCREEN_W,
    height: SCREEN_H,
  },
  permissionContainer: {
    flex: 1,
    backgroundColor: '#070b14',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 28,
  },
  permTitle: { color: '#f97316', fontSize: 22, fontWeight: '900', marginBottom: 12 },
  permSub: { color: '#cbd5e1', fontSize: 15, textAlign: 'center', marginBottom: 24, lineHeight: 22 },
  permButton: {
    backgroundColor: '#f97316',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 12,
  },
  permButtonText: { color: '#fff', fontWeight: '800', fontSize: 16 },
  arFloorContainer: {
    position: 'absolute',
    bottom: SCREEN_H * 0.21,
    left: 0,
    right: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  chevronRow: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  chevronBox: {
    width: 68,
    height: 68,
    borderTopWidth: 14,
    borderRightWidth: 14,
    transform: [{ rotate: '-45deg' }],
    borderRadius: 6,
  },
  topHudWrapper: {
    position: 'absolute',
    top: Platform.OS === 'android' ? 32 : 0,
    left: 10,
    right: 10,
  },
  hudCard: {
    backgroundColor: 'rgba(11, 15, 25, 0.86)',
    borderRadius: 16,
    padding: 12,
    borderWidth: 1.5,
    borderColor: 'rgba(34, 211, 238, 0.55)',
  },
  hudCardFirefighter: {
    borderColor: 'rgba(249, 115, 22, 0.75)',
  },
  hudCardMayday: {
    borderColor: '#ef4444',
    backgroundColor: 'rgba(69, 10, 10, 0.92)',
  },
  modeSwitchRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 8,
  },
  modeBtn: {
    flex: 1,
    paddingVertical: 7,
    borderRadius: 8,
    backgroundColor: '#1e293b',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155',
  },
  modeBtnOccupantActive: {
    backgroundColor: '#0891b2',
    borderColor: '#22d3ee',
  },
  modeBtnFirefighterActive: {
    backgroundColor: '#ea580c',
    borderColor: '#fb923c',
  },
  modeBtnText: {
    color: '#ffffff',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.4,
  },
  hudHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  statusBadge: { flexDirection: 'row', alignItems: 'center' },
  statusDot: { width: 9, height: 9, borderRadius: 5, marginRight: 6 },
  statusText: { color: '#e2e8f0', fontSize: 11, fontWeight: '800', letterSpacing: 0.3 },
  cfgBtn: {
    backgroundColor: '#1e293b',
    paddingHorizontal: 9,
    paddingVertical: 4,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#475569',
  },
  cfgBtnText: { color: '#94a3b8', fontSize: 10, fontWeight: '700' },
  urlRow: { flexDirection: 'row', gap: 8, marginBottom: 8 },
  urlInput: {
    flex: 1,
    backgroundColor: '#0f172a',
    color: '#f8fafc',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#334155',
    fontSize: 12,
  },
  connectBtn: {
    backgroundColor: '#f97316',
    paddingHorizontal: 14,
    justifyContent: 'center',
    borderRadius: 8,
  },
  connectBtnText: { color: '#fff', fontWeight: '800', fontSize: 12 },
  fireBanner: {
    backgroundColor: 'rgba(220, 38, 38, 0.9)',
    paddingVertical: 5,
    paddingHorizontal: 10,
    borderRadius: 8,
    marginBottom: 6,
    borderWidth: 1,
    borderColor: '#fca5a5',
  },
  fireBannerText: {
    color: '#ffffff',
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.3,
  },
  locationLabel: { color: '#94a3b8', fontSize: 10, fontWeight: '800', letterSpacing: 0.7 },
  roomTitle: { color: '#ffffff', fontSize: 21, fontWeight: '900', marginTop: 1 },
  targetSelectorWrap: {
    marginTop: 8,
    paddingTop: 6,
    borderTopWidth: 1,
    borderTopColor: 'rgba(148, 163, 184, 0.2)',
  },
  targetSelectorLabel: {
    color: '#fdba74',
    fontSize: 10,
    fontWeight: '800',
    marginBottom: 5,
  },
  roomChipScroll: {
    flexDirection: 'row',
  },
  roomChip: {
    backgroundColor: '#1e293b',
    paddingHorizontal: 11,
    paddingVertical: 6,
    borderRadius: 8,
    marginRight: 6,
    borderWidth: 1,
    borderColor: '#475569',
  },
  roomChipSelected: {
    backgroundColor: '#ea580c',
    borderColor: '#fed7aa',
  },
  roomChipFire: {
    backgroundColor: '#991b1b',
    borderColor: '#f87171',
  },
  roomChipText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '800',
  },
  egressRow: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(148, 163, 184, 0.2)',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  egressTarget: { color: '#22d3ee', fontSize: 12, fontWeight: '900' },
  turnInstruction: { color: '#f8fafc', fontSize: 14, fontWeight: '800', marginTop: 2 },
  distanceBadge: {
    color: '#fff',
    fontSize: 19,
    fontWeight: '900',
    backgroundColor: 'rgba(34, 211, 238, 0.22)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
    overflow: 'hidden',
  },
  minimapCard: {
    position: 'absolute',
    right: 12,
    bottom: 102,
    backgroundColor: 'rgba(11, 15, 25, 0.88)',
    borderRadius: 12,
    padding: 8,
    borderWidth: 1,
    borderColor: '#334155',
  },
  minimapTitle: {
    color: '#94a3b8',
    fontSize: 8,
    fontWeight: '800',
    textAlign: 'center',
    marginBottom: 2,
  },
  minimapCanvas: {
    width: 155,
    height: 70,
    position: 'relative',
  },
  bottomBar: {
    position: 'absolute',
    bottom: 24,
    left: 14,
    right: 14,
  },
  ffActionRow: {
    flexDirection: 'row',
    gap: 10,
  },
  reportFireBtn: {
    flex: 1.2,
    backgroundColor: '#dc2626',
    paddingVertical: 12,
    borderRadius: 14,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#fca5a5',
  },
  reportClearBtn: {
    flex: 1,
    backgroundColor: '#15803d',
    paddingVertical: 12,
    borderRadius: 14,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#86efac',
  },
  reportBtnText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '900',
  },
  reportBtnSub: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: 10,
    fontWeight: '700',
    marginTop: 2,
  },
  maydayBtn: {
    backgroundColor: '#b91c1c',
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#f87171',
  },
  maydayBtnActive: {
    backgroundColor: '#ef4444',
    borderColor: '#ffffff',
  },
  maydayText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
});
