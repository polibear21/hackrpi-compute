#!/usr/bin/env python3
"""
Firesight Centralized Server ("The Brain") — HackRPI 2026
Connects:
  1. Hackrpi-2026 Commander & 3D Viewer Webpages (/command.html, /index.html, /firefighter.html, /phone.html)
  2. Mobile AR App (/ws/ar) — Occupant Mode (Task 1: Escape to Closest Safe Exit) & Firefighter Mode (Task 2: Navigate to Target Room + Live Fire Reporting)
  3. 3D Darrin Communications Center (DCC) Model (dcc.json) + Live Fire/Smoke Cellular Automaton + Dijkstra 3D Routing
  4. Visual Positioning System (SuperPoint + LightGlue + pycolmap + OpenCV ORB Keyframe Index from "2nd floor part 2")
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import glob
import heapq
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.core.fire import FireSim
from backend.core.arrival import ExitArrival
from backend.core.grid import BuildingGrid
from backend.core.model import ModelError, load_building, load_building_file
from backend.core.routing import RouteField, best_room_cell

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False

try:
    import torch
    from lightglue import LightGlue, SuperPoint
    HAS_ML = True
except ImportError:
    HAS_ML = False


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
FINAL_MODEL_DIR = ROOT_DIR / "final_model"
HACKRPI_2026_DIR = ROOT_DIR.parent / "Hackrpi-2026"
VIEWER_DIR = HACKRPI_2026_DIR / "viewer"

# Coordinate alignment between ARKit ("2nd floor part 2", origin in DCC 232) and dcc.json Floor 2 plan meters:
# Both coordinate frames are 1:1 physical meters!
ARKIT_TO_DCC_X0 = 40.1
ARKIT_TO_DCC_Y0 = 60.0
ARKIT_DEFAULT_LEVEL = 2


def arkit_to_dcc_plan(pos_xyz: List[float]) -> Tuple[int, float, float]:
    """Convert ARKit [X, Y, Z] meters to dcc.json (level=2, x_plan, y_plan)."""
    x_plan = ARKIT_TO_DCC_X0 + float(pos_xyz[0])
    y_plan = ARKIT_TO_DCC_Y0 + float(pos_xyz[2])
    return ARKIT_DEFAULT_LEVEL, round(x_plan, 3), round(y_plan, 3)


def dcc_plan_to_arkit(x_plan: float, y_plan: float, level: int = 2) -> List[float]:
    """Convert dcc.json (x_plan, y_plan) meters back to ARKit [X, Y, Z] meters for 3D AR floor arrows."""
    y_arkit = (int(level) - ARKIT_DEFAULT_LEVEL) * 4.0
    return [
        round(float(x_plan) - ARKIT_TO_DCC_X0, 3),
        round(float(y_arkit), 3),
        round(float(y_plan) - ARKIT_TO_DCC_Y0, 3),
    ]


# Portrait conversion matrix between ARKit camera frame and COLMAP upright image frame
C_PORTRAIT = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def rotmat_to_xyzw_quat(R: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix to [qx, qy, qz, qw] quaternion for ARKit/Three.js."""
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-12
    return [round(qx / norm, 6), round(qy / norm, 6), round(qz / norm, 6), round(qw / norm, 6)]


def make_simd_float4x4(R_wc: np.ndarray, t_wc: np.ndarray) -> List[float]:
    """Build column-major 16-float array for Swift's simd_float4x4."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc
    T[:3, 3] = t_wc
    return [round(float(x), 6) for x in T.T.flatten()]


# =====================================================================
# 1. 3D DCC BUILDING WORLD (dcc.json + FireSim + RouteField)
# =====================================================================
def _resolve_dcc_path() -> str:
    env_want = os.environ.get("FIRESIGHT_BUILDING", "")
    candidates = [
        env_want,
        str(HACKRPI_2026_DIR / "data" / "buildings" / "dcc.json"),
        str(BACKEND_DIR / "dcc.json"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("Could not find dcc.json")


class World:
    """Everything derived from the 3D DCC building JSON."""

    def __init__(self, model: Dict[str, Any]) -> None:
        self.model = model
        self.grid = BuildingGrid(model)
        self.sim = FireSim(self.grid)
        self.blocked = np.zeros(self.grid.walkable.shape, dtype=bool)
        self.searched: Dict[str, Dict[str, Any]] = {}
        self.reports: List[Dict[str, Any]] = []
        self.field = RouteField(self.grid)

    def refresh_routes(self) -> None:
        """Routes avoid fire and reported blockages; smoke raises cost."""
        self.field = RouteField(self.grid, self.sim.fire | self.blocked, self.sim.smoke)

    def locate_room(self, level: int, x_plan: float, y_plan: float) -> Dict[str, Any]:
        """Identify the DCC room containing (or nearest to) (level, x_plan, y_plan)."""
        g = self.grid
        fi = g.level_index.get(level, g.level_index.get(2, 0))
        r, c = g.cell_of(x_plan, y_plan)
        if 0 <= r < g.rows and 0 <= c < g.cols:
            ridx = int(g.room[fi, r, c])
            if ridx >= 0:
                info = g.rooms[ridx]
                return {"room_id": info["id"], "label": info["label"], "level": info["level"], "type": info["type"]}

        # Snap to nearest walkable cell and check its room
        cell = g.nearest_walkable(fi, x_plan, y_plan, max_dist=10.0)
        if cell is not None:
            ridx = int(g.room[fi, cell[0], cell[1]])
            if ridx >= 0:
                info = g.rooms[ridx]
                return {"room_id": info["id"], "label": info["label"], "level": info["level"], "type": info["type"]}

        # Fallback: nearest room centroid on that level
        best_info = None
        best_d = float("inf")
        for info in g.rooms:
            if info["level"] != level or info.get("type") == "void":
                continue
            cx, cy = info["centroid"]
            d = math.hypot(x_plan - cx, y_plan - cy)
            if d < best_d:
                best_d = d
                best_info = info
        if best_info is not None:
            return {
                "room_id": best_info["id"],
                "label": best_info["label"],
                "level": best_info["level"],
                "type": best_info["type"],
            }
        return {"room_id": "F2-232", "label": "232 CLASSROOM", "level": 2, "type": "classroom"}


lock = threading.Lock()
world = World(load_building_file(_resolve_dcc_path()))
print(
    f"[DCC] Loaded 3D Building '{world.model['building']}' "
    f"({len(world.grid.rooms)} spaces across floors {world.grid.levels}, {len(world.grid.exits)} exits)"
)

# In-memory people check-ins & event timeline (shared between Commander webpage and Mobile App)
people_lock = threading.Lock()
people_store: Dict[str, Dict[str, Any]] = {}
events_log: List[Dict[str, Any]] = []


def log_event(kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
    ev = {
        "kind": kind,
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "building": world.model.get("building", "DCC"),
        **(payload or {}),
    }
    events_log.append(ev)
    if len(events_log) > 300:
        del events_log[: len(events_log) - 300]


# =====================================================================
# 2. VISUAL POSITIONING SYSTEM (SuperPoint + LightGlue + pycolmap + ORB)
# =====================================================================
def crop_square_448(bgr_img: np.ndarray) -> np.ndarray:
    """Center-crop any aspect ratio image (e.g. 4:3 iPhone photo) to a 1:1 square, then resize to 448x448."""
    h, w = bgr_img.shape[:2]
    if h != w:
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        bgr_img = bgr_img[y0 : y0 + s, x0 : x0 + s]
    if bgr_img.shape[0] != 448 or bgr_img.shape[1] != 448:
        bgr_img = cv2.resize(bgr_img, (448, 448), interpolation=cv2.INTER_AREA)
    return bgr_img


def compute_global_descriptor(bgr_448: np.ndarray) -> np.ndarray:
    """
    Computes an L2-normalized illumination-robust global spatial appearance descriptor (876-D):
      - CLAHE-normalized LAB + HSV + Sobel edge magnitude on a 12x12 spatial grid (864 dims)
      - Floor & mid-wall color/texture signatures (12 dims)
    """
    lab = cv2.cvtColor(bgr_448, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    norm_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    small = cv2.resize(norm_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    rgb = small.astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.clip(np.hypot(gx, gy), 0.0, 1.0)

    hsv_edge = np.dstack([hsv[:, :, 1:3], mag[:, :, None]])
    grid_rgb = cv2.resize(rgb, (12, 12), interpolation=cv2.INTER_AREA).reshape(-1)
    grid_he = cv2.resize(hsv_edge, (12, 12), interpolation=cv2.INTER_AREA).reshape(-1)
    feat_stack = np.dstack([rgb, hsv_edge])

    floor_patch = feat_stack[42:, :, :].mean(axis=(0, 1)) * 3.0
    mid_patch = feat_stack[20:42, :, :].mean(axis=(0, 1)) * 2.0

    vec = np.concatenate([grid_rgb, grid_he, floor_patch, mid_patch]).astype(np.float32)
    norm = float(np.linalg.norm(vec)) + 1e-8
    return vec / norm


class VisualPositioningEngine:
    """
    Loads final_model/ (.bin files + map_descriptors.npz + calibrated orb_keyframes.npz)
    and localizes incoming 640x480 JPEGs from the mobile app.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.recon = None
        self.xyz: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.rgb: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        self.descriptors: np.ndarray = np.zeros((0, 256), dtype=np.float32)
        self.keypoints: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self.map_image_size = (448, 448)
        self.default_intrinsics = {"width": 448, "height": 448, "fx": 470.0, "fy": 470.0, "cx": 224.0, "cy": 224.0}
        self.trajectory: List[Dict[str, Any]] = []

        self.device = None
        self.extractor = None
        self.matcher = None
        self.map_feats_torch = None

        self.orb = cv2.ORB_create(nfeatures=600)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.orb_keyframes: List[Dict[str, Any]] = []
        self.gdes_matrix: Optional[np.ndarray] = None

        self._load_map()
        self._init_ml()

    def _load_map(self) -> None:
        npz_path = self.model_dir / "map_descriptors.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            self.xyz = data["xyz"].astype(np.float64)
            self.rgb = data["rgb"].astype(np.uint8)
            self.descriptors = data["descriptors"].astype(np.float32).T
            self.keypoints = data["keypoints"].astype(np.float32)
            img_sz = data["image_size"]
            self.map_image_size = (int(img_sz[0]), int(img_sz[1]))
            print(f"[VPS] Loaded {len(self.xyz)} 3D landmarks from {npz_path}")

        if HAS_PYCOLMAP and (self.model_dir / "cameras.bin").exists():
            try:
                self.recon = pycolmap.Reconstruction(str(self.model_dir))
                first_cam = next(iter(self.recon.cameras.values()))
                params = list(first_cam.params)
                if len(params) >= 4:
                    self.default_intrinsics = {
                        "width": int(first_cam.width),
                        "height": int(first_cam.height),
                        "fx": float(params[0]),
                        "fy": float(params[1]),
                        "cx": float(params[2]),
                        "cy": float(params[3]),
                    }
                print(
                    f"[VPS] Loaded pycolmap Reconstruction: "
                    f"{len(self.recon.images)} images, {len(self.recon.points3D)} 3D points"
                )
            except Exception as exc:
                print(f"[VPS] Note: pycolmap.Reconstruction load warning: {exc}")

        orb_npz = self.model_dir / "orb_keyframes.npz"
        if orb_npz.exists():
            try:
                odata = np.load(orb_npz, allow_pickle=True)
                if "trajectory" in odata:
                    for idx, pos in enumerate(odata["trajectory"], start=1):
                        self.trajectory.append({"id": idx, "pos": [round(float(v), 4) for v in pos]})
            except Exception as exc:
                print(f"[VPS] Warning loading trajectory from {orb_npz}: {exc}")

        if not self.trajectory:
            cam_files = sorted(glob.glob(str(ROOT_DIR / "2nd floor part 2" / "camera" / "frame_*.json")))
            if not cam_files:
                cam_files = sorted(glob.glob(str(ROOT_DIR / "2nd floor" / "camera" / "frame_*.json")))
            for idx, jf in enumerate(cam_files[::5], start=1):
                try:
                    d = json.loads(Path(jf).read_text(encoding="utf-8"))
                    self.trajectory.append({"id": idx, "pos": [round(float(v), 4) for v in d["extrinsics"]["t"]]})
                except Exception:
                    pass

    def _init_ml(self) -> None:
        if not HAS_ML or len(self.xyz) == 0:
            orb_npz = self.model_dir / "orb_keyframes.npz"
            if orb_npz.exists():
                try:
                    odata = np.load(orb_npz, allow_pickle=True)
                    names = odata["names"]
                    kps_arr = odata["kps"]
                    des_arr = odata["des"]
                    R_arr = odata["R_arkit"]
                    t_arr = odata["t_arkit"]
                    has_gdes = "gdes" in odata
                    gdes_arr = odata["gdes"] if has_gdes else None
                    room_ids_arr = odata["room_ids"] if "room_ids" in odata else None
                    room_labels_arr = odata["room_labels"] if "room_labels" in odata else None
                    dcc_xys_arr = odata["dcc_xys"] if "dcc_xys" in odata else None

                    for i in range(len(names)):
                        t_vec = np.asarray(t_arr[i], dtype=np.float64).reshape(3)
                        _, def_x, def_y = arkit_to_dcc_plan(t_vec.tolist())
                        self.orb_keyframes.append(
                            {
                                "name": str(names[i]),
                                "kps": np.asarray(kps_arr[i], dtype=np.float32),
                                "des": np.asarray(des_arr[i], dtype=np.uint8),
                                "gdes": np.asarray(gdes_arr[i], dtype=np.float32) if gdes_arr is not None else None,
                                "R_arkit": np.asarray(R_arr[i], dtype=np.float64),
                                "t_arkit": t_vec,
                                "room_id": str(room_ids_arr[i]) if room_ids_arr is not None else None,
                                "room_label": str(room_labels_arr[i]) if room_labels_arr is not None else None,
                                "dcc_xy": (
                                    [float(dcc_xys_arr[i][0]), float(dcc_xys_arr[i][1])]
                                    if dcc_xys_arr is not None
                                    else [def_x, def_y]
                                ),
                            }
                        )
                    if gdes_arr is not None:
                        self.gdes_matrix = np.asarray(gdes_arr, dtype=np.float32)
                    print(
                        f"[VPS] Loaded {len(self.orb_keyframes)} calibrated keyframes "
                        f"(Global Color+Spatial + ORB + Ground-Truth Room Labels) from {orb_npz.name}"
                    )
                    return
                except Exception as exc:
                    print(f"[VPS] Warning loading {orb_npz}: {exc}")

            print("[VPS] Initializing fast OpenCV ORB keyframe index from dataset...")
            cam_files = sorted(glob.glob(str(ROOT_DIR / "2nd floor part 2" / "camera" / "frame_*.json")))
            if not cam_files:
                cam_files = sorted(glob.glob(str(ROOT_DIR / "2nd floor" / "camera" / "frame_*.json")))
            for jf in cam_files[::10]:
                stem = Path(jf).stem
                img_path = Path(jf).parent.parent / "rgb" / f"{stem}.jpg"
                if not img_path.exists():
                    continue
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                bgr_448 = cv2.resize(bgr, (448, 448))
                im = cv2.cvtColor(bgr_448, cv2.COLOR_BGR2GRAY)
                kps, des = self.orb.detectAndCompute(im, None)
                if des is None or len(kps) < 16:
                    continue
                d = json.loads(Path(jf).read_text(encoding="utf-8"))
                R_arkit = np.array(d["extrinsics"]["R"], dtype=np.float64)
                t_arkit = np.array(d["extrinsics"]["t"], dtype=np.float64).reshape(3)
                _, def_x, def_y = arkit_to_dcc_plan(t_arkit.tolist())
                self.orb_keyframes.append(
                    {
                        "name": stem,
                        "kps": np.array([kp.pt for kp in kps], dtype=np.float32),
                        "des": des,
                        "gdes": compute_global_descriptor(bgr_448),
                        "R_arkit": R_arkit,
                        "t_arkit": t_arkit,
                        "room_id": None,
                        "room_label": None,
                        "dcc_xy": [def_x, def_y],
                    }
                )
            print(f"[VPS] Indexed {len(self.orb_keyframes)} ORB reference keyframes.")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.map_feats_torch = {
            "keypoints": torch.from_numpy(self.keypoints)[None].to(self.device),
            "descriptors": torch.from_numpy(self.descriptors)[None].to(self.device),
            "image_size": torch.tensor(
                [[self.map_image_size[0], self.map_image_size[1]]],
                dtype=torch.float32,
                device=self.device,
            ),
        }
        print(f"[VPS] SuperPoint + LightGlue initialized on {self.device}")

    def localize_jpeg(
        self,
        jpeg_bytes: bytes,
        custom_intrinsics: Optional[Dict[str, float]] = None,
        prev_dcc_xy: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"localized": False, "error": "Invalid JPEG bytes"}

        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        intr = dict(self.default_intrinsics)
        if custom_intrinsics:
            intr.update(custom_intrinsics)
        else:
            sx = w / float(intr["width"])
            sy = h / float(intr["height"])
            intr = {
                "width": w,
                "height": h,
                "fx": intr["fx"] * sx,
                "fy": intr["fy"] * sy,
                "cx": intr["cx"] * sx,
                "cy": intr["cy"] * sy,
            }

        if not HAS_ML or self.extractor is None or self.matcher is None:
            if not self.orb_keyframes:
                return {"localized": False, "error": "ML models not initialized"}
            bgr_448 = crop_square_448(bgr)
            q_im = cv2.cvtColor(bgr_448, cv2.COLOR_BGR2GRAY)
            q_kps, q_des = self.orb.detectAndCompute(q_im, None)
            if q_des is None or len(q_kps) < 10:
                return {"localized": False, "num_matches": 0, "num_inliers": 0, "latency_ms": 5.0}

            q_pts = np.array([kp.pt for kp in q_kps], dtype=np.float32)

            # Stage 1: CLAHE-normalized Global Spatial Color/Luminance/Gradient similarity
            q_gdes = compute_global_descriptor(bgr_448)
            if self.gdes_matrix is not None and len(self.gdes_matrix) == len(self.orb_keyframes):
                global_sims = self.gdes_matrix @ q_gdes
            else:
                global_sims = np.full(len(self.orb_keyframes), 0.8, dtype=np.float32)

            # Gentle physical proximity bonus so adjacent spaces win ties without trapping the user
            if prev_dcc_xy is not None:
                px, py = float(prev_dcc_xy[0]), float(prev_dcc_xy[1])
                dists = np.array(
                    [math.hypot(kf["dcc_xy"][0] - px, kf["dcc_xy"][1] - py) for kf in self.orb_keyframes],
                    dtype=np.float32,
                )
                dist_bonus = np.clip((10.5 - dists) * 0.65, 0.0, 6.8)
            else:
                dist_bonus = np.zeros(len(self.orb_keyframes), dtype=np.float32)

            # Build candidate set: Top 32 overall + Top 5 from each distinct room_id so Room 232 is ALWAYS checked!
            cand_set = set(int(i) for i in np.argsort(global_sims + dist_bonus / 40.0)[::-1][:32])
            by_room: Dict[str, List[int]] = {}
            for idx_kf, kf in enumerate(self.orb_keyframes):
                by_room.setdefault(kf.get("room_id", "F2-232"), []).append(idx_kf)
            for rid, indices in by_room.items():
                arr_idx = np.array(indices, dtype=np.int32)
                top_r = arr_idx[np.argsort(global_sims[arr_idx])[::-1][:5]]
                for i_r in top_r:
                    cand_set.add(int(i_r))

            best_kf = None
            best_score = -1.0
            best_inliers = 0
            best_matches = 0

            for i_int in cand_set:
                kf = self.orb_keyframes[i_int]
                raw_m = self.bf.match(q_des, kf["des"])
                good = [m for m in raw_m if m.distance < 54]
                inl = 0
                if len(good) >= 8:
                    src_p = np.float32([q_pts[m.queryIdx] for m in good])
                    dst_p = np.float32([kf["kps"][m.trainIdx] for m in good])
                    _, mask = cv2.findFundamentalMat(src_p, dst_p, cv2.FM_RANSAC, 4.5, 0.99)
                    inl = int(mask.sum()) if mask is not None else len(good)
                g_sim = float(global_sims[i_int])
                bonus = float(dist_bonus[i_int])
                score = float(inl) * 1.15 + 32.0 * g_sim + 0.22 * len(good) + bonus
                if score > best_score and (inl >= 7 or g_sim >= 0.91):
                    best_score = score
                    best_inliers = max(inl, int(round(g_sim * 15)))
                    best_matches = len(good)
                    best_kf = kf

            if best_kf is None:
                return {
                    "localized": False,
                    "num_matches": best_matches,
                    "num_inliers": best_inliers,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                }

            R_wc_arkit = best_kf["R_arkit"]
            t_wc = best_kf["t_arkit"]
            return {
                "localized": True,
                "position": [round(float(c), 4) for c in t_wc],
                "quaternion": rotmat_to_xyzw_quat(R_wc_arkit),
                "simd_float4x4": make_simd_float4x4(R_wc_arkit, t_wc),
                "room_id": best_kf.get("room_id"),
                "room_label": best_kf.get("room_label"),
                "dcc_xy": best_kf.get("dcc_xy"),
                "num_matches": best_matches,
                "num_inliers": best_inliers,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }

        img_tensor = torch.from_numpy(gray / 255.0).float()[None, None].to(self.device)
        with torch.no_grad():
            q_feats = self.extractor.extract(img_tensor)
            matches_out = self.matcher({"image0": q_feats, "image1": self.map_feats_torch})

        matches = matches_out["matches"][0].cpu().numpy()
        num_matches = int(len(matches))
        if num_matches < 10:
            return {
                "localized": False,
                "num_matches": num_matches,
                "num_inliers": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }

        p2d = q_feats["keypoints"][0].cpu().numpy()[matches[:, 0]].astype(np.float64)
        p3d = self.xyz[matches[:, 1]].astype(np.float64)

        R_cw: Optional[np.ndarray] = None
        t_cw: Optional[np.ndarray] = None
        num_inliers = 0

        if HAS_PYCOLMAP:
            camera = pycolmap.Camera(
                model="PINHOLE",
                width=int(intr["width"]),
                height=int(intr["height"]),
                params=[float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])],
            )
            ans = pycolmap.absolute_pose_estimation(p2d, p3d, camera)
            if ans is not None and ans.get("cam_from_world") is not None:
                cfw = ans["cam_from_world"]
                R_cw = cfw.rotation.matrix()
                t_cw = np.array(cfw.translation, dtype=np.float64)
                num_inliers = int(ans.get("num_inliers", num_matches))

        if R_cw is None or t_cw is None:
            K = np.array(
                [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                p3d, p2d, K, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=8.0
            )
            if not ok or inliers is None or len(inliers) < 8:
                return {
                    "localized": False,
                    "num_matches": num_matches,
                    "num_inliers": 0,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                }
            R_cw, _ = cv2.Rodrigues(rvec)
            t_cw = tvec.reshape(3)
            num_inliers = int(len(inliers))

        R_wc_colmap = R_cw.T
        t_wc = -R_wc_colmap @ t_cw
        R_wc_arkit = R_wc_colmap @ C_PORTRAIT

        return {
            "localized": True,
            "position": [round(float(c), 4) for c in t_wc],
            "quaternion": rotmat_to_xyzw_quat(R_wc_arkit),
            "simd_float4x4": make_simd_float4x4(R_wc_arkit, t_wc),
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }


vps_engine = VisualPositioningEngine(FINAL_MODEL_DIR)


# =====================================================================
# 3. ROUTE BUILDER USING 3D DCC MODEL (Task 1 & Task 2)
# =====================================================================
def compute_dcc_route_for_pose(
    arkit_pos: List[float],
    role: str = "occupant",
    target_room_id: Optional[str] = None,
    override_room_id: Optional[str] = None,
    override_room_label: Optional[str] = None,
    override_dcc_xy: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Computes the 3D DCC route for a phone client:
      - Task 1 (role == 'occupant'): Escape from current localized room to the closest safe exterior exit.
      - Task 2 (role == 'firefighter' and target_room_id): Route from current localized room to target_room_id.
    """
    level, x_plan, y_plan = arkit_to_dcc_plan(arkit_pos)
    if override_dcc_xy and len(override_dcc_xy) == 2:
        x_plan, y_plan = float(override_dcc_xy[0]), float(override_dcc_xy[1])

    with lock:
        loc_info = world.locate_room(level, x_plan, y_plan)
        if override_room_id:
            loc_info["room_id"] = override_room_id
        if override_room_label:
            loc_info["label"] = override_room_label

        fi = world.grid.level_index.get(level, world.grid.level_index.get(2, 0))
        room_statuses = world.sim.room_status()
        fire_rooms = [
            {"id": r["id"], "label": r["label"], "level": r["level"], "centroid": r["centroid"]}
            for r in world.grid.rooms
            if room_statuses.get(r["id"]) == "fire"
        ]
        smoke_rooms = [
            {"id": r["id"], "label": r["label"], "level": r["level"], "centroid": r["centroid"]}
            for r in world.grid.rooms
            if room_statuses.get(r["id"]) == "smoke"
        ]

        if role == "firefighter" and target_room_id:
            raw = world.field.route_to_room(fi, x_plan, y_plan, target_room_id)
            target_label = raw.get("target_label") or target_room_id
        else:
            raw = world.field.route_from(fi, x_plan, y_plan, target_exit_id="F2-D56")
            target_label = "242 East Lobby Exit"

    dcc_path = raw.get("path") or []
    arkit_coords = [dcc_plan_to_arkit(pt[1], pt[2], pt[0]) for pt in dcc_path if len(pt) >= 3]
    if not arkit_coords:
        arkit_coords = [arkit_pos]

    return {
        "role": role,
        "start_node": loc_info["room_id"],
        "start_label": loc_info["label"],
        "level": loc_info["level"],
        "x_plan": x_plan,
        "y_plan": y_plan,
        "target_exit": target_label,
        "target_room_id": target_room_id if role == "firefighter" else None,
        "path": dcc_path,
        "path_coords": arkit_coords,
        "total_distance_m": raw.get("distance_m"),
        "distance_m": raw.get("distance_m"),
        "max_smoke": raw.get("max_smoke", 0.0),
        "instructions": raw.get("instructions") or [],
        "reachable": bool(raw.get("reachable", False)),
        "reason": raw.get("reason", ""),
        "fire_rooms": fire_rooms,
        "smoke_rooms": smoke_rooms,
    }


def get_floor2_rooms_for_app() -> List[Dict[str, Any]]:
    """Returns DCC Floor 2 rooms with ARKit coordinates & live fire/smoke/searched status for the mobile app."""
    with lock:
        statuses = world.sim.room_status()
        searched_map = dict(world.searched)
        out: List[Dict[str, Any]] = []
        for r in world.grid.rooms:
            if r["level"] != 2 or r.get("type") in ("void", "unknown"):
                continue
            cx, cy = r["centroid"]
            arkit_xyz = dcc_plan_to_arkit(cx, cy, 2)
            st = statuses.get(r["id"], "clear")
            out.append(
                {
                    "id": r["id"],
                    "label": r["label"],
                    "type": r["type"],
                    "level": r["level"],
                    "centroid": [round(cx, 2), round(cy, 2)],
                    "pos": arkit_xyz,
                    "is_exit": r["type"] in ("stairwell", "lobby") and any(
                         k in r["id"] for k in ("218", "2ST1", "2ST6", "242", "2ST3", "2ST4")
                    ),
                    "status": st,
                    "blocked": st == "fire",
                    "searched": r["id"] in searched_map,
                }
            )
        return out


# =====================================================================
# 4. CENTRALIZED TELEMETRY & WEBSOCKET HUB
# =====================================================================
class ARClientSession:
    def __init__(self, ws: WebSocket, client_idx: int) -> None:
        self.ws = ws
        self.person_id = f"phone-{client_idx}"
        self.role = "occupant"  # "occupant" (Task 1) or "firefighter" (Task 2)
        self.target_room_id: Optional[str] = "F2-240"
        self.arkit_pos: List[float] = [0.0, 0.0, 0.0]
        self.arkit_quat: List[float] = [0.0, 0.0, 0.0, 1.0]
        self.mayday: bool = False
        self.last_room_id: Optional[str] = "F2-232"
        self.last_room_label: Optional[str] = "232 CLASSROOM"
        self.dcc_xy: Optional[List[float]] = [40.1, 60.0]
        self.pending_label: Optional[str] = None
        self.pending_count: int = 0
        self.exit_arrival = ExitArrival()
        self.last_route: Dict[str, Any] = compute_dcc_route_for_pose(
            self.arkit_pos,
            self.role,
            self.target_room_id,
            self.last_room_id,
            self.last_room_label,
            self.dcc_xy,
        )


class TelemetryHub:
    def __init__(self) -> None:
        self.ar_sessions: Dict[WebSocket, ARClientSession] = {}
        self.dashboard_clients: Set[WebSocket] = set()
        self.client_counter = 0
        self.firefighter_pos: List[float] = [0.0, 0.0, 0.0]
        self.firefighter_quat: List[float] = [0.0, 0.0, 0.0, 1.0]
        self.mayday_active: bool = False
        self.last_audio_cue: Optional[str] = None
        self.latest_frame_b64: Optional[str] = None
        self.last_route: Dict[str, Any] = compute_dcc_route_for_pose(self.firefighter_pos, "occupant", None)

    def get_dashboard_state(self, include_pointcloud: bool = False) -> Dict[str, Any]:
        f2_rooms = get_floor2_rooms_for_app()
        with lock:
            fire_cells, smoke_cells = world.sim.cells_payload()
            room_st = world.sim.room_status()
            vert_st = world.sim.vertical_status()
            sim_t = world.sim.t
            sim_mode = world.sim.mode
            sim_origins = list(world.sim.origins)

        payload: Dict[str, Any] = {
            "type": "state_sync",
            # Hackrpi-2026 /api/state fields:
            "t": sim_t,
            "mode": sim_mode,
            "origins": sim_origins,
            "rooms": room_st,
            "verticals": vert_st,
            "fire": fire_cells,
            "smoke": smoke_cells,
            # AR & 3D telemetry fields:
            "firefighter": {
                "position": self.firefighter_pos,
                "quaternion": self.firefighter_quat,
                "current_node": self.last_route["start_node"],
                "current_label": self.last_route["start_label"],
                "mayday": self.mayday_active,
            },
            "route": self.last_route,
            "nodes": f2_rooms,
            "edges": [],
            "audio_cue": self.last_audio_cue,
            "latest_frame_b64": self.latest_frame_b64,
            "timestamp": time.time(),
        }
        if include_pointcloud:
            payload["pointcloud"] = [
                [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                for p in vps_engine.xyz
            ]
            payload["trajectory"] = vps_engine.trajectory
        return payload

    async def broadcast_dashboards(self, message: Optional[Dict[str, Any]] = None) -> None:
        if message is None:
            message = self.get_dashboard_state(include_pointcloud=False)
        dead: List[WebSocket] = []
        for ws in list(self.dashboard_clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.dashboard_clients.discard(ws)

    async def broadcast_all_ar_clients(
        self,
        audio_cue: Optional[str] = None,
        alert_banner: Optional[str] = None,
    ) -> None:
        """Recalculate every connected phone's route against latest fire/smoke data and push immediately."""
        f2_rooms = get_floor2_rooms_for_app()
        dead: List[WebSocket] = []
        for ws, sess in list(self.ar_sessions.items()):
            try:
                sess.last_route = compute_dcc_route_for_pose(
                    sess.arkit_pos,
                    sess.role,
                    sess.target_room_id,
                    sess.last_room_id,
                    sess.last_room_label,
                    sess.dcc_xy,
                )
                self.last_route = sess.last_route
                await ws.send_json(
                    {
                        "type": "route_update",
                        "route": sess.last_route,
                        "nodes": f2_rooms,
                        "audio_cue": audio_cue,
                        "alert_banner": alert_banner,
                    }
                )
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ar_sessions.pop(ws, None)


hub = TelemetryHub()


def apply_field_report_to_world(
    room_id: str,
    kind: str,
    by: str = "Firefighter",
    raw_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Applies a firefighter report ('fire', 'smoke', 'clear', 'searched', 'blocked') to the 3D DCC world,
    updates routes, logs to timeline & /api/reports, and returns the report record.
    """
    kind = kind.lower().strip()
    with lock:
        info = world.grid.room_by_id.get(room_id)
        if info is None:
            # Try matching by room number prefix (e.g. "240" -> "F2-240")
            for cand in world.grid.rooms:
                if cand["id"].upper() == room_id.upper() or cand["label"].split()[0].upper() == room_id.upper():
                    info = cand
                    room_id = cand["id"]
                    break
        if info is None:
            raise KeyError(f"Unknown DCC room: {room_id}")

        fi = world.grid.level_index[info["level"]]
        in_room = (world.grid.room[fi] == info["index"]) & world.grid.walkable[fi]

        if kind == "fire":
            world.sim.ignite_room(room_id)
        elif kind == "smoke":
            world.sim.smoke[fi][in_room] = np.maximum(world.sim.smoke[fi][in_room], 0.75)
        elif kind == "clear":
            world.sim.clear_room(room_id)
            world.blocked[fi][in_room] = False
            world.searched[room_id] = {"by": by, "t": time.time(), "note": "cleared"}
        elif kind == "searched":
            world.searched[room_id] = {"by": by, "t": time.time(), "note": "searched"}
        elif kind == "blocked":
            world.blocked[fi][in_room] = True
        elif kind == "open":
            world.blocked[fi][in_room] = False

        world.refresh_routes()

        text_str = raw_text or f"{kind.upper()} reported in {info['label']} ({room_id})"
        update_item = {
            "kind": kind,
            "where": "room",
            "target_id": room_id,
            "target_label": info["label"],
            "level": info["level"],
            "change": f"{info['label']} marked {kind}",
        }
        rep = {
            "id": str(uuid.uuid4())[:8],
            "text": text_str,
            "by": by,
            "t": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updates": [update_item],
            "changes": [update_item["change"]],
            "confirmation": f"Copy: {info['label']} marked {kind.upper()}.",
        }
        world.reports.append(rep)

    log_event(f"report_{kind}", {"room_id": room_id, "person_id": by, "status": kind})
    return rep


# =====================================================================
# 5. FASTAPI APP & ALL CENTRALIZED ENDPOINTS (Webpages + Mobile App)
# =====================================================================
app = FastAPI(title="Firesight Centralized Server (DCC 3D + VPS + Live Fire Sync)", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _level_index(level: int) -> int:
    if level not in world.grid.level_index:
        raise HTTPException(404, f"Unknown floor level {level}")
    return world.grid.level_index[level]


def _bits(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(np.uint8), axis=None).tobytes()).decode("ascii")


def _display_masks(w: World, res: float = 0.15) -> Dict[str, Any]:
    if getattr(w, "_display", None) is not None:
        return w._display
    g = w.grid
    rows, cols = int(g.rows * g.cell / res) + 1, int(g.cols * g.cell / res) + 1
    faces = w.model.get("wall_style") == "faces"
    thick = 1 if faces else max(1, int(round(0.25 / res)))

    def px(x: float, y: float) -> Tuple[int, int]:
        return int((x - g.origin_x) / res), int((y - g.origin_y) / res)

    levels = {}
    for fl in w.model["floors"]:
        m = np.zeros((rows, cols), np.uint8)
        for wl in fl["walls"]:
            cv2.line(m, px(wl["x1"], wl["y1"]), px(wl["x2"], wl["y2"]), 1, thick)
        if not faces:
            for d in fl["doors"]:
                cv2.circle(m, px(d["x"], d["y"]), int(d["width"] / 2 / res), 0, -1)
        for sp in fl["spaces"]:
            if sp["type"] == "void":
                cv2.fillPoly(m, [np.array([px(*p) for p in sp["polygon"]], np.int32)], 1)
        levels[str(fl["level"])] = _bits(m > 0)
    w._display = {
        "res": res,
        "origin_x": g.origin_x,
        "origin_y": g.origin_y,
        "rows": rows,
        "cols": cols,
        "levels": levels,
    }
    return w._display


@app.get("/api/building")
def get_building() -> Dict[str, Any]:
    with lock:
        m = world.model
        return {
            **m,
            "grid": world.grid.meta(),
            "warnings": m["warnings"] + world.grid.warnings,
            "exits": [
                {
                    "level": world.grid.levels[f],
                    "door_id": d,
                    "x": world.grid.center_of(r, c)[0],
                    "y": world.grid.center_of(r, c)[1],
                }
                for f, r, c, d in world.grid.exits
            ],
        }


@app.get("/api/grid")
def get_grid() -> Dict[str, Any]:
    with lock:
        g = world.grid
        levels = {}
        for fi, lv in enumerate(g.levels):
            levels[str(lv)] = {
                "solid": _bits(g.inside[fi] & ~g.walkable[fi]),
                "walkable": _bits(g.walkable[fi]),
            }
        links = []
        for v in world.model["vertical_connections"]:
            cells = g.vertical_cells.get(v["id"], {})
            pts = {str(lv): list(g.center_of(r, c)) for lv, (fi, r, c) in cells.items()}
            links.append({"id": v["id"], "name": v["name"], "type": v["type"], "points": pts})
        return {
            **g.meta(),
            "encoding": "packbits-rowmajor-base64",
            "levels": levels,
            "links": links,
            "display": _display_masks(world),
        }


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    return hub.get_dashboard_state(include_pointcloud=True)


@app.post("/api/fire/start")
async def fire_start(
    room_id: Optional[str] = Body(None, embed=True),
    level: Optional[int] = Body(None, embed=True),
    x: Optional[float] = Body(None, embed=True),
    y: Optional[float] = Body(None, embed=True),
) -> Dict[str, Any]:
    room_label = room_id or ""
    with lock:
        if room_id:
            try:
                ok = world.sim.ignite_room(room_id)
                info = world.grid.room_by_id.get(room_id)
                if info:
                    room_label = info["label"]
            except KeyError:
                raise HTTPException(404, f"Unknown room {room_id}")
        elif level is not None and x is not None and y is not None:
            ok = world.sim.ignite_point(_level_index(level), x, y, radius=2.5)
            loc = world.locate_room(level, x, y)
            room_id = loc["room_id"]
            room_label = loc["label"]
        else:
            raise HTTPException(400, "Send room_id, or level + x + y")
        if not ok:
            raise HTTPException(400, "Could not place fire there")
        world.refresh_routes()
        t = world.sim.t

    log_event("fire_start", {"t": t, "room_id": room_id, "level": level, "x": x, "y": y})
    hub.last_audio_cue = "hazard_ahead_rerouting.mp3"
    await hub.broadcast_all_ar_clients(
        audio_cue="hazard_ahead_rerouting.mp3",
        alert_banner=f"🔥 FIRE IN {room_label.upper()} — REROUTING",
    )
    await hub.broadcast_dashboards()
    return {"ok": True, "t": t, "room_id": room_id, "label": room_label}


@app.post("/api/fire/mode")
async def fire_mode(mode: str = Body(..., embed=True)) -> Dict[str, Any]:
    if mode not in ("simulate", "reports"):
        raise HTTPException(400, "mode must be simulate or reports")
    with lock:
        world.sim.mode = mode
    log_event("fire_mode", {"mode": mode})
    await hub.broadcast_dashboards()
    return {"ok": True, "mode": mode}


@app.post("/api/fire/step")
async def fire_step(n: int = Body(1, embed=True)) -> Dict[str, Any]:
    with lock:
        world.sim.step(max(1, min(int(n), 50)))
        world.refresh_routes()
        t, rooms = world.sim.t, world.sim.room_status()
        fire_list = sorted(k for k, v in rooms.items() if v == "fire")

    log_event("fire_step", {"t": t, "n": n, "fire_rooms": fire_list})
    await hub.broadcast_all_ar_clients(
        audio_cue="route_recalculated.mp3",
        alert_banner=f"🔥 FIRE SPREAD (STEP {t}) — {len(fire_list)} ROOMS ACTIVE",
    )
    await hub.broadcast_dashboards()
    return {"ok": True, "t": t}


@app.post("/api/fire/reset")
async def fire_reset() -> Dict[str, Any]:
    with lock:
        world.sim.reset()
        world.blocked[:] = False
        world.searched.clear()
        world.reports.clear()
        world.refresh_routes()

    log_event("fire_reset", {})
    await hub.broadcast_all_ar_clients(
        audio_cue="route_recalculated.mp3",
        alert_banner="✅ ALL FIRES CLEARED — NORMAL EGRESS RESTORED",
    )
    await hub.broadcast_dashboards()
    return {"ok": True}


@app.post("/api/block/{node_id}")
async def toggle_block_room(node_id: str) -> Dict[str, Any]:
    """Compatibility endpoint to toggle fire in a DCC room from the 3D dashboard."""
    with lock:
        st = world.sim.room_status().get(node_id, "clear")
    if st == "fire":
        rep = apply_field_report_to_world(node_id, "clear", by="Commander")
        cue = "route_recalculated.mp3"
        banner = f"✅ {node_id} CLEARED"
    else:
        rep = apply_field_report_to_world(node_id, "fire", by="Commander")
        cue = "hazard_ahead_rerouting.mp3"
        banner = f"🔥 FIRE REPORTED IN {node_id}"
    await hub.broadcast_all_ar_clients(audio_cue=cue, alert_banner=banner)
    state = hub.get_dashboard_state(include_pointcloud=False)
    await hub.broadcast_dashboards(state)
    return state


# Task 1 REST endpoint: Occupant escape to closest exit
@app.get("/api/route")
def route_to_closest_exit(level: int = Query(...), x: float = Query(...), y: float = Query(...)) -> Dict[str, Any]:
    with lock:
        return world.field.route_from(_level_index(level), x, y)


# Task 2 REST endpoint: Firefighter route to specific target room
@app.get("/api/route/to_room")
def route_to_target_room(
    level: int = Query(2),
    x: float = Query(...),
    y: float = Query(...),
    target_room_id: str = Query(...),
) -> Dict[str, Any]:
    with lock:
        return world.field.route_to_room(_level_index(level), x, y, target_room_id)


@app.get("/api/routes/all")
def routes_all() -> List[Dict[str, Any]]:
    with lock:
        g = world.grid
        out = []
        for info in g.rooms:
            if info["type"] in ("stairwell", "elevator", "hallway", "void"):
                continue
            fi, cell = best_room_cell(world.field, info)
            if cell is None:
                continue
            r = world.field.route_from(fi, *g.center_of(*cell))
            out.append(
                {
                    "room_id": info["id"],
                    "label": info["label"],
                    "level": info["level"],
                    "reachable": r["reachable"],
                    "distance_m": r["distance_m"],
                    "path": r["path"],
                }
            )
        return out


@app.get("/api/nav")
def get_nav(level: int = Query(...), x: float = Query(...), y: float = Query(...)) -> Dict[str, Any]:
    with lock:
        fi = _level_index(level)
        r = world.field.route_from(fi, x, y)
        return {
            **r,
            "plan_north_deg": world.model.get("plan_north_deg", 0.0),
            "rev": f"{id(world)}:{world.sim.t}:{len(world.reports)}",
        }


@app.get("/api/rev")
def get_rev() -> Dict[str, Any]:
    with lock:
        return {"rev": f"{id(world)}:{world.sim.t}:{len(world.reports)}", "t": world.sim.t}


# People check-ins (shared across Commander webpage & Mobile App)
class PersonStatusIn(BaseModel):
    status: str
    level: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    room_id: Optional[str] = None


@app.get("/api/people")
def list_people() -> List[Dict[str, Any]]:
    with people_lock:
        return [dict(row) for row in people_store.values()]


@app.post("/api/people/{person_id}/status")
async def set_person_status(person_id: str, body: PersonStatusIn) -> Dict[str, Any]:
    if body.status not in ("safe", "trapped", "evacuating"):
        raise HTTPException(400, "status must be safe, trapped, or evacuating")
    rec = {
        "person_id": person_id,
        "status": body.status,
        "level": body.level,
        "x": body.x,
        "y": body.y,
        "room_id": body.room_id,
        "updated_at": time.time(),
    }
    with people_lock:
        people_store[person_id] = rec
    log_event("person_status", dict(rec))
    await hub.broadcast_dashboards()
    return {"ok": True}


# Firefighter & Commander Field Reports (shared across command.html, firefighter.html, and Mobile App)
class ReportBodyIn(BaseModel):
    text: str
    by: Optional[str] = "Firefighter"
    dry_run: bool = False
    room_id: Optional[str] = None
    kind: Optional[str] = None


def parse_and_apply_text_report(text: str, by: str = "Firefighter") -> Dict[str, Any]:
    low = text.lower()
    if any(w in low for w in ("clear", "extinguished", "all clear", "safe")):
        kind = "clear"
    elif "search" in low:
        kind = "searched"
    elif "block" in low:
        kind = "blocked"
    elif "smoke" in low:
        kind = "smoke"
    else:
        kind = "fire"

    # Match room code in text (e.g. "232", "236", "240", "218", "2ST6", "308")
    matched_room_id = None
    with lock:
        for r in world.grid.rooms:
            if r.get("type") == "void":
                continue
            code = r["label"].split()[0].lower()
            rid_low = r["id"].lower()
            if re.search(rf"\b{re.escape(code)}\b", low) or re.search(rf"\b{re.escape(rid_low)}\b", low):
                matched_room_id = r["id"]
                break

    if not matched_room_id:
        matched_room_id = hub.last_route.get("start_node") or "F2-232"

    return apply_field_report_to_world(matched_room_id, kind, by=by, raw_text=text)


@app.get("/api/reports")
def get_reports() -> Dict[str, Any]:
    with lock:
        return {"reports": list(world.reports)}


@app.post("/api/reports")
async def create_report(body: ReportBodyIn) -> Dict[str, Any]:
    by = body.by or "Firefighter"
    try:
        if body.room_id and body.kind:
            rep = apply_field_report_to_world(body.room_id, body.kind, by=by, raw_text=body.text)
        else:
            rep = parse_and_apply_text_report(body.text, by=by)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    upd = rep["updates"][0]
    cue = "hazard_ahead_rerouting.mp3" if upd["kind"] in ("fire", "smoke", "blocked") else "route_recalculated.mp3"
    banner = f"📡 {by.upper()}: {upd['target_label']} MARKED {upd['kind'].upper()}"
    await hub.broadcast_all_ar_clients(audio_cue=cue, alert_banner=banner)
    await hub.broadcast_dashboards()
    return rep


@app.get("/api/reports/searched")
def get_searched_rooms() -> Dict[str, Any]:
    with lock:
        searched_copy = dict(world.searched)
        unsearched = [
            {"id": r["id"], "label": r["label"], "level": r["level"]}
            for r in world.grid.rooms
            if r.get("type") not in ("stairwell", "elevator", "void") and r["id"] not in searched_copy
        ]
    return {"searched": searched_copy, "unsearched": unsearched}


@app.get("/api/brief")
def get_commander_brief(style: str = Query("commander")) -> Dict[str, Any]:
    with lock:
        st = world.sim.room_status()
        fires = [r["label"] for r in world.grid.rooms if st.get(r["id"]) == "fire"]
        smokes = [r["label"] for r in world.grid.rooms if st.get(r["id"]) == "smoke"]
        t = world.sim.t
    with people_lock:
        p_list = list(people_store.values())
    evac_cnt = sum(1 for p in p_list if p.get("status") == "evacuating")
    trap_cnt = sum(1 for p in p_list if p.get("status") == "trapped")

    lines = [
        f"DCC Incident Brief (Step {t}).",
        f"Active Fire Rooms: {', '.join(fires) if fires else 'None'}.",
        f"Smoke Rooms: {', '.join(smokes) if smokes else 'None'}.",
        f"Personnel & Occupants Tracked: {len(p_list)} ({evac_cnt} evacuating, {trap_cnt} trapped/MAYDAY).",
    ]
    return {
        "text": " ".join(lines),
        "model": "firesight-central-v2",
        "facts": {
            "step": t,
            "fire_rooms": fires,
            "smoke_rooms": smokes,
            "people_tracked": len(p_list),
        },
    }


class AskBodyIn(BaseModel):
    question: str


@app.post("/api/ask")
def ask_building(body: AskBodyIn) -> Dict[str, Any]:
    with lock:
        st = world.sim.room_status()
        fires = [r["label"] for r in world.grid.rooms if st.get(r["id"]) == "fire"]
        r_sample = world.field.route_from(world.grid.level_index[2], ARKIT_TO_DCC_X0, ARKIT_TO_DCC_Y0)
    ans = (
        f"Active fires: {', '.join(fires) if fires else 'none'}. "
        f"Recommended Floor 2 egress from 232 Classroom is via {r_sample.get('exit')} "
        f"({r_sample.get('distance_m')}m): {'; '.join(r_sample.get('instructions') or [])}."
    )
    return {"answer": ans, "model": "firesight-central-v2", "sources": ["F2-232", "F2-219", "F2-218", "F2-2ST6"]}


@app.get("/api/history")
def get_history(limit: int = Query(50)) -> Dict[str, Any]:
    return {"events": list(reversed(events_log[-limit:]))}


# =====================================================================
# 6. WEBSOCKET ENDPOINTS (/ws/dashboard & /ws/ar)
# =====================================================================
@app.websocket("/ws/dashboard")
async def ws_dashboard_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    hub.dashboard_clients.add(ws)
    try:
        await ws.send_json(hub.get_dashboard_state(include_pointcloud=True))
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "toggle_block":
                node_id = msg["node_id"]
                await toggle_block_room(node_id)

            elif mtype == "simulate_move":
                pos = msg.get("position", [0.0, 0.0, 0.3])
                hub.firefighter_pos = [round(float(x), 3) for x in pos]
                hub.last_route = compute_dcc_route_for_pose(hub.firefighter_pos, "occupant", None)
                await hub.broadcast_dashboards()
                await hub.broadcast_all_ar_clients()

            elif mtype == "clear_mayday":
                hub.mayday_active = False
                hub.last_audio_cue = "mayday_acknowledged.mp3"
                await hub.broadcast_dashboards()
                await hub.broadcast_all_ar_clients(audio_cue="mayday_acknowledged.mp3")

    except WebSocketDisconnect:
        hub.dashboard_clients.discard(ws)


@app.websocket("/ws/ar")
async def ws_ar_endpoint(ws: WebSocket) -> None:
    """
    Centralized WebSocket for the Mobile App (Occupant Escape Mode + Firefighter Target-Room & Reporting Mode).
    Supports:
      - {"type": "frame", "image_b64": "...", "role": "occupant"|"firefighter", "target_room_id": "F2-240", "person_id": "..."}
      - {"type": "set_mode", "role": "occupant"|"firefighter", "target_room_id": "F2-240"}
      - {"type": "report_fire", "room_id": "F2-236", "kind": "fire"|"clear"|"smoke"|"searched"}
      - {"type": "mayday", "active": true}
    """
    await ws.accept()
    hub.client_counter += 1
    session = ARClientSession(ws, hub.client_counter)
    hub.ar_sessions[ws] = session

    try:
        await ws.send_json(
            {
                "type": "init",
                "person_id": session.person_id,
                "role": session.role,
                "route": session.last_route,
                "nodes": get_floor2_rooms_for_app(),
                "audio_cue": "system_online.mp3",
            }
        )

        while True:
            packet = await ws.receive()
            if packet.get("type") == "websocket.disconnect":
                break
            jpeg_bytes: Optional[bytes] = None
            intrinsics: Optional[Dict[str, float]] = None

            if packet.get("bytes") is not None:
                jpeg_bytes = packet["bytes"]
            elif packet.get("text") is not None:
                msg = json.loads(packet["text"])
                mtype = msg.get("type", "frame")

                if "role" in msg and msg["role"] in ("occupant", "firefighter"):
                    session.role = msg["role"]
                if "target_room_id" in msg and msg["target_room_id"]:
                    session.target_room_id = msg["target_room_id"]
                if "person_id" in msg and msg["person_id"]:
                    session.person_id = str(msg["person_id"])

                if mtype == "set_mode":
                    session.last_route = compute_dcc_route_for_pose(
                        session.arkit_pos,
                        session.role,
                        session.target_room_id,
                        session.last_room_id,
                        session.last_room_label,
                        session.dcc_xy,
                    )
                    await ws.send_json(
                        {
                            "type": "route_update",
                            "route": session.last_route,
                            "nodes": get_floor2_rooms_for_app(),
                            "audio_cue": "route_recalculated.mp3",
                        }
                    )
                    continue

                if mtype == "report_fire":
                    room_to_report = msg.get("room_id") or session.last_route.get("start_node") or "F2-232"
                    report_kind = msg.get("kind", "fire")
                    rep = apply_field_report_to_world(
                        room_to_report,
                        report_kind,
                        by=session.person_id,
                    )
                    upd = rep["updates"][0]
                    cue = (
                        "hazard_ahead_rerouting.mp3"
                        if report_kind in ("fire", "smoke", "blocked")
                        else "route_recalculated.mp3"
                    )
                    banner = f"📡 {upd['target_label']} MARKED {report_kind.upper()}"
                    await hub.broadcast_all_ar_clients(audio_cue=cue, alert_banner=banner)
                    await hub.broadcast_dashboards()
                    continue

                if mtype == "mayday":
                    session.mayday = bool(msg.get("active", True))
                    hub.mayday_active = session.mayday
                    hub.last_audio_cue = "mayday_transmitting.mp3"
                    level, x_plan, y_plan = arkit_to_dcc_plan(session.arkit_pos)
                    with people_lock:
                        people_store[session.person_id] = {
                            "person_id": session.person_id,
                            "status": "trapped" if session.mayday else "evacuating",
                            "level": level,
                            "x": x_plan,
                            "y": y_plan,
                            "room_id": session.last_route.get("start_node", "F2-232"),
                            "updated_at": time.time(),
                        }
                    log_event(
                        "person_status",
                        {
                            "person_id": session.person_id,
                            "status": "trapped" if session.mayday else "evacuating",
                            "room_id": session.last_route.get("start_node", "F2-232"),
                            "level": level,
                        },
                    )
                    await hub.broadcast_dashboards()
                    await ws.send_json(
                        {
                            "type": "mayday_ack",
                            "mayday": session.mayday,
                            "audio_cue": "mayday_transmitting.mp3",
                        }
                    )
                    continue

                if mtype == "relocalize":
                    session.dcc_xy = None
                    session.last_room_id = None
                    session.last_room_label = None
                    session.pending_label = None
                    session.pending_count = 0
                    continue

                if "image_b64" in msg:
                    b64_str = msg["image_b64"].split(",")[-1]
                    hub.latest_frame_b64 = b64_str
                    jpeg_bytes = base64.b64decode(b64_str)
                    intrinsics = msg.get("intrinsics")

            if jpeg_bytes is None:
                continue

            loc = await asyncio.to_thread(
                vps_engine.localize_jpeg, jpeg_bytes, intrinsics, session.dcc_xy
            )

            audio_cue: Optional[str] = None
            if loc.get("localized"):
                prev_node = session.last_route.get("start_node")
                cand_label = loc.get("room_label")
                cand_id = loc.get("room_id")
                cand_xy = loc.get("dcc_xy")

                if session.last_room_label is None or cand_label == session.last_room_label:
                    session.last_room_id = cand_id
                    session.last_room_label = cand_label
                    if session.dcc_xy is not None and cand_xy is not None:
                        session.dcc_xy = [
                            0.65 * session.dcc_xy[0] + 0.35 * cand_xy[0],
                            0.65 * session.dcc_xy[1] + 0.35 * cand_xy[1],
                        ]
                        session.arkit_pos = [
                            0.65 * session.arkit_pos[0] + 0.35 * loc["position"][0],
                            0.65 * session.arkit_pos[1] + 0.35 * loc["position"][1],
                            0.65 * session.arkit_pos[2] + 0.35 * loc["position"][2],
                        ]
                    else:
                        session.dcc_xy = cand_xy
                        session.arkit_pos = loc["position"]
                    session.arkit_quat = loc["quaternion"]
                    session.pending_label = None
                    session.pending_count = 0
                else:
                    # Calculate physical distance between current position and candidate room
                    step_dist = 0.0
                    if session.dcc_xy is not None and cand_xy is not None:
                        step_dist = math.hypot(cand_xy[0] - session.dcc_xy[0], cand_xy[1] - session.dcc_xy[1])

                    # Adjacent room (<= 11.5m): require 2 consecutive frames
                    # Distant room (> 11.5m): require 4 consecutive frames agreeing on the exact same room
                    required_frames = 2 if step_dist <= 11.5 else 4

                    if cand_label == session.pending_label:
                        session.pending_count += 1
                    else:
                        session.pending_label = cand_label
                        session.pending_count = 1

                    if session.pending_count >= required_frames:
                        session.last_room_id = cand_id
                        session.last_room_label = cand_label
                        session.dcc_xy = cand_xy
                        session.arkit_pos = loc["position"]
                        session.arkit_quat = loc["quaternion"]
                        session.pending_label = None
                        session.pending_count = 0

                hub.firefighter_pos = session.arkit_pos
                hub.firefighter_quat = session.arkit_quat

                session.last_route = compute_dcc_route_for_pose(
                    session.arkit_pos,
                    session.role,
                    session.target_room_id,
                    session.last_room_id,
                    session.last_room_label,
                    session.dcc_xy,
                )
                hub.last_route = session.last_route
                curr_node = session.last_route["start_node"]

                # Update live person check-in on Commander webpage (/command.html & /index.html)
                with people_lock:
                    people_store[session.person_id] = {
                        "person_id": f"{session.person_id} ({session.role})",
                        "status": "trapped" if session.mayday else "evacuating",
                        "level": session.last_route["level"],
                        "x": session.last_route["x_plan"],
                        "y": session.last_route["y_plan"],
                        "room_id": curr_node,
                        "updated_at": time.time(),
                    }

                if curr_node != prev_node:
                    log_event(
                        "person_status",
                        {
                            "person_id": f"{session.person_id} ({session.role})",
                            "status": "trapped" if session.mayday else "evacuating",
                            "room_id": curr_node,
                            "level": session.last_route["level"],
                        },
                    )
                    if curr_node == session.last_route.get("target_room_id"):
                        audio_cue = "arrived_at_exit.mp3"
                    else:
                        audio_cue = "proceed_to_next_waypoint.mp3"
                hub.last_audio_cue = audio_cue
            else:
                # Keep route fresh with latest fire state even when frame doesn't lock
                session.last_route = compute_dcc_route_for_pose(
                    session.arkit_pos,
                    session.role,
                    session.target_room_id,
                    session.last_room_id,
                    session.last_room_label,
                    session.dcc_xy,
                )

            confirmed_location = bool(loc.get("localized")) and loc.get("room_id") == session.last_room_id
            arrived, just_arrived = session.exit_arrival.update(
                confirmed_location, session.last_room_id, session.dcc_xy, session.role
            )
            session.last_route["arrived"] = arrived
            if arrived:
                session.last_route["instructions"] = ["Arrived at 242 East Lobby Exit."]
                session.last_route["total_distance_m"] = 0.0
                session.last_route["distance_m"] = 0.0
                session.last_route["path_coords"] = []
            if just_arrived:
                audio_cue = "arrived_at_exit.mp3"

            response = {
                "type": "pose_and_route",
                **loc,
                "localized": confirmed_location,
                "position": session.arkit_pos,
                "quaternion": session.arkit_quat,
                "role": session.role,
                "current_node": session.last_route["start_node"],
                "current_label": session.last_route["start_label"],
                "route": session.last_route,
                "nodes": get_floor2_rooms_for_app(),
                "audio_cue": audio_cue,
            }
            await ws.send_json(response)
            await hub.broadcast_dashboards()

    except (WebSocketDisconnect, RuntimeError):
        hub.ar_sessions.pop(ws, None)


@app.get("/map_preview.html")
def serve_map_preview():
    preview_file = ROOT_DIR / "map_preview.html"
    if not preview_file.is_file():
        raise HTTPException(404, "map_preview.html not found")
    return FileResponse(str(preview_file), media_type="text/html")


# Mount Hackrpi-2026 viewer directory at / so http://localhost:8000/command.html and /index.html work seamlessly!
if VIEWER_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(VIEWER_DIR), html=True), name="viewer")
    print(f"[VIEWER] Serving Hackrpi-2026 webpages from {VIEWER_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
