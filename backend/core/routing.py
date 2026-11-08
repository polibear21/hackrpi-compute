"""Evacuation & Firefighter target-room routing on the 3D DCC building grid."""
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

SMOKE_PENALTY = 12.0
NO_PRED = -9999

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class RouteField:
    def __init__(self, grid, fire=None, smoke=None, use_elevators=False):
        self.grid = grid
        g = grid
        F, R, C = g.walkable.shape
        fire = np.zeros_like(g.walkable) if fire is None else fire
        smoke = np.zeros(g.walkable.shape, np.float32) if smoke is None else smoke
        self.passable = g.walkable & ~fire
        self.smoke = smoke
        idx = np.arange(F * R * C).reshape(F, R, C)

        rows, cols, wts = [], [], []
        P = self.passable
        for dr, dc in DIRS:
            r0, r1 = max(0, -dr), R - max(0, dr)
            c0, c1 = max(0, -dc), C - max(0, dc)
            a = P[:, r0:r1, c0:c1]
            b = P[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            ok = a & b
            if dr and dc:
                ok &= P[:, r0 + dr:r1 + dr, c0:c1] & P[:, r0:r1, c0 + dc:c1 + dc]
            step = g.cell * (math.sqrt(2) if dr and dc else 1.0)
            sa = smoke[:, r0:r1, c0:c1][ok]
            sb = smoke[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc][ok]
            rows.append(idx[:, r0:r1, c0:c1][ok])
            cols.append(idx[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc][ok])
            wts.append(step * (1.0 + SMOKE_PENALTY * (sa + sb) / 2.0))

        self.link_names = {}
        for (fa, ra, ca), (fb, rb, cb), cost, v in g.links:
            if v["type"] == "elevator" and not use_elevators:
                continue
            if not (P[fa, ra, ca] and P[fb, rb, cb]):
                continue
            a, b = g.flat(fa, ra, ca), g.flat(fb, rb, cb)
            if (a, b) in self.link_names:
                continue
            s = (smoke[fa, ra, ca] + smoke[fb, rb, cb]) / 2.0
            w = cost * (1.0 + SMOKE_PENALTY * s)
            rows += [np.array([a]), np.array([b])]
            cols += [np.array([b]), np.array([a])]
            wts += [np.array([w]), np.array([w])]
            self.link_names[(a, b)] = self.link_names[(b, a)] = v["name"]

        n = F * R * C
        self.graph = coo_matrix(
            (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsr()

        sources = [g.flat(fi, r, c) for fi, r, c, _ in g.exits if P[fi, r, c]]
        self.exit_ids = {g.flat(fi, r, c): did for fi, r, c, did in g.exits}
        if sources:
            self.dist, self.pred, _ = dijkstra(
                self.graph, directed=True, indices=sources,
                return_predecessors=True, min_only=True,
            )
        else:
            self.dist = np.full(n, np.inf)
            self.pred = np.full(n, NO_PRED)

    def route_from(self, fi, x, y, target_exit_id=None):
        """Mode 1 (Occupant): Escape route from (fi, x, y) to the closest safe exterior exit."""
        g = self.grid
        start = g.nearest_walkable(fi, x, y, max_dist=14.0, mask=self.passable[fi])
        if start is None:
            on_fire = False
            r, c = g.cell_of(x, y)
            if 0 <= r < g.rows and 0 <= c < g.cols:
                on_fire = bool(g.walkable[fi, r, c] and not self.passable[fi, r, c])
            return {"reachable": False, "reason": "in fire" if on_fire else "not on a walkable area",
                    "path": [], "distance_m": None, "instructions": []}
        node = g.flat(fi, *start)
        dist, pred = self.dist, self.pred
        if target_exit_id is not None:
            sources = [k for k, did in self.exit_ids.items()
                       if did == target_exit_id and self.passable[g.unflat(k)]]
            if not sources:
                return {"reachable": False, "reason": "selected exit unavailable",
                        "path": [], "distance_m": None, "instructions": []}
            dist, pred, _ = dijkstra(
                self.graph, directed=True, indices=sources,
                return_predecessors=True, min_only=True,
            )
        if not math.isfinite(dist[node]):
            return {"reachable": False, "reason": "no safe route to any exit",
                    "path": [], "distance_m": None, "instructions": []}
        nodes = [node]
        while pred[node] != NO_PRED and len(nodes) < 1_000_000:
            node = int(pred[node])
            nodes.append(node)
        cells = [g.unflat(k) for k in nodes]
        return {
            "reachable": True,
            "reason": "",
            "path": self._simplify(cells),
            "distance_m": round(self._length(cells), 1),
            "max_smoke": round(float(max(self.smoke[c] for c in cells)), 2),
            "exit": self.exit_ids.get(nodes[-1]),
            "instructions": self._instructions(cells, nodes),
        }

    def route_to_room(self, fi, x, y, target_room_id):
        """Mode 2 (Firefighter): Shortest safe route from (fi, x, y) to a specific room in the 3D DCC model."""
        g = self.grid
        info = g.room_by_id.get(target_room_id)
        if info is None:
            return {"reachable": False, "reason": f"Unknown room {target_room_id}",
                    "path": [], "distance_m": None, "instructions": []}

        start = g.nearest_walkable(fi, x, y, max_dist=14.0, mask=self.passable[fi])
        if start is None:
            return {"reachable": False, "reason": "Start position not on walkable area",
                    "path": [], "distance_m": None, "instructions": []}

        t_fi = g.level_index[info["level"]]
        in_room = (g.room[t_fi] == info["index"]) & self.passable[t_fi]
        if not in_room.any():
            # If the target room itself is on fire, target the nearest passable cell outside its door/centroid
            cx, cy = info["centroid"]
            t_cell = g.nearest_walkable(t_fi, cx, cy, max_dist=12.0, mask=self.passable[t_fi])
            if t_cell is None:
                return {"reachable": False, "reason": f"Room {info['label']} completely unreachable",
                        "path": [], "distance_m": None, "instructions": []}
            target_nodes = [g.flat(t_fi, t_cell[0], t_cell[1])]
        else:
            rs, cs = np.nonzero(in_room)
            target_nodes = [g.flat(t_fi, int(r), int(c)) for r, c in zip(rs, cs)]

        dist, pred, _ = dijkstra(
            self.graph, directed=True, indices=target_nodes,
            return_predecessors=True, min_only=True,
        )
        start_node = g.flat(fi, *start)
        if not math.isfinite(dist[start_node]):
            return {"reachable": False, "reason": f"No safe path to {info['label']}",
                    "path": [], "distance_m": None, "instructions": []}

        nodes = [start_node]
        curr = start_node
        while pred[curr] != NO_PRED and len(nodes) < 1_000_000:
            curr = int(pred[curr])
            nodes.append(curr)
        cells = [g.unflat(k) for k in nodes]
        steps = self._instructions(cells, nodes, destination_label=info["label"])
        return {
            "reachable": True,
            "reason": "",
            "target_room_id": target_room_id,
            "target_label": info["label"],
            "path": self._simplify(cells),
            "distance_m": round(self._length(cells), 1),
            "max_smoke": round(float(max(self.smoke[c] for c in cells)), 2),
            "exit": info["label"],
            "instructions": steps,
        }

    def _length(self, cells):
        g = self.grid
        total = 0.0
        link_cost = {(l[0], l[1]): l[2] for l in g.links}
        link_cost.update({(l[1], l[0]): l[2] for l in g.links})
        for a, b in zip(cells, cells[1:]):
            if a[0] == b[0]:
                total += g.cell * math.hypot(a[1] - b[1], a[2] - b[2])
            else:
                total += link_cost.get((tuple(a), tuple(b)), 5.0)
        return total

    def _simplify(self, cells):
        g = self.grid
        if not cells:
            return []
        keep = [cells[0]]
        for prev, cur, nxt in zip(cells, cells[1:], cells[2:]):
            if cur[0] != prev[0] or cur[0] != nxt[0]:
                keep.append(cur); continue
            d1 = (cur[1] - prev[1], cur[2] - prev[2])
            d2 = (nxt[1] - cur[1], nxt[2] - cur[2])
            if d1 != d2:
                keep.append(cur)
        if len(cells) > 1:
            keep.append(cells[-1])
        out = []
        for fi, r, c in keep:
            x, y = g.center_of(r, c)
            out.append([g.levels[fi], round(x, 2), round(y, 2)])
        return out

    def _instructions(self, cells, nodes, destination_label=None):
        g = self.grid
        steps = []
        seg_room, seg_len = None, 0.0

        def flush():
            if seg_room is not None and seg_len >= 1.0:
                info = g.rooms[seg_room]
                name = info["label"] if info["type"] != "hallway" else f"the hallway ({info['label']})"
                steps.append(f"Walk {round(seg_len)} m through {name}")

        for i, (fi, r, c) in enumerate(cells):
            if i > 0:
                pfi, pr, pc = cells[i - 1]
                if pfi != fi:
                    flush(); seg_room, seg_len = None, 0.0
                    name = self.link_names.get((nodes[i - 1], nodes[i]), "stairs")
                    direction = "down" if fi < pfi else "up"
                    steps.append(f"Take {name} {direction} to floor {g.levels[fi]}")
                    continue
                seg_len += g.cell * math.hypot(r - pr, c - pc)
            rid = int(g.room[fi, r, c])
            if rid != -1 and rid != seg_room:
                if seg_room is not None:
                    flush()
                    seg_len = 0.0
                seg_room = rid
        flush()
        if destination_label:
            steps.append(f"Arrive at {destination_label}")
        else:
            exit_id = self.exit_ids.get(nodes[-1], "the exit")
            steps.append(f"Exit the building at {exit_id}")
        return steps


def best_room_cell(field, info):
    g = field.grid
    fi = g.level_index[info["level"]]
    in_room = (g.room[fi] == info["index"]) & g.walkable[fi]
    if not in_room.any():
        return fi, None
    dist = field.dist.reshape(g.walkable.shape)[fi]
    ok = in_room & np.isfinite(dist)
    pool = ok if ok.any() else in_room
    cx, cy = info["centroid"]
    d2 = np.where(pool, (g.X - cx) ** 2 + (g.Y - cy) ** 2, np.inf)
    k = int(np.argmin(d2))
    return fi, divmod(k, g.cols)
