// Follow the route in travel order, looking ahead from its nearest segment.
function nextRoutePoint(coords, position, lookAhead = 2) {
  if (!coords.length) return null;
  let best = Infinity;
  let segment = 0;
  let projected = coords[0];
  for (let i = 0; i < coords.length - 1; i++) {
    const a = coords[i], b = coords[i + 1];
    const dx = b[0] - a[0], dz = b[2] - a[2];
    const length2 = dx * dx + dz * dz;
    const t = length2 ? Math.max(0, Math.min(1,
      ((position[0] - a[0]) * dx + (position[2] - a[2]) * dz) / length2)) : 0;
    const p = [a[0] + t * dx, a[1] + t * (b[1] - a[1]), a[2] + t * dz];
    const distance = Math.hypot(position[0] - p[0], position[2] - p[2]);
    if (distance < best) { best = distance; segment = i; projected = p; }
  }
  for (let i = segment + 1; i < coords.length; i++) {
    const end = coords[i];
    const length = Math.hypot(end[0] - projected[0], end[2] - projected[2]);
    if (length >= lookAhead) {
      const t = lookAhead / length;
      return projected.map((v, axis) => v + t * (end[axis] - v));
    }
    lookAhead -= length;
    projected = end;
  }
  return coords[coords.length - 1];
}

function smoothTurn(previous, desired) {
  // Unwrap around 180 degrees so tiny pose changes cannot flip left/right.
  const delta = ((desired - previous + 540) % 360) - 180;
  return Math.abs(delta) < 3 ? previous : previous + delta * 0.25;
}

module.exports = { nextRoutePoint, smoothTurn };
