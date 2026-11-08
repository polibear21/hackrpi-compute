const assert = require('node:assert/strict');
const { nextRoutePoint, smoothTurn } = require('./navigation');

// Ignore waypoints behind the user and look through segment boundaries.
assert.deepEqual(nextRoutePoint([[0, 0, 0], [10, 0, 0], [20, 0, 0]], [9, 0, 0]), [11, 0, 0]);
assert.deepEqual(nextRoutePoint([[0, 0, 0], [10, 0, 0], [10, 0, 10]], [10, 0, 8]), [10, 0, 10]);
assert.deepEqual(nextRoutePoint([[0, 0, 0], [0, 0, 0], [10, 0, 0]], [0, 0, 0]), [2, 0, 0]);
assert.equal(nextRoutePoint([], [0, 0, 0]), null);
// Small changes stay inside the deadband, including across +/-180.
assert.equal(smoothTurn(179, -179), 179);
assert.equal(smoothTurn(170, -170), 175);
assert.equal(smoothTurn(0, 40), 10);
console.log('Navigation checks passed');
