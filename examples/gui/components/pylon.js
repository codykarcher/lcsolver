/**
 * Engine pylon.
 *
 * Carries a podded engine from a wing. It has no size of its own: the root is
 * found on the engine's core and the top is put wherever the wing is, so the
 * only inputs are the two things it spans. That is why this takes an engine
 * object rather than numbers -- it reads the core line out of the engine's own
 * published geometry instead of being told about it.
 *
 * Three parts, bottom to top:
 *
 *   root      carried down to the core CENTRELINE, its chord spanning the core
 *             case end to end, so the strut breaks the core surface just
 *             behind the fan and again at the back of the case -- those two
 *             places being the mounts
 *   strut     a streamwise section lofted upward, PIERCING the nacelle. Its
 *             leading edge is vertical until clear of the cowl and only rakes
 *             forward above it.
 *   flat top  the face that meets the wing underside
 *
 * The strut passing through the nacelle is deliberate and not a clash. On a
 * real installation the pylon is structure and the nacelle is fairing: the
 * cowl is cut around the pylon, not the other way about. Modelling the cutout
 * would mean booleans on the cowl for no visible gain at this fidelity, so the
 * two simply intersect and read correctly.
 *
 * Frame is the engine's: origin at the fan plane, +Z forward, +Y up. Add the
 * pylon to the same parent as the engine and it lands in the right place.
 */
import * as THREE from 'three';
import * as M from './materials.js';

/** Proportions, x fan radius unless noted. Visual, not structural. */
const P = {
  // Root chord is read from the engine, not set here -- see below. These are
  // only the fallback for an engine that publishes no core line.
  rootZ0:   -0.30,
  rootZ1:   -2.90,
  rootT:     0.085, // root half-thickness
  topZ0:     0.45,  // top chord at the wing, forward end
  topZ1:    -2.35,  // ... and aft
  topT:      0.130, // top half-thickness
  attachY:   1.95,  // wing underside, above the engine axis
  noseA:     0.010, // section nose bluntness
  teFrac:    0.45,  // trailing-edge thickness, fraction of maximum
  kink:      1.02,  // leading-edge kink height, x nacelle max radius
  below:     5,     // stations below the kink
  above:     6,     // ... and above it
  nSide:     22,    // section points per side
  fillet:    0.14,  // corner radius on the over-mount outline
  reach:     1.85,  // side-mount: out to the fuselage side
  sideSweep: 0.10,  // ... and how far the trailing edge runs aft at the wall,
                    //     as a fraction of NACELLE length
  sideStations: 10,
};

/** Half-thickness of the streamwise section at chord fraction `s`. */
function halfT(s, tMax) {
  return tMax * Math.sqrt(s / (s + P.noseA)) * (1 - (1 - P.teFrac) * s * s);
}

/**
 * @param {THREE.Object3D} engine  a podded or bare turbofan
 * @param {object} [opts]
 * @param {number} [opts.attachY]  wing underside height, engine radii
 * @param {number} [opts.topZ0]    forward end of the wing attachment
 * @param {number} [opts.topZ1]    aft end
 */
export function underMountPylon(engine, opts = {}) {
  const u = engine.userData;
  const R = u.rFan ?? (u.rMax ? u.rMax / 1.037 : 1);
  const g = new THREE.Group();

  const engineZ = opts.engineZ ?? 0;
  const attachY = (opts.attachY ?? P.attachY) * R;
  const topZ0 = (opts.topZ0 ?? P.topZ0) * R;
  const topZ1 = (opts.topZ1 ?? P.topZ1) * R;

  // Root chord, in the PARENT frame: the engine's own core stations shifted by
  // wherever the engine has been put. Forward mount just behind the fan case
  // -- the fan's outer shell, not the fan itself -- and aft at the core
  // outlet.
  const w = u.coreWall;
  const rootZ0 = (u.caseAft ?? P.rootZ0 * R) + engineZ;
  const rootZ1 = (w ? w[w.length - 1][0] : P.rootZ1 * R) + engineZ;

  // Leading edge runs VERTICALLY from the root up through the nacelle, and
  // only changes angle once clear of it. A pylon that starts raking the moment
  // it leaves the core would be cutting across the cowl at a slant; in life it
  // goes straight up through the fairing and sweeps forward above it.
  const yKink = (u.nacelleMaxRadius ?? 1.22 * R) * P.kink;

  // Stations: closely spaced up to the kink so it stays a crease rather than
  // being rounded off by the loft, then out to the wing.
  const ys = [];
  for (let i = 0; i < P.below; i++) ys.push((i / P.below) * yKink);
  for (let i = 0; i <= P.above; i++) {
    ys.push(yKink + (i / P.above) * (attachY - yKink));
  }

  const nP = P.nSide;
  const pos = [], idx = [], ring = [];

  for (const y of ys) {
    const tAll = y / attachY;                      // 0 at the root, 1 at the wing
    const e = tAll * tAll * (3 - 2 * tAll);
    // Leading edge: held at the root station until the kink, then straight to
    // the wing. Trailing edge sweeps the whole way.
    const zLE = y <= yKink ? rootZ0
      : rootZ0 + (topZ0 - rootZ0) * ((y - yKink) / (attachY - yKink));
    const zTE = rootZ1 + (topZ1 - rootZ1) * e;
    const c = zLE - zTE;
    const tMax = (P.rootT + (P.topT - P.rootT) * e) * R;

    ring.push(pos.length / 3);
    for (let side = 0; side < 2; side++) {
      for (let j = 0; j <= nP; j++) {
        if (side === 1 && (j === 0 || j === nP)) continue;
        const sv = side === 0 ? j / nP : 1 - j / nP;
        pos.push((side === 0 ? 1 : -1) * halfT(sv, tMax), y, zLE - sv * c);
      }
    }
  }
  const perRing = 2 * (nP + 1) - 2;
  const nS = ys.length;

  for (let i = 0; i < nS - 1; i++) {
    for (let j = 0; j < perRing; j++) {
      const a = ring[i] + j, b = ring[i] + ((j + 1) % perRing);
      const a2 = ring[i + 1] + j, b2 = ring[i + 1] + ((j + 1) % perRing);
      idx.push(a, b, a2, b, b2, a2);
    }
  }
  for (const [base, flip] of [[ring[0], true], [ring[nS - 1], false]]) {
    for (let j = 1; j < perRing - 1; j++) {
      if (flip) idx.push(base, base + j + 1, base + j);
      else idx.push(base, base + j, base + j + 1);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  const strut = new THREE.Mesh(geo, M.skin);
  strut.name = 'pylonStrut';
  g.add(strut);


  // No plate: the pylon runs up to the wing and stops there. It had one so the
  // top had something to end on when there was no wing in the scene, and with a
  // wing above it that plate is a slab sitting on the skin.
  g.userData.attachY = attachY;
  g.userData.attachZ = [topZ0, topZ1];
  g.userData.rootZ = [rootZ0, rootZ1];
  g.userData.kinkY = yKink;
  g.name = 'underMountPylon';
  return g;
}

/**
 * Reverse a geometry's winding if it faces inward.
 *
 * Cheaper than reasoning about handedness. The side-mount stacks its sections
 * along X where the others stack along Y, which flips the sense of the same
 * index pattern -- and getting that wrong costs a whole component rendered
 * inside out.
 */
function faceOutward(geo) {
  if (volumeOf(geo) >= 0) return geo;
  const idx = geo.getIndex();
  if (idx) {
    const a = idx.array;
    for (let t = 0; t + 2 < a.length; t += 3) {
      const tmp = a[t + 1]; a[t + 1] = a[t + 2]; a[t + 2] = tmp;
    }
    idx.needsUpdate = true;
    geo.computeVertexNormals();
  }
  return geo;
}

/**
 * Signed volume; negative means the winding faces inward.
 *
 * Handles non-indexed geometry, which ExtrudeGeometry produces. An earlier
 * version returned 1 when there was no index buffer, so the extruded fairing
 * reported a clean orientation without ever being looked at.
 */
function volumeOf(geo) {
  const pos = geo.getAttribute('position');
  if (!pos) return 0;
  const idx = geo.getIndex();
  const n = pos.count, count = idx ? idx.count : n;
  const at = (k) => (idx ? idx.getX(k) : k);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const x = new THREE.Vector3();
  let v = 0;
  for (let t = 0; t + 2 < count; t += 3) {
    a.fromBufferAttribute(pos, at(t));
    b.fromBufferAttribute(pos, at(t + 1));
    c.fromBufferAttribute(pos, at(t + 2));
    v += a.dot(x.crossVectors(b, c)) / 6;
  }
  return v;
}

/**
 * Replace sharp corners of a polygon with quadratic fillets.
 *
 * Only where there is actually a corner: the cowl and core runs arrive here as
 * dense polylines of nearly-collinear points, and filleting every vertex would
 * spend arcs on straight sections while doing nothing about the joints that
 * look blocky. The turn angle decides.
 *
 * Radius is clamped to under half of each adjacent edge, so a fillet can never
 * eat past the neighbouring corner however short the segment.
 */
function roundCorners(pts, radius, minTurn = 0.30, seg = 7) {
  const out = [];
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    const a = pts[(i - 1 + n) % n], b = pts[(i + 1) % n];
    const v1 = new THREE.Vector2().subVectors(a, p);
    const v2 = new THREE.Vector2().subVectors(b, p);
    const l1 = v1.length(), l2 = v2.length();
    if (l1 < 1e-9 || l2 < 1e-9) { out.push(p); continue; }
    v1.divideScalar(l1); v2.divideScalar(l2);
    const turn = Math.PI - Math.acos(Math.max(-1, Math.min(1, v1.dot(v2))));
    if (turn < minTurn) { out.push(p); continue; }
    const r = Math.min(radius, l1 * 0.45, l2 * 0.45);
    const p1 = p.clone().addScaledVector(v1, r);
    const p2 = p.clone().addScaledVector(v2, r);
    for (let k = 0; k <= seg; k++) {
      const t = k / seg, m = 1 - t;
      out.push(new THREE.Vector2(m * m * p1.x + 2 * m * t * p.x + t * t * p2.x,
                                 m * m * p1.y + 2 * m * t * p.y + t * t * p2.y));
    }
  }
  return out;
}

/**
 * Over-mount pylon: the engine sits above the wing and the pylon hangs down.
 *
 * ONE closed outline, extruded. Built as a strut plus a separate fairing it
 * rendered as two overlapping shapes with no way to tell which surface you
 * were looking at; as a single polygon there is one body and one silhouette.
 *
 * The outline, forward to aft along the top and back along the bottom:
 *
 *   wing, forward   ---LE line--->   nacelle leading edge
 *   nacelle leading edge  ---along the cowl--->  back of the fan case
 *   back of the fan case at the NACELLE  --->  at the CORE CASE   (vertical)
 *   along the core case  --->  trailing edge of the core case
 *   trailing edge of the core case  --->  trailing edge of the NACELLE
 *   nacelle trailing edge  --->  wing, aft
 *   along the wing back to the start
 *
 * Every vertex is an engine feature or a wing feature; nothing is invented
 * here. The two verticals -- at the back of the fan case and at the two
 * trailing edges -- are what tie the cowl and the core together, and without
 * them the body has no connection to the core at all.
 *
 * Nothing touches the fan or the stator row: the upper boundary is the cowl
 * until the back of the fan case, so over the fan the body only exists
 * outboard of the nacelle.
 */
export function overMountPylon(engine, opts = {}) {
  const u = engine.userData;
  const R = u.rFan ?? (u.rMax ? u.rMax / 1.037 : 1);
  const g = new THREE.Group();

  const engineZ = opts.engineZ ?? 0;
  const attachY = (opts.attachY ?? P.attachY) * R;
  const topZ0 = (opts.topZ0 ?? P.topZ0) * R;
  const topZ1 = (opts.topZ1 ?? P.topZ1) * R;

  const cowl = u.cowlOuter ?? [[0.78 * R, 0.96 * R], [-2.35 * R, 1.03 * R]];
  const core = u.coreWall ?? [[-0.30 * R, 0.32 * R], [-2.90 * R, 0.40 * R]];
  const rootZ0 = (u.caseAft ?? -0.70 * R) + engineZ;
  const zAt = (arr, i) => arr[i][0] + engineZ;
  const interp = (arr, z) => {
    for (let i = 0; i < arr.length - 1; i++) {
      const a0 = zAt(arr, i), b0 = zAt(arr, i + 1);
      if (z <= a0 && z >= b0) {
        return arr[i][1] + (arr[i + 1][1] - arr[i][1])
          * ((a0 - z) / ((a0 - b0) || 1));
      }
    }
    return z > zAt(arr, 0) ? arr[0][1] : arr[arr.length - 1][1];
  };

  const pts = [];
  const V = (z, y) => pts.push(new THREE.Vector2(z, y));

  V(topZ0, -attachY);                                   // wing, forward
  V(zAt(cowl, 0), -cowl[0][1] * 0.995);                 // nacelle leading edge
  for (const [z, r] of cowl) {                          // aft along the cowl
    const zz = z + engineZ;
    if (zz < rootZ0) break;
    V(zz, -r * 0.995);
  }
  V(rootZ0, -interp(cowl, rootZ0) * 0.995);             // back of fan case, cowl
  V(rootZ0, -interp(core, rootZ0) * 0.995);             // ... and at the core
  for (const [z, r] of core) {                          // aft along the core
    const zz = z + engineZ;
    if (zz > rootZ0) continue;
    V(zz, -r * 0.995);                                  // just inside the skin
  }
  V(zAt(cowl, cowl.length - 1), -cowl[cowl.length - 1][1] * 0.995);  // cowl TE
  V(topZ1, -attachY);                                   // wing, aft

  // Extrude wants a counter-clockwise outline, and the sign of the enclosed
  // area flips as the engine slides past the wing -- so it is measured, not
  // assumed.
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const p0 = pts[i], p1 = pts[(i + 1) % pts.length];
    area += p0.x * p1.y - p1.x * p0.y;
  }
  if (area < 0) pts.reverse();

  const rounded = roundCorners(pts, P.fillet * R);

  const th = P.rootT * 2 * R;
  const geo = new THREE.ExtrudeGeometry(new THREE.Shape(rounded), {
    depth: th, bevelEnabled: false, curveSegments: 1,
  });
  geo.translate(0, 0, -th / 2);
  geo.computeVertexNormals();
  const body = new THREE.Mesh(geo, M.skin);
  body.rotation.y = -Math.PI / 2;            // shape X -> world Z, extrude -> X
  body.name = 'pylonStrut';
  g.add(body);

  g.userData.attachY = -attachY;
  g.userData.attachZ = [topZ0, topZ1];
  g.userData.rootZ = [rootZ0, zAt(core, core.length - 1)];
  g.userData.outline = rounded.length;
  g.userData.volumes = g.children.map((c) => volumeOf(c.geometry));
  g.name = 'overMountPylon';
  return g;
}

/**
 * Side-mount pylon: the engine hangs off the side of a fuselage.
 *
 * A horizontal stub rather than a vertical one, so the section is stacked in Y
 * and the planform lives in the XZ plane. Both edges come off the core case:
 *
 *   LEADING EDGE  leaves the front of the core case -- which is the back of
 *                 the fan case -- and runs STRAIGHT OUT, unswept.
 *   TRAILING EDGE leaves the rear of the core case and sweeps AFT, so the
 *                 aft attachment on the fuselage wall lands sideSweep nacelle
 *                 lengths behind the case. The section still closes to a thin
 *                 edge; it is the planform that grows.
 *
 * So the chord is longer at the wall than at the engine. `reach` is where the
 * fuselage is; on an installation both that and the sweep come from fuselage
 * geometry, not from here.
 *
 * The root is buried inside the core rather than started on its skin, for the
 * same reason as the other two: the load goes into the core's structure, and a
 * stub that begins at the surface reads as bolted to the fairing.
 */
export function sideMountPylon(engine, opts = {}) {
  const u = engine.userData;
  const R = u.rFan ?? (u.rMax ? u.rMax / 1.037 : 1);
  const g = new THREE.Group();

  const engineZ = opts.engineZ ?? 0;
  const side = opts.side ?? 1;
  const reach = (opts.reach ?? P.reach) * R;          // fuselage side
  const w = u.coreWall;
  const nacLen = u.cowlLength ?? 3.13 * R;
  const zLE = (u.caseAft ?? -0.70 * R) + engineZ;     // front of the core case
  const zTErt = (w ? w[w.length - 1][0] : -2.90 * R) + engineZ;

  // Start inside the core: half the core radius at mid-chord.
  const rc = w
    ? (() => {
        const zm = (zLE + zTErt) / 2 - engineZ;
        for (let i = 0; i < w.length - 1; i++) {
          if (zm <= w[i][0] && zm >= w[i + 1][0]) {
            return w[i][1] + (w[i + 1][1] - w[i][1])
              * ((w[i][0] - zm) / ((w[i][0] - w[i + 1][0]) || 1));
          }
        }
        return w[w.length - 1][1];
      })()
    : 0.35 * R;
  const xRoot = 0.5 * rc;

  const nS = P.sideStations, nP = P.nSide;
  const pos = [], idx = [], ring = [];
  for (let i = 0; i < nS; i++) {
    const t = i / (nS - 1);
    const x = side * (xRoot + (reach - xRoot) * t);
    // Trailing edge sweeping aft: at the wall it sits sideSweep nacelle
    // lengths behind the rear of the core case.
    const zTE = zTErt - P.sideSweep * nacLen * t;
    const c = zLE - zTE;
    const tMax = P.rootT * R * (1 - 0.55 * t);

    ring.push(pos.length / 3);
    for (let s2 = 0; s2 < 2; s2++) {
      for (let j = 0; j <= nP; j++) {
        if (s2 === 1 && (j === 0 || j === nP)) continue;
        const sv = s2 === 0 ? j / nP : 1 - j / nP;
        pos.push(x, (s2 === 0 ? 1 : -1) * halfT(sv, tMax), zLE - sv * c);
      }
    }
  }
  const perRing = 2 * (nP + 1) - 2;
  for (let i = 0; i < nS - 1; i++) {
    for (let j = 0; j < perRing; j++) {
      const a = ring[i] + j, b = ring[i] + ((j + 1) % perRing);
      const a2 = ring[i + 1] + j, b2 = ring[i + 1] + ((j + 1) % perRing);
      idx.push(a, b, a2, b, b2, a2);
    }
  }
  for (const [base, flip] of [[ring[0], true], [ring[nS - 1], false]]) {
    for (let j = 1; j < perRing - 1; j++) {
      if (flip) idx.push(base, base + j + 1, base + j);
      else idx.push(base, base + j, base + j + 1);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  faceOutward(geo);
  const strut = new THREE.Mesh(geo, M.skin);
  strut.name = 'pylonStrut';
  g.add(strut);

  g.userData.attachX = side * reach;
  g.userData.rootZ = [zLE, zTErt];
  g.userData.wallZ = [zLE, zTErt - P.sideSweep * nacLen];
  g.userData.volumes = [volumeOf(geo)];
  g.name = 'sideMountPylon';
  return g;
}

export { underMountPylon as pylon };
export default underMountPylon;
