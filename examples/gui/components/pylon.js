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
  plateT:    0.075, // thickness of the flat attachment plate
  plateX:    0.34,  // ... and its half-width
  noseA:     0.010, // section nose bluntness
  teFrac:    0.45,  // trailing-edge thickness, fraction of maximum
  kink:      1.02,  // leading-edge kink height, x nacelle max radius
  below:     5,     // stations below the kink
  above:     6,     // ... and above it
  nSide:     22,    // section points per side
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
  const strut = new THREE.Mesh(geo, M.structure);
  strut.name = 'pylonStrut';
  g.add(strut);

  const plate = new THREE.Mesh(
    new THREE.BoxGeometry(P.plateX * 2 * R, P.plateT * R, topZ0 - topZ1),
    M.structure);
  plate.position.set(0, attachY + P.plateT * R * 0.5, (topZ0 + topZ1) / 2);
  plate.name = 'pylonPlate';
  g.add(plate);

  g.userData.attachY = attachY + P.plateT * R;
  g.userData.attachZ = [topZ0, topZ1];
  g.userData.rootZ = [rootZ0, rootZ1];
  g.userData.kinkY = yKink;
  g.name = 'underMountPylon';
  return g;
}

/**
 * Over-mount pylon: the engine sits above the wing and the strut hangs down.
 *
 * Geometrically the under-mount inverted, with two differences that are not
 * cosmetic.
 *
 * The forward fairing runs all the way to the NACELLE'S LEADING EDGE, along
 * the outside of the cowl. On an over-wing installation that fairing is what
 * you see; it is the reason the pylon reads as part of the nacelle rather than
 * as a post under it.
 *
 * And nothing may cut the fan or the stator row. The strut therefore keeps its
 * root aft of the fan case, exactly as the under-mount does, and the forward
 * reach to the nacelle nose is carried by a fin that sits ON the cowl instead
 * of by the strut passing through it. Extending the strut itself forward would
 * take it straight through the fan.
 */
export function overMountPylon(engine, opts = {}) {
  const u = engine.userData;
  const R = u.rFan ?? (u.rMax ? u.rMax / 1.037 : 1);
  const g = new THREE.Group();

  // The strut is the under-mount, mirrored. Building it that way rather than
  // writing a second loft keeps the two in step: a change to the section or
  // the kink shows up in both.
  const under = underMountPylon(engine, opts);
  const strut = under.getObjectByName('pylonStrut');
  under.remove(strut);
  strut.scale.y = -1;
  strut.name = 'pylonStrut';
  g.add(strut);

  const plate = under.getObjectByName('pylonPlate');
  under.remove(plate);
  plate.position.y = -plate.position.y;
  g.add(plate);

  // ---- forward fairing on the cowl ---------------------------------------
  const cowl = u.cowlOuter;
  const engineZ = opts.engineZ ?? 0;
  const rootZ0 = under.userData.rootZ[0];
  const yKink = under.userData.kinkY;

  if (cowl) {
    const zNose = cowl[0][0] + engineZ;
    const finT = P.rootT * R * 0.55;
    const pos = [], idx = [], ring = [];
    const N = 22;

    for (let i = 0; i <= N; i++) {
      const t = i / N;
      const z = zNose + (rootZ0 - zNose) * t;
      // Nacelle skin at this station, from the engine's own cowl line.
      let rN = cowl[0][1];
      for (let k = 0; k < cowl.length - 1; k++) {
        const [z0, r0] = cowl[k], [z1, r1] = cowl[k + 1];
        const a = z0 + engineZ, b = z1 + engineZ;
        if (z <= a && z >= b) { rN = r0 + (r1 - r0) * ((a - z) / ((a - b) || 1)); break; }
      }
      const yIn = -rN * 0.985;                       // just inside the skin
      const yOut = -(rN + (yKink - rN) * t * t);     // sweeping down to the kink
      const th = finT * Math.pow(t, 0.6);            // grows from a point
      ring.push(pos.length / 3);
      pos.push(th, yIn, z,  th, yOut, z,  -th, yOut, z,  -th, yIn, z);
    }
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < 4; j++) {
        const a = ring[i] + j, b = ring[i] + ((j + 1) % 4);
        const a2 = ring[i + 1] + j, b2 = ring[i + 1] + ((j + 1) % 4);
        idx.push(a, b, a2, b, b2, a2);
      }
    }
    idx.push(ring[0], ring[0] + 2, ring[0] + 1, ring[0], ring[0] + 3, ring[0] + 2);
    const e = ring[N];
    idx.push(e, e + 1, e + 2, e, e + 2, e + 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    const fin = new THREE.Mesh(geo, M.structure);
    fin.name = 'pylonFairing';
    g.add(fin);
  }

  g.userData.attachY = -under.userData.attachY;
  g.userData.attachZ = under.userData.attachZ;
  g.userData.rootZ = under.userData.rootZ;
  g.userData.kinkY = -yKink;
  g.name = 'overMountPylon';
  return g;
}

export { underMountPylon as pylon };
export default underMountPylon;
