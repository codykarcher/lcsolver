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
 *   root      sits on the core, aft of the fan, where the load actually goes
 *   strut     a streamwise section lofted upward, PIERCING the nacelle
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
  rootZ0:   -0.55,  // root chord on the core, forward end
  rootZ1:   -2.15,  // ... and aft
  rootT:     0.085, // root half-thickness
  topZ0:     0.45,  // top chord at the wing, forward end
  topZ1:    -2.35,  // ... and aft
  topT:      0.130, // top half-thickness
  attachY:   1.95,  // wing underside, above the engine axis
  plateT:    0.075, // thickness of the flat attachment plate
  plateX:    0.34,  // ... and its half-width
  noseA:     0.010, // section nose bluntness
  teFrac:    0.45,  // trailing-edge thickness, fraction of maximum
  stations:  9,     // loft stations from root to top
  nSide:     22,    // section points per side
};

/** Half-thickness of the streamwise section at chord fraction `s`. */
function halfT(s, tMax) {
  return tMax * Math.sqrt(s / (s + P.noseA)) * (1 - (1 - P.teFrac) * s * s);
}

/** Read the engine's core radius at an axial station. */
function coreRadiusAt(engine, z) {
  const w = engine.userData.coreWall;
  if (!w) return 0;
  for (let i = 0; i < w.length - 1; i++) {
    const [z0, r0] = w[i], [z1, r1] = w[i + 1];
    if (z <= z0 && z >= z1) {
      return r0 + (r1 - r0) * ((z0 - z) / ((z0 - z1) || 1));
    }
  }
  return z > w[0][0] ? w[0][1] : w[w.length - 1][1];
}

/**
 * @param {THREE.Object3D} engine  a podded or bare turbofan
 * @param {object} [opts]
 * @param {number} [opts.attachY]  wing underside height, engine radii
 * @param {number} [opts.topZ0]    forward end of the wing attachment
 * @param {number} [opts.topZ1]    aft end
 */
export function pylon(engine, opts = {}) {
  const R = engine.userData.rFan
    ?? (engine.userData.rMax ? engine.userData.rMax / 1.037 : 1);
  const g = new THREE.Group();

  const attachY = (opts.attachY ?? P.attachY) * R;
  const topZ0 = (opts.topZ0 ?? P.topZ0) * R;
  const topZ1 = (opts.topZ1 ?? P.topZ1) * R;

  // Root sits ON the core: take the core radius under the middle of the root
  // chord so the strut starts at the skin rather than in mid air.
  const rootZ0 = P.rootZ0 * R, rootZ1 = P.rootZ1 * R;
  const rootY = coreRadiusAt(engine, (rootZ0 + rootZ1) / 2) * 0.92;

  // ---- loft ---------------------------------------------------------------
  const nS = P.stations, nP = P.nSide;
  const pos = [], idx = [];
  const ring = [];                       // vertex index of each station's ring

  for (let i = 0; i < nS; i++) {
    const t = i / (nS - 1);
    const e = t * t * (3 - 2 * t);       // smoothstep, so the loft eases in
    const y = rootY + (attachY - rootY) * t;
    const zLE = rootZ0 + (topZ0 - rootZ0) * e;
    const zTE = rootZ1 + (topZ1 - rootZ1) * e;
    const c = zLE - zTE;
    const tMax = (P.rootT + (P.topT - P.rootT) * e) * R;

    const base = pos.length / 3;
    ring.push(base);
    // Down the +X side from the leading edge, then back up the -X side. The
    // segment across the aft end is the blunt trailing edge.
    for (let side = 0; side < 2; side++) {
      for (let j = 0; j <= nP; j++) {
        if (side === 1 && (j === 0 || j === nP)) continue;   // shared points
        const s = side === 0 ? j / nP : 1 - j / nP;
        const x = (side === 0 ? 1 : -1) * halfT(s, tMax);
        pos.push(x, y, zLE - s * c);
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
  // Cap both ends with fans so the strut is a closed solid.
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

  // ---- flat attachment plate ---------------------------------------------
  const plate = new THREE.Mesh(
    new THREE.BoxGeometry(P.plateX * 2 * R, P.plateT * R, topZ0 - topZ1),
    M.structure);
  plate.position.set(0, attachY + P.plateT * R * 0.5, (topZ0 + topZ1) / 2);
  plate.name = 'pylonPlate';
  g.add(plate);

  g.userData.attachY = attachY + P.plateT * R;
  g.userData.attachZ = [topZ0, topZ1];
  g.userData.rootY = rootY;
  g.name = 'pylon';
  return g;
}

export default pylon;
