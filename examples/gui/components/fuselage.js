/**
 * Fuselages.
 *
 * A fuselage here is a **lofted body**: a stack of closed sections swept along
 * a curved centreline. Three functions define one completely --
 *
 *   r(z)     the size of the section at station z
 *   yc(z)    where the section's centre sits (droop at the nose, upsweep aft)
 *   sec(th)  the SHAPE of the section, as a radius multiplier by angle
 *
 * -- and everything else in this file is either one of those three for a
 * particular aeroplane, or machinery for turning them into triangles.
 *
 * Splitting the section shape out from the size distribution is the whole
 * point. A conventional jetliner and a double bubble have very nearly the same
 * r(z) and yc(z); what makes one a D8 is `sec`. So the second fuselage is a
 * section function, not a second body, and anything that works on one -- the
 * window rows, the doors, the surface-patch machinery, whatever attaches to it
 * later -- works on the other without being told about it.
 *
 * Axis convention: nose tip at the ORIGIN, body running aft along **-Z**, +Y
 * up. Same as every other component in the library, so a fuselage and an engine
 * dropped into the same scene are already in the same frame.
 */
import * as THREE from 'three';
import { skin, glass, trim, cavity, painted } from './materials.js';
import { orientOutward } from './geom.js';

/* ---- section shapes ---------------------------------------------------- */

/**
 * A circle. What almost every pressurised metal tube actually is -- a cylinder
 * is the only shape that carries pressure in pure tension.
 */
export const circularSection = () => 1;

/* ---- shape law --------------------------------------------------------- */

/**
 * Proportions of a conventional single-aisle jetliner, in diameters and radii
 * so they hold at any size. Numbers are off a 737-800 / A320: 37-38 m long,
 * 3.8-4.0 m across, so a fineness ratio right around 10.
 *
 * The two taper laws are both `(1 - x^A)^B`. That family is worth the two
 * exponents: B < 1 makes the tip round rather than pointed, A > 1 makes the
 * slope vanish where the taper meets the barrel. So the nose and the tailcone
 * each blend into the constant section with no crease, which is the thing you
 * notice immediately if it is missing.
 */
const JET = {
  fineness:  10.1,   // overall length / diameter
  noseD:      1.70,  // nose length, in diameters
  tailD:      2.90,  // tailcone length, in diameters
  noseA: 2.2, noseB: 0.55,
  tailA: 1.6, tailB: 0.75,
  tipR:       0.10,  // tailcone tip radius, in radii -- the APU exhaust
  droop:      0.10,  // nose centreline drop, in radii
  upsweep:    1.05,  // tailcone centreline rise, in radii
};

/**
 * Build the r(z) and yc(z) pair for a three-part body: taper, barrel, taper.
 *
 * Returned as one object carrying `at(z)` and the stations it was cut at, so
 * that everything downstream -- skin, windows, doors, and later whatever mounts
 * to the side of it -- asks the same question of the same object and cannot
 * disagree about where the surface is.
 */
function jetShape({ length, radius, p = JET }) {
  const lNose = p.noseD * 2 * radius;
  const lTail = p.tailD * 2 * radius;
  const zNose = -lNose;                    // nose taper ends here
  const zTail = -(length - lTail);         // tailcone begins here
  const rTip  = p.tipR * radius;

  function at(z) {
    if (z > zNose) {                                     // nose
      const t = Math.min(1, Math.max(0, -z / lNose));
      return {
        r:  radius * Math.pow(1 - Math.pow(1 - t, p.noseA), p.noseB),
        yc: -p.droop * radius * Math.pow(1 - t, 2),
      };
    }
    if (z > zTail) return { r: radius, yc: 0 };          // barrel
    const s = Math.min(1, Math.max(0, (zTail - z) / lTail));   // tailcone
    return {
      r:  rTip + (radius - rTip) * Math.pow(1 - Math.pow(s, p.tailA), p.tailB),
      yc: p.upsweep * radius * Math.pow(s, 1.8),
    };
  }

  return { at, length, radius, lNose, lTail, zNose, zTail, rTip };
}

/* ---- lofting ----------------------------------------------------------- */

/** The surface point at station z, angle th. th = 0 is +X, th = pi/2 is up. */
function surfacePoint(shape, sec, z, th, out = new THREE.Vector3()) {
  const { r, yc } = shape.at(z);
  const rr = r * sec(th);
  return out.set(rr * Math.cos(th), yc + rr * Math.sin(th), z);
}

/**
 * Skin the body: a grid of quads over (station, angle), closed at both ends.
 *
 * Stations are placed on a cosine spacing rather than evenly. The body is
 * nearly straight down the middle and turns hard at both tips, so uniform
 * spacing spends most of its triangles where nothing is happening and then
 * facets the nose. Cosine clustering puts them where the curvature is.
 */
function skinBody(shape, sec, { nStation = 140, nSeg = 64 } = {}) {
  const pos = [], idx = [];
  const v = new THREE.Vector3();
  const L = shape.length;

  const zs = [];
  for (let i = 0; i < nStation; i++) {
    zs.push(-L * (1 - Math.cos((i / (nStation - 1)) * Math.PI)) / 2);
  }

  for (const z of zs) {
    for (let j = 0; j < nSeg; j++) {
      surfacePoint(shape, sec, z, (j / nSeg) * Math.PI * 2, v);
      pos.push(v.x, v.y, v.z);
    }
  }
  for (let i = 0; i < nStation - 1; i++) {
    for (let j = 0; j < nSeg; j++) {
      const a = i * nSeg + j, b = i * nSeg + ((j + 1) % nSeg);
      idx.push(a, b, a + nSeg, b, b + nSeg, a + nSeg);
    }
  }

  // Close both ends onto a centre vertex. At the nose the ring has collapsed to
  // a point already and these triangles are near-degenerate, which is harmless;
  // at the tail the ring is the APU exhaust and the cap is real.
  for (const [ring, z] of [[0, zs[0]], [nStation - 1, zs[nStation - 1]]]) {
    const c = pos.length / 3;
    pos.push(0, shape.at(z).yc, z);
    for (let j = 0; j < nSeg; j++) {
      idx.push(c, ring * nSeg + j, ring * nSeg + ((j + 1) % nSeg));
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return orientOutward(g);
}

/**
 * A patch of the skin, lifted clear of it -- one window, one door, one pane.
 *
 * Everything applied to a fuselage is applied to a curved, varying surface, so
 * building these as flat quads placed near the body means picking between
 * floating above it and sinking into it, differently at every station. Sampling
 * the same `shape` the skin was built from and pushing out along the local
 * normal instead means a window is by construction the right shape and sits the
 * same distance proud everywhere, on any body, without a single special case.
 */
function patch(shape, sec, z0, z1, th0, th1, lift, material, nz = 3, nt = 5) {
  const pos = [], idx = [];
  const v = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3();

  for (let i = 0; i < nz; i++) {
    const z = z0 + (z1 - z0) * (i / (nz - 1));
    for (let j = 0; j < nt; j++) {
      const th = th0 + (th1 - th0) * (j / (nt - 1));
      // Local outward normal from two finite differences on the surface, so it
      // is correct for a non-circular section too, where radial is not normal.
      const d = 1e-3;
      surfacePoint(shape, sec, z, th, v);
      a.copy(surfacePoint(shape, sec, z, th + d, new THREE.Vector3())).sub(v);
      b.copy(surfacePoint(shape, sec, z + d, th, new THREE.Vector3())).sub(v);
      const n = new THREE.Vector3().crossVectors(a, b).normalize();
      if (n.dot(new THREE.Vector3(v.x, v.y - shape.at(z).yc, 0)) < 0) n.negate();
      pos.push(v.x + n.x * lift, v.y + n.y * lift, v.z + n.z * lift);
    }
  }
  for (let i = 0; i < nz - 1; i++) {
    for (let j = 0; j < nt - 1; j++) {
      const p = i * nt + j;
      idx.push(p, p + 1, p + nt, p + 1, p + nt + 1, p + nt);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();

  const m = new THREE.Mesh(g, material);
  // A patch is an open sheet, so signed volume says nothing about it; check it
  // the direct way instead, against the normal we already know is outward.
  const nrm = g.getAttribute('normal');
  const p0 = new THREE.Vector3().fromBufferAttribute(g.getAttribute('position'), 0);
  const n0 = new THREE.Vector3().fromBufferAttribute(nrm, 0);
  if (n0.dot(p0.setZ(0)) < 0) {
    const arr = g.getIndex().array;
    for (let t = 0; t + 2 < arr.length; t += 3) {
      const s = arr[t + 1]; arr[t + 1] = arr[t + 2]; arr[t + 2] = s;
    }
    g.getIndex().needsUpdate = true;
    g.computeVertexNormals();
  }
  return m;
}

/** Both sides at once. Mirroring in theta is exact; mirroring meshes is not. */
function pair(shape, sec, z0, z1, th0, th1, lift, material, nz, nt) {
  return [
    patch(shape, sec, z0, z1, th0, th1, lift, material, nz, nt),
    patch(shape, sec, z0, z1, Math.PI - th0, Math.PI - th1, lift, material, nz, nt),
  ];
}

/* ---- the aeroplane ----------------------------------------------------- */

const DEG = Math.PI / 180;

/**
 * A conventional jetliner fuselage.
 *
 * @param {number} radius     body radius. Default 1.88 -- a 737's 3.76 m tube.
 * @param {number} length     nose to tailcone tip. Default sets fineness 10.1.
 * @param {Function} section  section shape, th -> radius multiplier.
 * @param {boolean} windows   cabin window row and flight-deck glass.
 * @param {boolean} doors     main doors and overwing exits.
 */
export function jetlinerFuselage({
  radius = 1.88,
  length = null,
  section = circularSection,
  windows = true,
  doors = true,
  nSeg = 64,
} = {}) {
  const L = length ?? JET.fineness * 2 * radius;
  const shape = jetShape({ length: L, radius });
  const g = new THREE.Group();
  const lift = radius * 0.004;

  g.add(new THREE.Mesh(skinBody(shape, section, { nSeg }), skin));

  // The radome is a different material to the skin on a real aeroplane -- it
  // has to be transparent to the radar behind it -- and is almost always left
  // unpainted or in a contrasting grey. It is also the one part of the nose
  // whose shape is set by the antenna rather than by the aerodynamics.
  const rad = patch(shape, section, -0.001, -shape.lNose * 0.22,
                    0, Math.PI * 2, lift * 0.5, painted, 5, 33);
  g.add(rad);

  if (windows) {
    // Flight deck. Two panes a side: a windscreen over the front quarter and a
    // side window aft of it, offset down and back the way a real one is.
    const n = shape.lNose;
    for (const m of pair(shape, section, -0.30 * n, -0.60 * n,
                         78 * DEG, 40 * DEG, lift, glass, 4, 5)) g.add(m);
    for (const m of pair(shape, section, -0.47 * n, -0.74 * n,
                         34 * DEG, 11 * DEG, lift, glass, 3, 4)) g.add(m);

    // Cabin windows, on a 0.51 m (20 in) pitch -- the frame spacing, which is
    // what actually sets it -- from just aft of the flight deck to well into
    // the tailcone taper.
    const pitch = 0.51, wz = 0.115, wth = 0.165 / 2;
    const zFwd = -shape.lNose * 1.02;
    const zAft = shape.zTail - shape.lTail * 0.34;
    const seat = 20 * DEG;                       // above the horizontal
    for (let z = zFwd; z > zAft; z -= pitch) {
      for (const m of pair(shape, section, z + wz, z - wz,
                           seat - wth, seat + wth, lift, glass, 2, 3)) g.add(m);
    }
  }

  if (doors) {
    // Two plug doors and two overwing exits a side. Doors reach from the cabin
    // floor up, so they straddle the window line rather than sitting on it.
    const dth = 0.86 / radius / 2, dz = 1.83 / 2;
    const mid = 6 * DEG;
    for (const z of [-shape.lNose * 1.14, shape.zTail - shape.lTail * 0.16]) {
      for (const m of pair(shape, section, z + dz, z - dz,
                           mid - dth, mid + dth, lift * 0.7, trim, 3, 4)) g.add(m);
    }
    const ez = 1.10 / 2, eth = 0.51 / radius / 2;
    for (const z of [-L * 0.50, -L * 0.57]) {
      for (const m of pair(shape, section, z + ez, z - ez,
                           14 * DEG - eth, 14 * DEG + eth, lift * 0.7, trim, 2, 3)) g.add(m);
    }
  }

  // The APU exhaust: the tailcone does not close to a point, it closes onto a
  // hole, and a hole has to read as one.
  const apu = new THREE.Mesh(
    new THREE.CircleGeometry(shape.rTip * 0.86, 24), cavity);
  apu.position.set(0, shape.at(-L).yc, -L + lift);
  apu.rotation.y = Math.PI;
  g.add(apu);

  Object.assign(g.userData, {
    length: L, radius, section,
    noseLength: shape.lNose, tailLength: shape.lTail,
    cabinZ: [shape.zNose, shape.zTail],
    fineness: L / (2 * radius),
    /** Surface geometry at a station -- what anything mounting to this asks. */
    shapeAt: (z) => shape.at(z),
    surfaceAt: (z, th) => surfacePoint(shape, section, z, th),
  });
  return g;
}

export const fuselages = { jetliner: jetlinerFuselage };
