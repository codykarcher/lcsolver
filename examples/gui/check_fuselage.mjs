/**
 * Headless checks on a lofted fuselage.
 *
 * Four things go wrong with a body built this way, and none of them are
 * reliably visible on screen:
 *
 *   1. the skin is not closed -- a seam or a missing cap, which shows only
 *      from one angle and only against a light background;
 *   2. faces are wound inward, so the body renders as a hollow shell. Note the
 *      test has to measure outward from the LOCAL SECTION CENTRE, not from the
 *      z axis: the tailcone centreline is a whole radius above the axis, so
 *      measuring from the axis condemns the entire upper tailcone as inverted
 *      when it is perfectly correct;
 *   3. windows and doors sink into the skin or float above it. They are placed
 *      by lifting off the surface, so this is really a check that the lift is
 *      bigger than the facet error of the skin and smaller than anything you
 *      would notice;
 *   4. the taper laws meet the barrel with a crease. A discontinuity in dr/dz
 *      is invisible in wireframe and obvious the moment there is a highlight.
 *
 *     node check_fuselage.mjs
 */
import * as THREE from 'three';
import { jetlinerFuselage } from './components/fuselage.js';

const CASES = [
  { name: '737-ish',   radius: 1.88, fineness: 10.1 },
  { name: 'stubby',    radius: 1.20, fineness: 7.0 },
  { name: 'widebody',  radius: 3.20, fineness: 13.0 },
];

let failures = 0;
const bad = (msg) => { console.log(`  FAIL  ${msg}`); failures++; };

for (const c of CASES) {
  const body = jetlinerFuselage({ radius: c.radius, length: c.fineness * 2 * c.radius });
  const u = body.userData;
  console.log(`${c.name}: L=${u.length.toFixed(2)} D=${(2 * u.radius).toFixed(2)} ` +
              `L/D=${u.fineness.toFixed(2)} parts=${body.children.length}`);

  const geo = body.children[0].geometry;
  const pos = geo.getAttribute('position'), nrm = geo.getAttribute('normal');
  const idx = geo.getIndex().array;

  /* 1. closed ------------------------------------------------------------ */
  // Weld by rounded position first: the seam at theta = 0 and the two apexes
  // are the same point reached by different indices, and an unwelded count
  // reports them as holes that are not there.
  const wi = new Int32Array(pos.count), w = new Map();
  for (let i = 0; i < pos.count; i++) {
    const k = [pos.getX(i), pos.getY(i), pos.getZ(i)]
      .map((v) => Math.round(v * 1e6) + 0).join(',');       // +0 kills -0
    if (!w.has(k)) w.set(k, w.size);
    wi[i] = w.get(k);
  }
  const edge = new Map();
  for (let t = 0; t + 2 < idx.length; t += 3) {
    const p = [wi[idx[t]], wi[idx[t + 1]], wi[idx[t + 2]]];
    if (p[0] === p[1] || p[1] === p[2] || p[0] === p[2]) continue;   // apex fans
    for (let k = 0; k < 3; k++) {
      const a = p[k], b = p[(k + 1) % 3], key = a < b ? `${a},${b}` : `${b},${a}`;
      edge.set(key, (edge.get(key) || 0) + 1);
    }
  }
  const open = [...edge.values()].filter((n) => n !== 2).length;
  if (open) bad(`skin has ${open} boundary edges -- not closed`);

  /* 2. wound outward ----------------------------------------------------- */
  let vol = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  const cc = new THREE.Vector3(), n = new THREE.Vector3();
  for (let t = 0; t + 2 < idx.length; t += 3) {
    a.fromBufferAttribute(pos, idx[t]);
    b.fromBufferAttribute(pos, idx[t + 1]);
    cc.fromBufferAttribute(pos, idx[t + 2]);
    vol += a.dot(n.crossVectors(b, cc)) / 6;
  }
  if (vol <= 0) bad(`enclosed volume ${vol.toFixed(3)} -- wound inward`);
  let inward = 0;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i) - u.shapeAt(pos.getZ(i)).yc;
    const r = Math.hypot(x, y);
    if (r < 1e-6) continue;                                  // on the centreline
    if ((nrm.getX(i) * x + nrm.getY(i) * y) / r < 0) inward++;
  }
  if (inward) bad(`${inward} vertex normals face inward`);

  /* 3. applied patches sit proud ---------------------------------------- */
  // Every window, door and pane is a lift off the surface. Measure each vertex
  // against the surface radius at its own station and angle.
  const lift = u.radius * 0.004;
  let sunk = 0, floating = 0, off = 0, patches = 0;
  const v = new THREE.Vector3();
  for (const child of body.children.slice(1)) {
    if (!child.isMesh || !child.geometry.getAttribute('position')) continue;
    if (child.geometry.type === 'CircleGeometry') continue;   // the APU disc
    patches++;
    const p = child.geometry.getAttribute('position');
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i);
      // Tolerance of one lift at each end: near a tip the outward normal
      // points along the axis, so lifting a patch off the skin there legally
      // carries it a hair past the nose. The radome does exactly this.
      if (v.z > lift || v.z < -u.length - lift) { off++; continue; }
      // Within a few lifts of the nose the surface is normal to the axis and
      // dr/dz is unbounded, so "distance out from the centreline at this z" is
      // not a measure of how proud anything is. Only the radome reaches here.
      if (v.z > -4 * lift) continue;
      const { r, yc } = u.shapeAt(v.z);
      const d = Math.hypot(v.x, v.y - yc) - r;
      if (d < lift * 0.15) sunk++;
      if (d > lift * 3.0) floating++;
    }
  }
  if (sunk) bad(`${sunk} patch vertices at or below the skin`);
  if (floating) bad(`${floating} patch vertices more than 3x the lift proud`);
  if (off) bad(`${off} patch vertices outside the body's z range`);

  /* 4. no crease at the joins ------------------------------------------- */
  for (const [what, z0] of [['nose/barrel', u.cabinZ[0]], ['barrel/tail', u.cabinZ[1]]]) {
    const h = 1e-3;
    const slope = (u.shapeAt(z0 - h).r - u.shapeAt(z0 + h).r) / (2 * h);
    if (Math.abs(slope) > 1e-3) bad(`${what} join has dr/dz = ${slope.toFixed(5)}`);
  }

  console.log(`  ${patches} applied patches, volume ${vol.toFixed(2)}, ` +
              `${open} open edges, ${inward} inward normals`);
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nPASS: bodies are closed, outward and clean at the joins');
process.exit(failures ? 1 : 0);
