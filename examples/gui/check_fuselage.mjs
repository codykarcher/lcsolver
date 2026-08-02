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
  // detail: true on purpose. The default body is bare OML now, and a check that
  // built the default would silently stop testing the decal machinery -- which
  // is precisely where the degenerate triangles and the overlaps came from.
  const body = jetlinerFuselage({
    radius: c.radius, length: c.fineness * 2 * c.radius, detail: true });
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
  // Every window, door and pane is placed by lifting off the surface along the
  // local normal, so the standoff should be exactly the lift, everywhere.
  //
  // It has to be measured as a true PERPENDICULAR distance. Comparing "how far
  // out from the centreline" against "the body radius at this station" is only
  // the same thing where the surface is parallel to the axis; on a surface of
  // slope m it over-reads by sqrt(1 + m^2), because the lift moved the vertex
  // in z as well as outward. That is harmless on the barrel and wrong by a
  // factor of four on a blunt nose -- which is exactly what this check reported
  // the first time the nose was given a real tip radius.
  //
  // So: minimise the distance from the vertex to the section circle over
  // station, coarse then fine. A handful of thousand vertices, once.
  const lift = u.radius * 0.004;
  const toSurface = (v) => {
    const gap = (z) => {
      const { r, yc } = u.shapeAt(z);
      return Math.hypot(z - v.z, Math.hypot(v.x, v.y - yc) - r);
    };
    let best = Infinity, bz = v.z;
    const span = u.radius * 2;
    for (let k = -40; k <= 40; k++) {
      const z = Math.min(0, Math.max(-u.length, v.z + span * k / 40));
      const g = gap(z);
      if (g < best) { best = g; bz = z; }
    }
    for (let step = span / 40; step > 1e-5; step /= 4) {
      for (let k = -4; k <= 4; k++) {
        const z = Math.min(0, Math.max(-u.length, bz + step * k / 4));
        const g = gap(z);
        if (g < best) { best = g; bz = z; }
      }
    }
    return best;
  };

  let sunk = 0, floating = 0, off = 0, patches = 0, worstStand = 0;
  const v = new THREE.Vector3();
  for (const child of body.children.slice(1)) {
    if (!child.isMesh || !child.geometry.getAttribute('position')) continue;
    if (child.geometry.type === 'CircleGeometry') continue;   // the APU disc
    patches++;
    const p = child.geometry.getAttribute('position');
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i);
      // Tolerance of one lift at each end: at a tip the outward normal points
      // along the axis, so lifting a patch off the skin there legally carries
      // it a hair past the nose. The radome does exactly this.
      if (v.z > lift || v.z < -u.length - lift) { off++; continue; }
      const d = toSurface(v);
      worstStand = Math.max(worstStand, d);
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

  /* 5. the crown is level aft ------------------------------------------- */
  // The tailcone taper is supposed to come entirely off the belly. If any of it
  // leaks into the centreline the roof develops a hump, which is hard to see
  // in a three-quarter view and glaring in profile.
  let crownDrift = 0, keelRise = 0;
  for (let i = 0; i <= 60; i++) {
    const z = u.cabinZ[1] - (u.length + u.cabinZ[1]) * (i / 60);
    crownDrift = Math.max(crownDrift, Math.abs(u.crownAt(z) - u.radius));
    keelRise = Math.max(keelRise, u.keelAt(z) - u.keelAt(u.cabinZ[1]));
  }
  if (crownDrift > 1e-9) bad(`crown moves ${crownDrift.toFixed(4)} over the tailcone`);
  if (keelRise < u.radius) bad(`belly only rises ${keelRise.toFixed(2)} -- taper is not on the underside`);

  /* 5b. the nose is the tailcone, mirrored ------------------------------- */
  // Same three claims as aft, with crown and keel swapped: the keel holds its
  // line, the crown spends the taper, and neither profile bulges past the
  // barrel on the way. The last one is what "the nose hangs below the tube"
  // was, stated as a measurement.
  let keelDrift = 0, crownFall = 0, hang = 0, bulge = 0;
  let prevC = null, prevK = null;
  for (let i = 0; i <= 80; i++) {
    const z = -u.noseLength * (i / 80);          // TIP (i=0) -> barrel join
    keelDrift = Math.max(keelDrift, Math.abs(u.keelAt(z) - u.keelAt(u.cabinZ[0])));
    crownFall = Math.max(crownFall, u.crownAt(u.cabinZ[0]) - u.crownAt(z));
    if (u.keelAt(z) < -u.radius - 1e-6) hang++;
    if (u.crownAt(z) > u.radius + 1e-6) hang++;
    // Both lines must open out monotonically going aft from the point. A bulge
    // is a profile that turns back on itself on the way, which no amount of
    // staring at a shaded three-quarter view reliably catches.
    if (prevC !== null && (u.crownAt(z) < prevC - 1e-9 || u.keelAt(z) > prevK + 1e-9)) bulge++;
    prevC = u.crownAt(z); prevK = u.keelAt(z);
  }
  const keelBudget = (1 - u.keelHold) * u.radius + 1e-6;
  if (hang) bad(`nose profile leaves the barrel envelope at ${hang} stations`);
  if (bulge) bad(`nose profile is not monotonic at ${bulge} stations`);
  if (keelDrift > keelBudget)
    bad(`keel moves ${keelDrift.toFixed(3)}, budget ${keelBudget.toFixed(3)} at keelHold ${u.keelHold}`);
  if (crownFall < u.radius) bad(`crown only falls ${crownFall.toFixed(2)} -- taper is not on the crown`);
  const joinYc = Math.abs(u.shapeAt(u.cabinZ[0]).yc);
  if (joinYc > 1e-6) bad(`centreline is ${joinYc.toFixed(4)} off axis at the nose join`);

  /* 5c. the point is a sphere of the radius asked for -------------------- */
  // The nose is specified by its tip radius of curvature, so measure it -- and
  // measure it on BOTH meridians, because the failure this replaced was a tip
  // that had the right curvature over the crown and a crease under the keel.
  // For a circle tangent to the axis, a lateral offset d from the tip sits at
  // an axial depth d^2/(2 rho); invert that at a small d.
  const tipRho = (which) => {
    const yTip = u.shapeAt(0).yc, d = 0.016 * u.radius;
    let lo = 0, hi = -u.noseLength;
    for (let i = 0; i < 60; i++) {
      const mid = (lo + hi) / 2;
      if (Math.abs(u[which](mid) - yTip) < d) lo = mid; else hi = mid;
    }
    return d * d / (2 * Math.abs(lo));
  };
  const rhoC = tipRho('crownAt'), rhoK = tipRho('keelAt');
  for (const [what, got] of [['crown', rhoC], ['keel', rhoK]]) {
    if (Math.abs(got / u.noseRadius - 1) > 0.08)
      bad(`${what} tip radius ${got.toFixed(3)} vs ${u.noseRadius.toFixed(3)} asked`);
  }
  if (Math.abs(rhoK / rhoC - 1) > 0.08)
    bad(`tip is not round: keel/crown curvature ratio ${(rhoK / rhoC).toFixed(3)}`);

  /* 6. no degenerate triangles ------------------------------------------ */
  // Slivers of near-zero area are what a collapsed grid row leaves behind, and
  // they are the classic source of shards flickering over a surface. The skin's
  // two apex fans are the one legitimate exception and are excluded above.
  let slivers = 0, worstPart = null;
  for (const child of body.children) {
    if (!child.isMesh || !child.geometry.getIndex()) continue;
    const p = child.geometry.getAttribute('position');
    const ix = child.geometry.getIndex().array;
    const scale = new THREE.Box3().setFromBufferAttribute(p).getSize(new THREE.Vector3()).length();
    let here = 0;
    for (let t = 0; t + 2 < ix.length; t += 3) {
      a.fromBufferAttribute(p, ix[t]);
      b.fromBufferAttribute(p, ix[t + 1]);
      cc.fromBufferAttribute(p, ix[t + 2]);
      const area = n.crossVectors(b.sub(a), cc.sub(a)).length() / 2;
      if (area < 1e-6 * scale * scale) here++;
    }
    if (child !== body.children[0] && here) { slivers += here; worstPart = child; }
  }
  if (slivers) bad(`${slivers} degenerate triangles on applied parts ` +
                   `(first at z=${worstPart.geometry.getAttribute('position').getZ(0).toFixed(2)})`);

  /* 7. no two decals over the same skin ---------------------------------- */
  // Two decals on the same patch of skin are two surfaces a few millimetres
  // apart competing for the depth buffer, which reads as flicker rather than as
  // either of them. This caught the overwing exits sitting on top of the window
  // row, which shared both a station and an angle band.
  const boxes = u.decals.map((m) => {
    const p = m.geometry.getAttribute('position');
    const bb = { z0: -1e9, z1: 1e9, t0: 1e9, t1: -1e9 };
    for (let i = 0; i < p.count; i++) {
      const z = p.getZ(i), yc = u.shapeAt(z).yc;
      const th = Math.atan2(p.getY(i) - yc, p.getX(i));
      bb.z0 = Math.max(bb.z0, z); bb.z1 = Math.min(bb.z1, z);
      bb.t0 = Math.min(bb.t0, th); bb.t1 = Math.max(bb.t1, th);
    }
    return bb;
  });
  let overlaps = 0;
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const A = boxes[i], B = boxes[j];
      if (A.t1 - A.t0 > 3 || B.t1 - B.t0 > 3) continue;   // the radome, all round
      if (A.z1 < B.z0 && B.z1 < A.z0 && A.t0 < B.t1 && B.t0 < A.t1) overlaps++;
    }
  }
  if (overlaps) bad(`${overlaps} pairs of decals overlap on the skin`);

  console.log(`  standoff: lift ${lift.toFixed(4)}, worst ${worstStand.toFixed(4)}`);
  console.log(`  ${patches} applied patches, volume ${vol.toFixed(2)}, ` +
              `${open} open edges, ${inward} inward normals, ` +
              `${slivers} slivers, ${overlaps} overlaps`);
  console.log(`  crown level to ${crownDrift.toExponential(1)}, ` +
              `belly rises ${keelRise.toFixed(2)} of ${u.radius.toFixed(2)}, ` +
              `nose: keel moves ${keelDrift.toFixed(3)}/${keelBudget.toFixed(3)}, ` +
              `crown falls ${crownFall.toFixed(2)}, tip y ${u.shapeAt(0).yc.toFixed(2)}`);
  console.log(`  tip radius asked ${u.noseRadius.toFixed(3)}, ` +
              `crown ${rhoC.toFixed(3)}, keel ${rhoK.toFixed(3)} ` +
              `(round to ${(100 * Math.abs(rhoK / rhoC - 1)).toFixed(1)}%)`);
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nPASS: bodies are closed, outward and clean at the joins');
process.exit(failures ? 1 : 0);
