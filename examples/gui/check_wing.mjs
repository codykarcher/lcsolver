/**
 * Headless checks on a lifting surface.
 *
 *     node check_wing.mjs
 *
 * A wing is parameterised by numbers that have textbook definitions, so unlike
 * the fuselage most of this can be checked against a CLOSED FORM rather than
 * against my own arithmetic restated. For an untwisted trapezoid,
 *
 *     MAC   = (2/3) c_root (1 + L + L^2) / (1 + L)
 *     y_MAC = (b/6) (1 + 2L) / (1 + L)
 *
 * and those are independent of everything in wing.js. If the built surface
 * agrees with them, the sweep, dihedral and station machinery are all consistent
 * with the planform they claim to be.
 */
import * as THREE from 'three';
import { wing, horizontalTail, liftingSurface } from './components/wing.js';
import { naca4, thicknessOf } from './components/airfoil.js';

const DEG = Math.PI / 180;
let failures = 0;
const bad = (msg) => { console.log(`  FAIL  ${msg}`); failures++; };

/* ---- sections ---------------------------------------------------------- */
{
  console.log('aerofoils:');
  for (const code of ['0010', '2412', '4415']) {
    const f = naca4(code);
    const t = thicknessOf(f);
    const want = parseInt(String(code).slice(2), 10) / 100;
    if (Math.abs(t - want) > 5e-4)
      bad(`NACA ${code} is ${t.toFixed(4)} thick, should be ${want}`);
    // Closed loop: the two surfaces must MEET at the trailing edge, or the wing
    // has a slot down its back. The published polynomial does not close -- its
    // last coefficient leaves 0.21% of chord -- so this is checking the
    // substitution, not the definition.
    const teUpper = f[0], teLower = f[f.length - 1];
    if (Math.abs(teUpper[0] - 1) > 1e-9)
      bad(`NACA ${code} does not start at the trailing edge`);
    const gap = Math.abs(naca4(code, 400)[0][1] - naca4(code, 400).at(-1)[1]);
    // Loop order: TE, up over the top, round the nose, back along the bottom.
    // Near index n-1, not exactly at it. On a cambered section the mean line at
    // x = 0 is not the furthest-forward point: the upper surface just aft of it
    // wraps round to slightly NEGATIVE x, which is real geometry and is why the
    // leading edge of a cambered aerofoil is not on its chord line.
    const nose = f.reduce((b, q, i) => (q[0] < f[b][0] ? i : b), 0);
    if (Math.abs(nose - (f.n - 1)) > 2)
      bad(`NACA ${code} leading edge is at index ${nose}, expected near ${f.n - 1}`);
    let camberMax = 0;
    for (let i = 1; i < f.n - 1; i++) {
      camberMax = Math.max(camberMax, (f[i][1] + f[f.length - i][1]) / 2);
    }
    const wantCamber = parseInt(String(code)[0], 10) / 100;
    if (Math.abs(camberMax - wantCamber) > 2e-3)
      bad(`NACA ${code} camber ${camberMax.toFixed(4)}, should be ${wantCamber}`);
    console.log(`  NACA ${code}: ${f.length} pts, t/c ${t.toFixed(4)}, ` +
                `camber ${camberMax.toFixed(4)}, TE closes to ${gap.toExponential(1)}`);
  }
}

/* ---- surfaces ---------------------------------------------------------- */
const CASES = [
  { name: 'wing',        build: () => wing() },
  { name: 'htail',       build: () => horizontalTail() },
  { name: 'no crank',    build: () => wing({ kink: null }) },
  { name: 'unswept',     build: () => wing({ sweep: 0, sweepOuter: 0, dihedral: 0 }) },
  { name: 'cranked out', build: () => wing({ sweepOuter: 35 }) },
  { name: 'one side',    build: () => liftingSurface({ mirror: false, kink: null,
                                                       dihedral: 0, twistTip: 0 }) },
];

for (const c of CASES) {
  const g = c.build(), u = g.userData, p = u.planform;
  const geo = g.children[0].geometry;
  let panelJump = 1;
  const pos = geo.getAttribute('position'), idx = geo.getIndex().array;
  console.log(`\n${c.name}: S ${u.area.toFixed(2)}, b ${u.span.toFixed(2)}, ` +
              `AR ${u.aspectRatio.toFixed(2)}, taper ${u.taperRatio.toFixed(3)}`);

  /* the planform is what was asked for --------------------------------- */
  const wantArea = u.mirror ? p.area : p.area / 2;
  if (Math.abs(u.area - wantArea) / wantArea > 2e-3)
    bad(`area ${u.area.toFixed(3)}, asked ${wantArea.toFixed(3)}`);
  if (Math.abs(u.taperRatio - p.taperRatio) > 1e-9)
    bad(`taper ${u.taperRatio.toFixed(4)}, asked ${p.taperRatio}`);

  /* closed, outward ------------------------------------------------------ */
  const wi = new Int32Array(pos.count), w = new Map();
  for (let i = 0; i < pos.count; i++) {
    const k = [pos.getX(i), pos.getY(i), pos.getZ(i)]
      .map((v) => Math.round(v * 1e6) + 0).join(',');
    if (!w.has(k)) w.set(k, w.size);
    wi[i] = w.get(k);
  }
  const edge = new Map();
  for (let t = 0; t + 2 < idx.length; t += 3) {
    const q = [wi[idx[t]], wi[idx[t + 1]], wi[idx[t + 2]]];
    if (q[0] === q[1] || q[1] === q[2] || q[0] === q[2]) continue;
    for (let k = 0; k < 3; k++) {
      const a = q[k], b = q[(k + 1) % 3], key = a < b ? `${a},${b}` : `${b},${a}`;
      edge.set(key, (edge.get(key) || 0) + 1);
    }
  }
  const open = [...edge.values()].filter((n) => n !== 2).length;
  if (open) bad(`skin has ${open} boundary edges -- not closed`);

  let vol = 0, degenerate = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  const cc = new THREE.Vector3(), n = new THREE.Vector3();
  for (let t = 0; t + 2 < idx.length; t += 3) {
    a.fromBufferAttribute(pos, idx[t]);
    b.fromBufferAttribute(pos, idx[t + 1]);
    cc.fromBufferAttribute(pos, idx[t + 2]);
    vol += a.dot(n.crossVectors(b, cc)) / 6;
    // Degenerate, not merely ELONGATED. A wing's quads are supposed to be long
    // and thin -- panels a few millimetres wide near the leading edge against
    // spanwise stations most of a metre apart -- and an area-over-longest-edge
    // test condemns every one of them. What matters is zero area: a face with
    // no normal. Measured against the two short edges rather than the long one.
    const e = [a.distanceTo(b), b.distanceTo(cc), cc.distanceTo(a)].sort((x, y) => x - y);
    const area2 = n.crossVectors(b.clone().sub(a), cc.clone().sub(a)).length() / 2;
    if (e[1] > 1e-12 && area2 / (e[0] * e[1]) < 1e-3) degenerate++;
  }
  if (vol <= 0) bad(`enclosed volume ${vol.toFixed(4)} -- wound inward`);
  if (degenerate) bad(`${degenerate} degenerate faces`);

  // And the chordwise panels should grow smoothly around the section: a jump
  // there is a band of stretched quads running the whole span.
  {
    const M2 = 2 * p.nChord - 2;
    const va = new THREE.Vector3(), vb = new THREE.Vector3();
    const mid = Math.floor((pos.count / M2) / 2) * M2;
    const wdt = [];
    for (let j = 0; j < M2; j++) {
      va.fromBufferAttribute(pos, mid + j);
      vb.fromBufferAttribute(pos, mid + ((j + 1) % M2));
      wdt.push(va.distanceTo(vb));
    }
    let worst = 1;
    for (let j = 0; j < M2; j++) {
      const q = wdt[(j + 1) % M2] / Math.max(wdt[j], 1e-12);
      worst = Math.max(worst, q, 1 / q);
    }
    if (worst > 4) bad(`chordwise panels jump ${worst.toFixed(1)}x between neighbours`);
    panelJump = worst;
  }

  /* MAC against the closed form ----------------------------------------- */
  // Only for a plain trapezoid: a cranked planform has no such formula, which
  // is exactly why the code integrates instead of using one.
  if (p.kink == null) {
    const L = u.taperRatio, cr = u.rootChord;
    const macWant = (2 / 3) * cr * (1 + L + L * L) / (1 + L);
    const yWant = (u.span / 6) * (1 + 2 * L) / (1 + L);
    if (Math.abs(u.mac - macWant) / macWant > 2e-3)
      bad(`MAC ${u.mac.toFixed(4)}, closed form gives ${macWant.toFixed(4)}`);
    if (Math.abs(u.yMac - yWant) / yWant > 3e-3)
      bad(`y_MAC ${u.yMac.toFixed(4)}, closed form gives ${yWant.toFixed(4)}`);
    console.log(`  MAC ${u.mac.toFixed(4)} vs ${macWant.toFixed(4)} closed form, ` +
                `y_MAC ${u.yMac.toFixed(4)} vs ${yWant.toFixed(4)}`);
  } else {
    console.log(`  MAC ${u.mac.toFixed(4)} at y ${u.yMac.toFixed(4)} (cranked -- integrated)`);
  }

  /* sweep, dihedral and twist come out as asked -------------------------- */
  // Measured off the built surface: the sweep of the reference line between two
  // stations, the rise of the same line, and the rotation of the tip section.
  const t0 = p.kink == null ? 0.30 : p.kink * 0.5, t1 = p.kink == null ? 0.70 : p.kink * 0.9;
  const s0 = u.at(t0), s1 = u.at(t1);
  const ref = (s) => s.xLE + p.sweepAt * s.chord;
  const gotSweep = Math.atan2(ref(s1) - ref(s0), (t1 - t0) * u.semiSpan) / DEG;
  if (Math.abs(gotSweep - p.sweep) > 0.05)
    bad(`inboard sweep measures ${gotSweep.toFixed(2)} deg, asked ${p.sweep}`);
  const gotDihedral = Math.atan2(s1.y - s0.y, (t1 - t0) * u.semiSpan) / DEG;
  if (Math.abs(gotDihedral - p.dihedral) > 0.05)
    bad(`dihedral measures ${gotDihedral.toFixed(2)} deg, asked ${p.dihedral}`);

  if (p.sweepOuter != null && p.kink != null) {
    const o0 = u.at(p.kink + 0.1 * (1 - p.kink)), o1 = u.at(1);
    const gotOuter = Math.atan2(ref(o1) - ref(o0),
      (1 - p.kink * 1.1 + p.kink * 0.1) * 0 + (1 - (p.kink + 0.1 * (1 - p.kink))) * u.semiSpan) / DEG;
    if (Math.abs(gotOuter - p.sweepOuter) > 0.05)
      bad(`outboard sweep measures ${gotOuter.toFixed(2)} deg, asked ${p.sweepOuter}`);
  }

  // Twist, off the geometry rather than off the parameter: the angle of the tip
  // section's own chord line against the root's.
  const M = 2 * p.nChord - 2;
  const nRing = pos.count / M;
  const chordAngle = (ring) => {
    const le = new THREE.Vector3().fromBufferAttribute(pos, ring * M + p.nChord - 1);
    const te = new THREE.Vector3().fromBufferAttribute(pos, ring * M);
    return Math.atan2(le.y - te.y, (le.z - te.z)) / DEG;
  };
  const gotTwist = chordAngle(nRing - 1) - chordAngle(u.mirror ? (nRing - 1) / 2 : 0);
  if (Math.abs(gotTwist - (p.twistTip - p.twistRoot)) > 0.1)
    bad(`tip twist measures ${gotTwist.toFixed(2)} deg, asked ` +
        `${(p.twistTip - p.twistRoot).toFixed(2)}`);

  console.log(`  sweep ${gotSweep.toFixed(2)} deg, dihedral ${gotDihedral.toFixed(2)} deg, ` +
              `tip twist ${gotTwist.toFixed(2)} deg`);
  console.log(`  ${nRing} stations x ${M} points, volume ${vol.toFixed(3)}, ` +
              `${open} open edges, ${degenerate} degenerate, ` +
              `panel jump ${panelJump.toFixed(2)}x`);
}

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: planforms match their standard definitions');
process.exit(failures ? 1 : 0);
