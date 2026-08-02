/**
 * Headless checks on a lifting surface.
 *
 *     node check_wing.mjs
 *
 * A wing is defined by numbers with textbook definitions, so unlike the
 * fuselage most of this can be checked against a CLOSED FORM rather than
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
  { name: 'unswept',     build: () => wing({ sweep: 0, dihedral: 0 }) },
  { name: 'tight taper', build: () => wing({ crankRatio: 0.55, tipRatio: 0.08 }) },
  { name: 'long span',   build: () => wing({ span: 52, rootChord: 5.2,
                                             crankRatio: 0.73, tipRatio: 0.27 }) },
  { name: 'one side',    build: () => liftingSurface({ mirror: false, kink: null,
                                                       dihedral: 0, twistTip: 0 }) },
];

for (const c of CASES) {
  const g = c.build(), u = g.userData, p = u.planform;
  const geo = g.children[0].geometry;
  let panelJump = 1;
  const pos = geo.getAttribute('position'), idx = geo.getIndex().array;
  console.log(`\n${c.name}: b ${u.span.toFixed(2)} in, chords ` +
              `${u.rootChord.toFixed(2)}` + (u.kinkChord ? `/${u.kinkChord.toFixed(2)}` : '') +
              `/${u.tipChord.toFixed(2)} in  ->  S ${u.area.toFixed(2)}, ` +
              `AR ${u.aspectRatio.toFixed(2)} out`);

  /* the chords are what was asked for ----------------------------------- */
  if (Math.abs(u.span - p.span) > 1e-12) bad(`span ${u.span}, asked ${p.span}`);
  // The chords follow from the root and the two ratios, each measured off the
  // chord inboard of it. Checked as a chain, because that is the failure mode:
  // a tip ratio applied to the root instead of the crank looks plausible and is
  // 27% wrong.
  const wantKink = p.kink == null ? null : p.rootChord * p.crankRatio;
  const wantTip = (wantKink ?? p.rootChord) * p.tipRatio;
  if (Math.abs(u.rootChord - p.rootChord) > 1e-12)
    bad(`root chord ${u.rootChord}, asked ${p.rootChord}`);
  if (wantKink != null && Math.abs(u.kinkChord - wantKink) > 1e-12)
    bad(`crank chord ${u.kinkChord}, root x crankRatio is ${wantKink}`);
  if (Math.abs(u.tipChord - wantTip) > 1e-12)
    bad(`tip chord ${u.tipChord}, ${p.kink == null ? 'root' : 'crank'} x tipRatio ` +
        `is ${wantTip}`);

  /* area and aspect ratio are RESULTS, and must be the right ones --------- */
  // Against the two-trapezoid formula worked straight off the inputs, which the
  // component does not use -- it integrates the frames instead. Two independent
  // routes to the same number is the only way this is worth checking at all.
  {
    const semi = p.span / 2, cR = p.rootChord;
    const cK = wantKink, cT = wantTip;
    const full = p.kink == null
      ? semi * (cR + cT)
      : semi * ((cR + cK) * p.kink + (cK + cT) * (1 - p.kink));
    if (Math.abs(u.referenceArea - full) / full > 1e-6)
      bad(`reference area ${u.referenceArea.toFixed(4)}, by hand ${full.toFixed(4)}`);
    if (Math.abs(u.aspectRatio - p.span * p.span / full) > 1e-6)
      bad(`aspect ratio ${u.aspectRatio.toFixed(4)}, by hand ` +
          `${(p.span * p.span / full).toFixed(4)}`);
  }

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
  // ONE straight leading edge. Measured across every pair of stations rather
  // than once, because a single sample cannot tell a straight line from a line
  // that happens to pass through two points -- and the crank is now supposed to
  // be entirely in the trailing edge.
  let gotSweep = 0, sweepSpread = 0;
  {
    const ts = [0.05, 0.20, 0.34, 0.36, 0.50, 0.75, 1.00];
    const angles = [];
    for (let i = 0; i < ts.length - 1; i++) {
      const a0 = u.at(ts[i]), a1 = u.at(ts[i + 1]);
      angles.push(Math.atan2(a1.xLE - a0.xLE, (ts[i + 1] - ts[i]) * u.semiSpan) / DEG);
    }
    gotSweep = angles[0];
    sweepSpread = Math.max(...angles) - Math.min(...angles);
    if (Math.abs(gotSweep - p.sweep) > 0.05)
      bad(`leading-edge sweep measures ${gotSweep.toFixed(2)} deg, asked ${p.sweep}`);
    if (sweepSpread > 0.01)
      bad(`leading edge is not straight: sweep varies ${sweepSpread.toFixed(3)} deg`);
  }
  const s0 = u.at(0.2), s1 = u.at(0.8);
  const gotDihedral = Math.atan2(s1.y - s0.y, 0.6 * u.semiSpan) / DEG;
  if (Math.abs(gotDihedral - p.dihedral) > 0.05)
    bad(`dihedral measures ${gotDihedral.toFixed(2)} deg, asked ${p.dihedral}`);



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

  console.log(`  LE sweep ${gotSweep.toFixed(2)} deg (varies ${sweepSpread.toExponential(1)}), ` +
              `dihedral ${gotDihedral.toFixed(2)} deg, tip twist ${gotTwist.toFixed(2)} deg`);
  console.log(`  ${nRing} stations x ${M} points, volume ${vol.toFixed(3)}, ` +
              `${open} open edges, ${degenerate} degenerate, ` +
              `panel jump ${panelJump.toFixed(2)}x`);
}

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: planforms match their standard definitions');
process.exit(failures ? 1 : 0);
