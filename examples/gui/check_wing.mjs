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
import { wing, horizontalTail, verticalTail, liftingSurface } from './components/wing.js';
import { naca4, sectionPoints, thicknessOf } from './components/airfoil.js';

const DEG = Math.PI / 180;
let failures = 0;
const bad = (msg) => { console.log(`  FAIL  ${msg}`); failures++; };

/* ---- sections ---------------------------------------------------------- */
{
  console.log('aerofoils:');
  for (const code of ['0010', '2412', '4415']) {
    const f = naca4(code);
    const want = parseInt(String(code).slice(2), 10) / 100;
    const wantCamber = parseInt(String(code)[0], 10) / 100;
    // Tolerance rather than equality, because these are the maxima of SAMPLED
    // arrays: the cosine stations do not land exactly on x = 0.30 where the
    // thickness peaks, nor on the camber position, so the sampled maximum sits
    // a little under the analytic one. That is discretisation, not error.
    if (Math.abs(thicknessOf(f) - want) > 1e-3)
      bad(`NACA ${code} is ${thicknessOf(f).toFixed(5)} thick, should be ${want}`);
    if (Math.abs(Math.max(...f.camber) - wantCamber) > 2e-4)
      bad(`NACA ${code} camber ${Math.max(...f.camber).toFixed(5)}, should be ${wantCamber}`);

    const pts = sectionPoints(f);
    if (pts.length !== 2 * f.n - 2)
      bad(`NACA ${code} gave ${pts.length} points, expected ${2 * f.n - 2}`);
    if (Math.abs(pts[0][0] - 1) > 1e-9)
      bad(`NACA ${code} does not start at the trailing edge`);

    // Thickness scaling must move the thickness and leave the CAMBER alone.
    // Scaling the y of a point list instead scales both, and turns a 12%
    // section into a differently-cambered 10% one -- which is exactly the
    // failure this representation exists to make impossible.
    const scaled = sectionPoints(f, 0.10 / Math.max(thicknessOf(f), 1e-9));
    let gotT = 0, gotCamber = 0, wasCamber = 0;
    for (let i = 1; i < f.n - 1; i++) {
      const j = 2 * f.n - 2 - i;
      gotT = Math.max(gotT, scaled[i][1] - scaled[j][1]);
      gotCamber = Math.max(gotCamber, (scaled[i][1] + scaled[j][1]) / 2);
      wasCamber = Math.max(wasCamber, (pts[i][1] + pts[j][1]) / 2);
    }
    if (Math.abs(gotT - 0.10) > 2e-3)
      bad(`NACA ${code} scaled to 0.10 came out ${gotT.toFixed(4)}`);
    if (Math.abs(gotCamber - wasCamber) > 2e-3)
      bad(`scaling NACA ${code} moved the camber ${wasCamber.toFixed(4)} -> ` +
          `${gotCamber.toFixed(4)}`);
    console.log(`  NACA ${code}: t/c ${thicknessOf(f).toFixed(4)}, ` +
                `camber ${Math.max(...f.camber).toFixed(4)}, ${pts.length} pts; ` +
                `scaled to 0.10 -> ${gotT.toFixed(4)} with camber held ` +
                `${wasCamber.toFixed(4)} -> ${gotCamber.toFixed(4)}`);
  }
}

/* ---- surfaces ---------------------------------------------------------- */
const CASES = [
  { name: 'wing',        build: () => wing() },
  { name: 'htail',       build: () => horizontalTail() },
  { name: 'no crank',    build: () => wing({ kink: null }) },
  { name: 'unswept',     build: () => wing({ sweep: 0, dihedral: 0 }) },
  { name: 'tight taper', build: () => wing({ crankRatio: 0.55, tipRatio: 0.08 }) },
  { name: 'thick root',  build: () => wing({ rootThickness: 0.17, crankThickness: 0.13,
                                             tipThickness: 0.085 }) },
  { name: 'const t/c',   build: () => wing({ thickness: 0.12 }) },
  { name: 'long span',   build: () => wing({ span: 52, rootChord: 5.2,
                                             crankRatio: 0.73, tipRatio: 0.27 }) },
  { name: 'one side',    build: () => liftingSurface({ mirror: false, kink: null,
                                                       dihedral: 0, twistTip: 0 }) },
  { name: 'fin',         build: () => verticalTail() },
  { name: 'tall fin',    build: () => verticalTail({ height: 9, taperRatio: 0.45 }) },
  { name: 'canted fin',  build: () => verticalTail({ cant: 30 }) },
  { name: 'canted -25',  build: () => verticalTail({ cant: -25 }) },
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
  // A fin's own conventions: height root to tip, area ONE surface, aspect ratio
  // on both of those. Mirror the same shape and read it as a wing and the
  // aspect ratio comes out twice as large, which is the error this separates.
  if (u.isFin) {
    if (Math.abs(u.height - p.height) > 1e-12)
      bad(`fin height ${u.height}, asked ${p.height}`);
    const byHand = u.height * (u.rootChord + u.tipChord) / 2;
    if (Math.abs(u.area - byHand) / byHand > 1e-6)
      bad(`fin area ${u.area.toFixed(4)}, by hand ${byHand.toFixed(4)}`);
    if (Math.abs(u.aspectRatio - u.height ** 2 / byHand) > 1e-6)
      bad(`fin AR ${u.aspectRatio.toFixed(4)}, h^2/S is ` +
          `${(u.height ** 2 / byHand).toFixed(4)}`);
    if (u.referenceArea !== u.area)
      bad(`fin reference area ${u.referenceArea} is not its own area ${u.area}`);

    // Cant turns the fin about its ROOT CHORD LINE. So that line does not move,
    // at any cant -- while the section's thickness sweeps about it, which is
    // what rotating about an axis means and is not the root moving.
    g.updateMatrixWorld(true);
    const M3 = 2 * p.nChord - 2;
    const le = new THREE.Vector3().fromBufferAttribute(pos, p.nChord - 1)
      .applyMatrix4(g.matrixWorld);
    const te = new THREE.Vector3().fromBufferAttribute(pos, 0).applyMatrix4(g.matrixWorld);
    if (le.length() > 1e-9)
      bad(`root leading edge moved to ${le.toArray().map((v) => v.toFixed(4))}`);
    if (Math.abs(te.x) > 1e-9 || Math.abs(te.y) > 1e-9
        || Math.abs(te.z + u.rootChord) > 1e-9)
      bad(`root trailing edge moved to ${te.toArray().map((v) => v.toFixed(4))}`);

    // And the tip goes where the cant says. Height is measured along the SPAN,
    // so a canted fin reaches only cos(cant) of it above the root -- confusing
    // the two is easy and quiet, so both are published and both are checked.
    const fr = u.frames[u.frames.length - 1];
    const tip = new THREE.Vector3(fr.x, fr.y, -fr.zLE).applyMatrix4(g.matrixWorld);
    if (Math.abs(tip.y - u.tipRise) > 1e-6)
      bad(`tip rises ${tip.y.toFixed(4)}, tipRise says ${u.tipRise.toFixed(4)}`);
    if (Math.abs(tip.x - u.tipOffset) > 1e-6)
      bad(`tip offset ${tip.x.toFixed(4)}, tipOffset says ${u.tipOffset.toFixed(4)}`);
    // Cant is a placement, not a shape: it cannot change the area.
    const upright = verticalTail({ ...p, cant: 0 }).userData;
    if (Math.abs(u.area - upright.area) > 1e-9 || Math.abs(u.mac - upright.mac) > 1e-9)
      bad(`cant changed the area or MAC`);
  } else if (Math.abs(u.span - p.span) > 1e-12) bad(`span ${u.span}, asked ${p.span}`);
  // The chords follow from the root and the two ratios, each measured off the
  // chord inboard of it. Checked as a chain, because that is the failure mode:
  // a tip ratio applied to the root instead of the crank looks plausible and is
  // 27% wrong.
  // Crankless surfaces take the plain taper ratio; cranked ones chain crankRatio
  // then tipRatio. Reading the wrong one gives a tip chord that is plausible and
  // 10% out, which is what this caught when the tail changed to taperRatio.
  const wantKink = p.kink == null ? null : p.rootChord * p.crankRatio;
  const ratio = p.kink == null ? (p.taperRatio ?? p.tipRatio) : p.tipRatio;
  const wantTip = (wantKink ?? p.rootChord) * ratio;
  if (Math.abs(u.rootChord - p.rootChord) > 1e-12)
    bad(`root chord ${u.rootChord}, asked ${p.rootChord}`);
  if (wantKink != null && Math.abs(u.kinkChord - wantKink) > 1e-12)
    bad(`crank chord ${u.kinkChord}, root x crankRatio is ${wantKink}`);
  if (Math.abs(u.tipChord - wantTip) > 1e-12)
    bad(`tip chord ${u.tipChord}, ${p.kink == null ? 'root' : 'crank'} x ` +
        `${p.kink == null ? 'taperRatio' : 'tipRatio'} is ${wantTip}`);

  /* area and aspect ratio are RESULTS, and must be the right ones --------- */
  if (!u.isFin)
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

  // The symmetry convention: a tail or fin is NACA 00xx however it was asked
  // for, and named for the thickness ACTUALLY built rather than the base
  // section the camber was stripped off.
  if (p.symmetric) {
    const pct = String(Math.round(u.rootThickness * 100)).padStart(2, '0');
    if (u.sections.root !== `NACA 00${pct}`)
      bad(`symmetric surface reports ${u.sections.root}, expected NACA 00${pct}`);
    // Measured: upper and lower must mirror about y = 0 at the root.
    const M2 = 2 * p.nChord - 2, base = (u.mirror ? (pos.count / M2 - 1) / 2 : 0) * M2;
    let worstSym = 0;
    for (let i = 1; i < p.nChord - 1; i++) {
      const yu = pos.getY(base + i), yl = pos.getY(base + M2 - i);
      worstSym = Math.max(worstSym, Math.abs(yu + yl - 2 * pos.getY(base)));
    }
    if (worstSym > 1e-6 * u.rootChord)
      bad(`symmetric surface is ${worstSym.toExponential(2)} off symmetric at the root`);
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
              `dihedral ${gotDihedral.toFixed(2)} deg, tip twist ${gotTwist.toFixed(2)} deg` +
              (u.isFin ? `, cant ${u.cant} -> tip ${u.tipRise.toFixed(2)} up / ` +
                         `${u.tipOffset.toFixed(2)} across` : ''));
  // Thickness measured off the BUILT surface at each named station, not read
  // back from the parameter it was set from.
  {
    const M2 = 2 * p.nChord - 2, nR = pos.count / M2;
    const tcAt = (ring) => {
      const va = new THREE.Vector3(), vb = new THREE.Vector3();
      let t = 0;
      for (let i = 1; i < p.nChord - 1; i++) {
        va.fromBufferAttribute(pos, ring * M2 + i);
        vb.fromBufferAttribute(pos, ring * M2 + (M2 - i));
        t = Math.max(t, va.distanceTo(vb));
      }
      va.fromBufferAttribute(pos, ring * M2 + p.nChord - 1);
      vb.fromBufferAttribute(pos, ring * M2);
      return t / va.distanceTo(vb);
    };
    // Rings are FOUND by where they are, not by counting stations. The spacing
    // changed when planform breaks got their own stations, and an index that
    // assumed the old layout landed one panel inboard of the crank -- reporting
    // 0.1075 against the 0.11 asked for, which looks like a thickness bug and
    // is a counting one.
    const ringAt = (eta) => {
      let best = 0, bd = Infinity;
      for (let r = 0; r < nR; r++) {
        const dd = Math.abs(pos.getX(r * M2) - eta * u.semiSpan);
        if (dd < bd) { bd = dd; best = r; }
      }
      return best;
    };
    const named = [['root', ringAt(0), u.rootThickness],
                   ['tip', ringAt(1), u.tipThickness]];
    if (p.kink != null && u.crankThickness != null) {
      named.push(['crank', ringAt(p.kink), u.crankThickness]);
    }
    for (const [what, ring, want] of named) {
      const got = tcAt(ring);
      if (Math.abs(got - want) > 2e-3)
        bad(`${what} t/c measures ${got.toFixed(4)}, asked ${want}`);
    }
    console.log(`  t/c: ${named.map(([w, r]) => `${w} ${tcAt(r).toFixed(3)}`).join(', ')}`);
  }

  console.log(`  ${nRing} stations x ${M} points, volume ${vol.toFixed(3)}, ` +
              `${open} open edges, ${degenerate} degenerate, ` +
              `panel jump ${panelJump.toFixed(2)}x`);
}

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: planforms match their standard definitions');
process.exit(failures ? 1 : 0);
