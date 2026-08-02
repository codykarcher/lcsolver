/**
 * Wings, and anything else that is aerofoils lofted along a span.
 *
 * A horizontal tail is a wing with no crank; a fin is a wing with no crank and
 * no port half. So there is one builder here and the rest are wrappers, because
 * three implementations of the same loft is three places for the same bug.
 *
 * **Parameterised the standard way.** Area, aspect ratio, taper ratio, sweep at
 * a reference chord fraction, dihedral, twist -- the numbers that appear on a
 * three-view and in every textbook -- rather than by chords and coordinates.
 * The chords are DERIVED: given the reference area and the planform ratios,
 * there is exactly one root chord that makes the area come out, and it is
 * solved for rather than typed in. Nothing else in this file is free to
 * disagree with the area you asked for.
 *
 * Axis convention, as everywhere else here: **+X starboard, +Y up, -Z aft**.
 * The origin is the ROOT LEADING EDGE on the centreline, so a wing dropped into
 * a scene beside a fuselage needs one number to place it.
 */
import * as THREE from 'three';
import { skin } from './materials.js';
import { asSection, thicknessOf } from './airfoil.js';

const DEG = Math.PI / 180;

/**
 * Standard planform parameters.
 *
 * `kink` is the one that makes it a wing rather than a tail: the fraction of
 * semispan at which the planform breaks. Null means a plain trapezoid, which is
 * what a tail is. Airliners break because the inboard trailing edge is carrying
 * the undercarriage and the wing box, so the inboard panel is deeper than a
 * straight taper would make it -- `kinkTaper` above the straight-taper value is
 * exactly that.
 */
const WING = {
  area:        124.0,  // reference area, both halves
  aspectRatio:   9.4,  // b^2 / S
  taperRatio:   0.16,  // tip chord / root chord
  sweep:        25.0,  // degrees, at `sweepAt`
  sweepAt:      0.25,  // chord fraction the sweep is quoted at
  sweepOuter:   null,  // outboard panel sweep; null means the same
  dihedral:      6.0,  // degrees
  twistRoot:     0.0,  // degrees, positive leading edge up
  twistKink:    null,  // null interpolates between root and tip
  twistTip:     -3.0,  // negative is washout
  twistAxis:    0.25,  // chord fraction the sections are twisted about
  kink:         0.35,  // fraction of semispan, or null for a plain trapezoid
  kinkTaper:    0.60,  // kink chord / root chord
  root: '2412', kinkFoil: null, tip: '2410',   // kinkFoil null means blended
  nChord:         80,  // points around each section
  nInner:          8,  // spanwise stations, root to kink
  nOuter:         16,  // and kink to tip
};

/** A plain swept trapezoid: no crank, and thinner symmetric sections. */
const TAIL = {
  ...WING,
  area:         32.8, aspectRatio: 5.0, taperRatio: 0.30,
  sweep:        30.0, dihedral: 5.0,
  twistTip:      0.0, kink: null,
  root: '0010', tip: '0010',
  nInner:          2, nOuter: 20,
};

/**
 * Chord at the root that makes the reference area come out.
 *
 * The planform is two trapezoids, so S = (b/2) * c_root * [(1 + k) eta
 * + (k + lambda)(1 - eta)] with k the kink taper and lambda the tip taper.
 * Inverting that is the whole of it -- but it is worth doing rather than
 * letting the caller supply a root chord, because then the area, the aspect
 * ratio and the chords are three numbers that can contradict each other and
 * two of them will be wrong.
 */
function rootChordFor({ area, span, taperRatio, kink, kinkTaper }) {
  const shape = kink == null
    ? 1 + taperRatio
    : (1 + kinkTaper) * kink + (kinkTaper + taperRatio) * (1 - kink);
  return 2 * area / (span * shape);
}

/**
 * The spanwise stations, as fractions of semispan, running port tip to
 * starboard tip.
 *
 * A station is placed EXACTLY on the kink. The crank is a real crease in the
 * planform and is meant to be sharp; letting the loft straddle it would round
 * it off over one panel width and put the break in the wrong place as well.
 */
function spanStations(kink, nInner, nOuter, mirror) {
  const half = [0];
  if (kink != null) {
    for (let i = 1; i <= nInner; i++) half.push(kink * i / nInner);
    for (let i = 1; i <= nOuter; i++) half.push(kink + (1 - kink) * i / nOuter);
  } else {
    for (let i = 1; i <= nInner + nOuter; i++) half.push(i / (nInner + nOuter));
  }
  return mirror ? [...half.slice(1).map((e) => -e).reverse(), ...half] : half;
}

/** Linear interpolation between the root, kink and tip values of something. */
function alongSpan(t, kink, root, mid, tip) {
  if (kink == null || mid == null) return root + (tip - root) * t;
  return t <= kink
    ? root + (mid - root) * (t / kink)
    : mid + (tip - mid) * ((t - kink) / (1 - kink));
}

/**
 * A lifting surface: aerofoils lofted along a span.
 *
 * @param {object} p  planform, overriding WING above.
 * @param {boolean} mirror  build both halves. False gives a single fin.
 */
export function liftingSurface({ mirror = true, ...overrides } = {}) {
  const p = { ...WING, ...overrides };
  const span = Math.sqrt(p.aspectRatio * p.area);
  const semi = span / 2;
  const cRoot = rootChordFor({ ...p, span });
  const cKink = p.kink == null ? null : cRoot * p.kinkTaper;
  const cTip = cRoot * p.taperRatio;
  const sweepOut = p.sweepOuter ?? p.sweep;

  const rootFoil = asSection(p.root, p.nChord);
  const tipFoil = asSection(p.tip, p.nChord);
  const kinkFoil = p.kinkFoil == null ? null : asSection(p.kinkFoil, p.nChord);

  const chordAt = (t) => alongSpan(t, p.kink, cRoot, cKink, cTip);
  const twistAt = (t) => alongSpan(t, p.kink, p.twistRoot, p.twistKink, p.twistTip);

  /**
   * Aft position of the sweep reference point at |eta| = t.
   *
   * Built by walking out panel by panel rather than from a single angle, so an
   * outboard sweep different from the inboard one starts where the inboard one
   * finished instead of from the centreline. Getting that wrong moves the whole
   * outer wing fore or aft and is invisible in a three-quarter view.
   */
  const refAt = (t) => {
    const at0 = p.sweepAt * cRoot;
    if (p.kink == null) return at0 + t * semi * Math.tan(p.sweep * DEG);
    const atKink = at0 + p.kink * semi * Math.tan(p.sweep * DEG);
    return t <= p.kink
      ? at0 + t * semi * Math.tan(p.sweep * DEG)
      : atKink + (t - p.kink) * semi * Math.tan(sweepOut * DEG);
  };

  /** Section shape at |eta| = t: root, kink and tip blended point for point. */
  const foilAt = (t) => {
    if (p.kink == null || kinkFoil == null) {
      return { a: rootFoil, b: tipFoil, f: t };
    }
    return t <= p.kink
      ? { a: rootFoil, b: kinkFoil, f: t / p.kink }
      : { a: kinkFoil, b: tipFoil, f: (t - p.kink) / (1 - p.kink) };
  };

  const etas = spanStations(p.kink, p.nInner, p.nOuter, mirror);
  const M = rootFoil.length;                       // points around a section
  const pos = [], idx = [];
  const stations = [];

  for (const eta of etas) {
    const t = Math.abs(eta);
    const c = chordAt(t), ref = refAt(t);
    const xLE = ref - p.sweepAt * c;
    const yD = t * semi * Math.tan(p.dihedral * DEG);
    const eps = twistAt(t) * DEG;
    const ce = Math.cos(eps), se = Math.sin(eps);
    const { a, b, f } = foilAt(t);

    for (let i = 0; i < M; i++) {
      // Blend, then twist about the twist axis, then scale and place. Blending
      // BEFORE twisting matters: twist is a rigid rotation of the finished
      // section, not something to interpolate through.
      const q0 = a[i][0] + (b[i][0] - a[i][0]) * f;
      const v0 = a[i][1] + (b[i][1] - a[i][1]) * f;
      const dq = q0 - p.twistAxis;
      const q = p.twistAxis + dq * ce + v0 * se;
      const v = -dq * se + v0 * ce;
      pos.push(eta * semi, yD + v * c, -(xLE + q * c));
    }
    stations.push({ eta, chord: c, xLE, y: yD, twist: twistAt(t) });
  }

  for (let i = 0; i < etas.length - 1; i++) {
    for (let j = 0; j < M; j++) {
      const A = i * M + j, B = i * M + ((j + 1) % M);
      idx.push(A, B, A + M, B, B + M, A + M);
    }
  }

  // Close the two tips. An aerofoil is a long thin loop, so a fan from its
  // centre is the wrong triangulation for exactly the reason the fuselage's
  // trailing edge was: every triangle gets a one-point base and a half-chord
  // length. Zipping the upper surface to the lower gives triangles one station
  // wide and the local thickness tall. The chains MEET at the leading and
  // trailing edges, so those two rungs are triangles rather than quads.
  const nHalf = p.nChord;
  for (const [ring, flip] of [[0, true], [etas.length - 1, false]]) {
    const base = ring * M;
    for (let i = 0; i < nHalf - 1; i++) {
      const uA = base + i, uB = base + i + 1;
      const lA = base + ((M - i) % M), lB = base + M - i - 1;
      const tri = (x, y, z) => (flip ? idx.push(x, z, y) : idx.push(x, y, z));
      if (uA === lA) tri(uA, uB, lB);              // trailing edge
      else if (uB === lB) tri(uA, uB, lA);         // leading edge
      else { tri(uA, uB, lA); tri(uB, lB, lA); }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();

  const g = new THREE.Group();
  const mesh = new THREE.Mesh(geo, skin);
  g.add(mesh);

  /* ---- what the planform actually came out as ------------------------- */
  // Integrated from the stations rather than from the formulae, so these are a
  // measurement of the surface that was built and not a restatement of the
  // request. If the two disagree, something between them is wrong and this is
  // where it shows.
  const half = stations.filter((s) => s.eta >= 0);
  let S = 0, macNum = 0, yNum = 0, xNum = 0;
  for (let i = 0; i < half.length - 1; i++) {
    const a = half[i], b = half[i + 1];
    const dy = (b.eta - a.eta) * semi;
    S += (a.chord + b.chord) / 2 * dy;
    macNum += (a.chord ** 2 + b.chord ** 2) / 2 * dy;
    yNum += (a.eta * a.chord + b.eta * b.chord) / 2 * semi * dy;
    xNum += (a.xLE * a.chord + b.xLE * b.chord) / 2 * dy;
  }
  // Both halves when both are built, one when only one is. The `area` PARAMETER
  // always means the full reference area, so a single-sided surface comes out
  // at half of it -- which is a statement about what was built, not a
  // disagreement with what was asked.
  const area = mirror ? 2 * S : S;
  const mac = macNum / S;

  Object.assign(g.userData, {
    span, semiSpan: semi, area, referenceArea: p.area, mirror,
    aspectRatio: span * span / p.area,
    taperRatio: cTip / cRoot,
    rootChord: cRoot, kinkChord: cKink, tipChord: cTip,
    sweep: p.sweep, sweepOuter: sweepOut, dihedral: p.dihedral,
    /** Mean aerodynamic chord, and where it is. Standard definitions. */
    mac, yMac: yNum / S, xMacLE: xNum / S,
    /** Aft position of the MAC quarter-chord -- what a tail arm is measured to. */
    xMacQuarter: xNum / S + 0.25 * mac,
    rootThickness: thicknessOf(rootFoil), tipThickness: thicknessOf(tipFoil),
    sections: { root: rootFoil.name, kink: kinkFoil?.name ?? null, tip: tipFoil.name },
    stations, planform: p, skinMesh: mesh,
    /** Chord, leading-edge position and twist at any fraction of semispan. */
    at: (t) => ({
      chord: chordAt(t), xLE: refAt(t) - p.sweepAt * chordAt(t),
      y: t * semi * Math.tan(p.dihedral * DEG), twist: twistAt(t),
    }),
  });
  return g;
}

/** A wing: cranked, cambered, washed out. */
export function wing(o = {}) { return liftingSurface({ ...WING, ...o }); }

/** A horizontal tail: the same loft with no crank and symmetric sections. */
export function horizontalTail(o = {}) { return liftingSurface({ ...TAIL, ...o }); }


// No fin here yet on purpose. A vertical tail's span is a height, its area is
// one surface not two, and its aspect ratio is defined on those -- so it is a
// different set of conventions rather than a wing with mirror off, and
// pretending otherwise would put three numbers in userData that quietly mean
// something else. `liftingSurface({ mirror: false })` builds the geometry.
export const surfaces = { wing, horizontalTail };
