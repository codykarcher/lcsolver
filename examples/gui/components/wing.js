/**
 * Wings, and anything else that is aerofoils lofted along a span.
 *
 * A horizontal tail is a wing with no crank, so there is one loft here and the
 * rest is what gets fed to it -- two implementations of one loft would be two
 * places for one bug.
 *
 * The loft runs over a list of **frames**: a leading-edge point, a chord and a
 * twist. Anything that wants to extend or bend a surface adds frames rather
 * than adding a code path.
 *
 * **Defined geometrically, and measured afterwards.** Span, chords, sweep,
 * dihedral, twist -- the numbers you can read off a drawing with a ruler --
 * rather than reference area and aspect ratio. Those two are DERIVED and
 * reported, because they are results: a wing has an area because of its shape,
 * not the other way round, and a sizing loop that hands you an area has already
 * decided a planform to go with it.
 *
 * Nothing here is solved for and nothing can contradict anything else. The
 * earlier version took the area as an input and back-solved a chord to make it
 * come out, which works but means one of the chords is not yours, and without a
 * crank there is no spare chord to give up -- the planform was over-determined
 * and quietly came out 5% off.
 *
 * Axis convention, as everywhere else here: **+X starboard, +Y up, -Z aft**.
 * The origin is the ROOT LEADING EDGE on the centreline.
 */
import * as THREE from 'three';
import { skin } from './materials.js';
import { asSection, thicknessOf } from './airfoil.js';

const DEG = Math.PI / 180;
const smooth = (s) => s * s * (3 - 2 * s);

/**
 * Standard planform parameters.
 *
 * Two things worth stating about the planform:
 *
 *   - the leading edge is ONE straight line from root to tip. Sweep is quoted
 *     at the leading edge and there is only one of it, so the crank is a break
 *     in the trailing edge alone -- which is what it is on a real airliner,
 *     where the inboard trailing edge is carrying the gear and the wing box;
 *   - the chords are one length and two ratios, and each ratio is measured off
 *     the chord INBOARD of it: the crank against the root, the tip against the
 *     crank. So they read outboard in the order the wing is built, and neither
 *     is the conventional "taper ratio" of tip over root -- which is reported
 *     separately, because on a cranked wing it describes a trapezoid that is
 *     not there.
 */
const WING = {
  span:         34.1,  // tip to tip
  rootChord:     6.0,  // at the centreline, in the same units as the span
  crankRatio:   0.73,  // crank chord / ROOT chord
  tipRatio:     0.27,  // tip chord / CRANK chord (/ root, if there is no crank)
  sweep:        27.0,  // degrees, at the LEADING EDGE
  dihedral:      6.0,  // degrees
  twistRoot:     0.0,  // degrees, positive leading edge up
  twistKink:    null,  // null interpolates between root and tip
  twistTip:     -3.0,  // negative is washout
  twistAxis:    0.25,  // chord fraction the sections are twisted about
  kink:         0.35,  // fraction of semispan, or null for a plain trapezoid
  root: '2412', kinkFoil: null, tip: '2410',   // kinkFoil null means blended


  nChord:         80,  // points around each section
  nInner:          8,  // spanwise stations, root to crank
  nOuter:         16,  // crank to tip
};

/** A plain swept trapezoid: no crank, symmetric sections. */
const TAIL = {
  ...WING,
  span:         12.8, rootChord: 3.94, tipRatio: 0.30,
  sweep:        32.0, dihedral: 5.0,
  twistTip:      0.0, kink: null,
  root: '0010', tip: '0010',
  nInner:          2, nOuter: 20,
};

/**
 * The three chords, from one length and two ratios. Nothing is solved for.
 *
 * Without a crank there is nothing for `crankRatio` to apply to, so `tipRatio`
 * falls back to being measured off the root -- otherwise a tail would silently
 * get a tip chord scaled by a ratio it never had.
 */
function chords({ rootChord, crankRatio, tipRatio, kink }) {
  if (kink == null) return { cRoot: rootChord, cKink: null, cTip: rootChord * tipRatio };
  const cKink = rootChord * crankRatio;
  return { cRoot: rootChord, cKink, cTip: cKink * tipRatio };
}

/** Linear interpolation between root, crank and tip values of something. */
function alongSpan(t, kink, root, mid, tip) {
  if (kink == null || mid == null) return root + (tip - root) * t;
  return t <= kink
    ? root + (mid - root) * (t / kink)
    : mid + (tip - mid) * ((t - kink) / (1 - kink));
}

/**
 * A lifting surface: aerofoils lofted along a span.
 *
 * @param {boolean} mirror  build both halves. False gives one side.
 */
export function liftingSurface({ mirror = true, ...overrides } = {}) {
  const p = { ...WING, ...overrides };
  const span = p.span, semi = span / 2;
  const { cRoot, cKink, cTip } = chords(p);

  const rootFoil = asSection(p.root, p.nChord);
  const tipFoil = asSection(p.tip, p.nChord);
  const kinkFoil = p.kinkFoil == null ? null : asSection(p.kinkFoil, p.nChord);

  const chordAt = (t) => alongSpan(t, p.kink, cRoot, cKink, cTip);
  const twistAt = (t) => alongSpan(t, p.kink, p.twistRoot, p.twistKink, p.twistTip);
  // One straight leading edge, so this is one line and not a walk panel by
  // panel. The crank lives entirely in the trailing edge.
  const leAt = (t) => t * semi * Math.tan(p.sweep * DEG);

  const foilAt = (t) => {
    if (p.kink == null || kinkFoil == null) return { a: rootFoil, b: tipFoil, f: t };
    return t <= p.kink
      ? { a: rootFoil, b: kinkFoil, f: t / p.kink }
      : { a: kinkFoil, b: tipFoil, f: (t - p.kink) / (1 - p.kink) };
  };

  /* ---- frames --------------------------------------------------------- */
  // One list for the starboard half, tip outward, then mirrored. A frame is a
  // leading-edge point, a chord and a twist.
  const half = [];
  const push = (x, y, zLE, chord, twist, foil) =>
    half.push({ x, y, zLE, chord, twist, foil });

  const etas = [0];
  if (p.kink != null) {
    for (let i = 1; i <= p.nInner; i++) etas.push(p.kink * i / p.nInner);
    for (let i = 1; i <= p.nOuter; i++) etas.push(p.kink + (1 - p.kink) * i / p.nOuter);
  } else {
    for (let i = 1; i <= p.nInner + p.nOuter; i++) etas.push(i / (p.nInner + p.nOuter));
  }
  for (const t of etas) {
    push(t * semi, t * semi * Math.tan(p.dihedral * DEG), leAt(t),
         chordAt(t), twistAt(t), foilAt(t));
  }

  const frames = mirror
    ? [...half.slice(1).map((f) => ({ ...f, x: -f.x })).reverse(), ...half]
    : half;

  /* ---- loft ----------------------------------------------------------- */
  const M = rootFoil.length;
  const pos = [], idx = [];
  for (const fr of frames) {
    const eps = fr.twist * DEG, ce = Math.cos(eps), se = Math.sin(eps);
    for (let i = 0; i < M; i++) {
      const { a, b, f } = fr.foil;
      const q0 = a[i][0] + (b[i][0] - a[i][0]) * f;
      const v0 = a[i][1] + (b[i][1] - a[i][1]) * f;
      const dq = q0 - p.twistAxis;
      const q = p.twistAxis + dq * ce + v0 * se;
      const v = (-dq * se + v0 * ce) * fr.chord;
      pos.push(fr.x, fr.y + v, -(fr.zLE + q * fr.chord));
    }
  }
  for (let i = 0; i < frames.length - 1; i++) {
    for (let j = 0; j < M; j++) {
      const A = i * M + j, B = i * M + ((j + 1) % M);
      idx.push(A, B, A + M, B, B + M, A + M);
    }
  }

  // Close the two ends. An aerofoil is a long thin loop, so a fan from its
  // centre gives every triangle a one-point base and a half-chord length.
  // Zipping the upper surface to the lower gives triangles one station wide and
  // the local thickness tall. The chains MEET at the leading and trailing
  // edges, so those two rungs are triangles rather than quads.
  for (const [ring, flip] of [[0, true], [frames.length - 1, false]]) {
    const base = ring * M;
    for (let i = 0; i < p.nChord - 1; i++) {
      const uA = base + i, uB = base + i + 1;
      const lA = base + ((M - i) % M), lB = base + M - i - 1;
      const tri = (a, b, c) => (flip ? idx.push(a, c, b) : idx.push(a, b, c));
      if (uA === lA) tri(uA, uB, lB);
      else if (uB === lB) tri(uA, uB, lA);
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
  // Integrated from the frames, so these measure the surface that was built
  // rather than restating the request.
  const panel = half;
  let S = 0, macNum = 0, yNum = 0, xNum = 0;
  for (let i = 0; i < panel.length - 1; i++) {
    const a = panel[i], b = panel[i + 1], dy = b.x - a.x;
    S += (a.chord + b.chord) / 2 * dy;
    macNum += (a.chord ** 2 + b.chord ** 2) / 2 * dy;
    yNum += (a.x * a.chord + b.x * b.chord) / 2 * dy;
    xNum += (a.zLE * a.chord + b.zLE * b.chord) / 2 * dy;
  }
  // Area and aspect ratio are RESULTS. Integrated from the frames, so they are
  // a measurement of the surface that was built.
  const area = mirror ? 2 * S : S;
  const refArea = 2 * S;                      // both halves, whatever was built
  const mac = macNum / S;

  Object.assign(g.userData, {
    span, semiSpan: semi, area, referenceArea: refArea, mirror,
    /** Derived, both of them: b^2 / S on the full reference area. */
    aspectRatio: span * span / refArea,
    rootChord: cRoot, kinkChord: cKink, tipChord: cTip,
    /**
     * The conventional taper ratio, tip over ROOT -- derived, and not one of the
     * inputs. On a cranked wing it describes a straight trapezoid that does not
     * exist, which is why it is not what the wing is built from.
     */
    taperRatio: cTip / cRoot,
    crankRatio: cKink ? cKink / cRoot : null, tipRatio: cTip / (cKink ?? cRoot),
    sweep: p.sweep, dihedral: p.dihedral,
    mac, yMac: yNum / S, xMacLE: xNum / S, xMacQuarter: xNum / S + 0.25 * mac,
    rootThickness: thicknessOf(rootFoil), tipThickness: thicknessOf(tipFoil),
    sections: { root: rootFoil.name, kink: kinkFoil?.name ?? null, tip: tipFoil.name },
    frames: half, planform: p, skinMesh: mesh,
    /** Chord, leading edge and twist at any fraction of semispan. */
    at: (t) => ({ chord: chordAt(t), xLE: leAt(t),
                  y: t * semi * Math.tan(p.dihedral * DEG), twist: twistAt(t) }),
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
