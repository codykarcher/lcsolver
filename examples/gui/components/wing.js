/**
 * Wings, and anything else that is aerofoils lofted along a span.
 *
 * A horizontal tail is a wing with no crank; a winglet is the same wing carried
 * on past the tip with its sections rolling up. So there is one loft here and
 * everything else is what gets fed to it, because three implementations of one
 * loft is three places for one bug.
 *
 * The loft runs over a list of **frames**. A frame is a leading-edge point, a
 * chord, a twist, and a ROLL -- the angle the section's own vertical makes with
 * the world's. Roll is what a winglet is: on a wing it is zero everywhere and
 * the sections stand streamwise, and past the tip it rolls up to the cant angle
 * and the same loft keeps going. Nothing about the winglet is a special case in
 * the meshing.
 *
 * **Parameterised the standard way.** Reference area, aspect ratio, taper,
 * leading-edge sweep, dihedral, twist -- the numbers on a three-view.
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
 * Three things about how the chords are pinned down, because they are the part
 * that is easy to get into a state where two numbers disagree:
 *
 *   - the leading edge is ONE straight line from root to tip. Sweep is quoted
 *     at the leading edge and there is only one of it, so the crank is a break
 *     in the trailing edge alone -- which is what it is on a real airliner,
 *     where the inboard trailing edge is carrying the gear and the wing box;
 *   - `taperRatio` is tip chord over CRANK chord, not over root chord. On a
 *     cranked wing the outboard panel is the one that has a taper ratio in any
 *     useful sense; the inboard panel is a fairing;
 *   - `rootChord` is an INPUT. The crank chord is then solved for so that the
 *     reference area comes out. One of the three chords has to be derived or
 *     the area is a wish rather than a constraint, and the crank is the one
 *     nobody has an independent opinion about.
 */
const WING = {
  area:        124.0,  // reference area, both halves
  aspectRatio:   9.4,  // b^2 / S
  rootChord:     6.0,  // INPUT; null derives it from the area instead
  taperRatio:   0.27,  // tip chord / CRANK chord
  sweep:        27.0,  // degrees, at the LEADING EDGE
  dihedral:      6.0,  // degrees
  twistRoot:     0.0,  // degrees, positive leading edge up
  twistKink:    null,  // null interpolates between root and tip
  twistTip:     -3.0,  // negative is washout
  twistAxis:    0.25,  // chord fraction the sections are twisted about
  kink:         0.35,  // fraction of semispan, or null for a plain trapezoid
  root: '2412', kinkFoil: null, tip: '2410',   // kinkFoil null means blended

  /* ---- winglet: visual only ------------------------------------------- */
  winglet:      true,
  wingletHeight: 0.11,  // developed length, in semispans
  wingletCant:  70.0,   // degrees from horizontal at the top
  wingletSweep: 42.0,   // degrees, its own leading edge
  wingletTaper: 0.35,   // its tip chord over the wing's tip chord
  wingletBlend:  0.55,  // how much of the height is spent rolling up

  nChord:         80,  // points around each section
  nInner:          8,  // spanwise stations, root to crank
  nOuter:         16,  // crank to tip
  nWinglet:       18,  // and up the winglet
};

/** A plain swept trapezoid: no crank, no winglet, symmetric sections. */
const TAIL = {
  ...WING,
  area:         32.8, aspectRatio: 5.0, rootChord: null, taperRatio: 0.30,
  sweep:        32.0, dihedral: 5.0,
  twistTip:      0.0, kink: null, winglet: false,
  root: '0010', tip: '0010',
  nInner:          2, nOuter: 20,
};

/**
 * Solve the chords so that the reference area comes out.
 *
 * With a straight leading edge the planform is still two trapezoids, so
 * S = (b/2)[(c_root + c_k) eta + (c_k + c_t)(1 - eta)] and, with the tip taper
 * measured off the crank, c_t = L c_k. Everything else follows:
 *
 *     c_k = (2S/b - c_root eta) / (1 + L (1 - eta))
 *
 * If the root chord is not given it is the root that gets derived instead, from
 * the plain trapezoid relation. Either way exactly one chord is solved for and
 * the area is exact.
 */
function chords({ area, span, rootChord, taperRatio, kink }) {
  // Without a crank there are only two chords and the taper ties them, so the
  // area fixes both. A root chord supplied here would over-determine the
  // planform -- three numbers, two degrees of freedom -- and the area would
  // silently come out as something else, which is how a tail ends up 5% too
  // big. So it is derived, and `rootChordDerived` says so rather than the input
  // being quietly dropped.
  if (kink == null) {
    const cRoot = 2 * area / (span * (1 + taperRatio));
    return { cRoot, cKink: null, cTip: cRoot * taperRatio, derived: 'root' };
  }
  if (rootChord == null) {
    // No opinion about the root: run the inboard panel at constant chord and
    // let the crank carry the area. Two panels, one of them untapered:
    //   S = (b/2) c_k [2 eta + (1 + L)(1 - eta)]
    const cKink = 2 * area / (span * (2 * kink + (1 + taperRatio) * (1 - kink)));
    return { cRoot: cKink, cKink, cTip: cKink * taperRatio, derived: 'root' };
  }
  const cKink = (2 * area / span - rootChord * kink) / (1 + taperRatio * (1 - kink));
  return { cRoot: rootChord, cKink, cTip: cKink * taperRatio, derived: 'crank' };
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
  const span = Math.sqrt(p.aspectRatio * p.area);
  const semi = span / 2;
  const { cRoot, cKink, cTip, derived } = chords({ ...p, span });

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
  // leading-edge point, a chord, a twist and a roll.
  const half = [];
  const push = (x, y, zLE, chord, twist, roll, foil, wing) =>
    half.push({ x, y, zLE, chord, twist, roll, foil, wing });

  const etas = [0];
  if (p.kink != null) {
    for (let i = 1; i <= p.nInner; i++) etas.push(p.kink * i / p.nInner);
    for (let i = 1; i <= p.nOuter; i++) etas.push(p.kink + (1 - p.kink) * i / p.nOuter);
  } else {
    for (let i = 1; i <= p.nInner + p.nOuter; i++) etas.push(i / (p.nInner + p.nOuter));
  }
  for (const t of etas) {
    push(t * semi, t * semi * Math.tan(p.dihedral * DEG), leAt(t),
         chordAt(t), twistAt(t), 0, foilAt(t), true);
  }

  /**
   * The winglet, carried on from the tip.
   *
   * Purely visual, and built as a continuation rather than as a separate part:
   * the frames keep coming, the roll winds up from zero to the cant angle over
   * the first `wingletBlend` of the height, and the loft never knows anything
   * happened. That is what makes it blend -- there is no joint to blend.
   *
   * The span direction and the section roll are wound up separately, because
   * they do not start together. Sections on a wing stand streamwise whatever
   * the dihedral is, so the section roll starts at zero while the span
   * direction starts at the dihedral angle. Winding one variable would put a
   * kink in the surface at the tip or leave the winglet leaning.
   */
  if (p.winglet && p.wingletHeight > 0) {
    const tip = half[half.length - 1];
    const H = p.wingletHeight * semi;
    const cant = p.wingletCant * DEG, dihedral = p.dihedral * DEG;
    const blend = Math.min(0.999, Math.max(0.02, p.wingletBlend));
    let x = tip.x, y = tip.y;
    for (let i = 1; i <= p.nWinglet; i++) {
      const s0 = (i - 1) / p.nWinglet, s1 = i / p.nWinglet, ds = s1 - s0;
      const w = (s) => smooth(Math.min(1, s / blend));
      // Step along the mid-point direction: cheap midpoint integration, and
      // enough that the tip lands where the height says it should.
      const psi = dihedral + (cant - dihedral) * w((s0 + s1) / 2);
      x += H * ds * Math.cos(psi);
      y += H * ds * Math.sin(psi);
      const roll = cant * w(s1);
      const chord = cTip * (1 + (p.wingletTaper - 1) * s1);
      push(x, y, tip.zLE + (tip.chord - chord) * 0
             + H * s1 * Math.tan(p.wingletSweep * DEG),
           chord, p.twistTip, roll, { a: tipFoil, b: tipFoil, f: 0 }, false);
    }
  }

  const frames = mirror
    ? [...half.slice(1).map((f) => ({ ...f, x: -f.x, roll: -f.roll })).reverse(), ...half]
    : half;

  /* ---- loft ----------------------------------------------------------- */
  const M = rootFoil.length;
  const pos = [], idx = [];
  for (const fr of frames) {
    const eps = fr.twist * DEG, ce = Math.cos(eps), se = Math.sin(eps);
    // The section's own vertical, rolled. On a wing this is straight up; up the
    // winglet it tips over until the section stands almost on edge.
    const uy = Math.cos(fr.roll * DEG), ux = -Math.sin(fr.roll * DEG);
    for (let i = 0; i < M; i++) {
      const { a, b, f } = fr.foil;
      const q0 = a[i][0] + (b[i][0] - a[i][0]) * f;
      const v0 = a[i][1] + (b[i][1] - a[i][1]) * f;
      const dq = q0 - p.twistAxis;
      const q = p.twistAxis + dq * ce + v0 * se;
      const v = (-dq * se + v0 * ce) * fr.chord;
      pos.push(fr.x + ux * v, fr.y + uy * v, -(fr.zLE + q * fr.chord));
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
  // rather than restating the request. The winglet is excluded: reference area
  // and aspect ratio are properties of the WING, and a winglet that quietly
  // added itself to them would make every number here disagree with the
  // three-view it came from.
  const panel = half.filter((f) => f.wing);
  let S = 0, macNum = 0, yNum = 0, xNum = 0;
  for (let i = 0; i < panel.length - 1; i++) {
    const a = panel[i], b = panel[i + 1], dy = b.x - a.x;
    S += (a.chord + b.chord) / 2 * dy;
    macNum += (a.chord ** 2 + b.chord ** 2) / 2 * dy;
    yNum += (a.x * a.chord + b.x * b.chord) / 2 * dy;
    xNum += (a.zLE * a.chord + b.zLE * b.chord) / 2 * dy;
  }
  const area = mirror ? 2 * S : S;
  const mac = macNum / S;
  const wingletTop = p.winglet ? half[half.length - 1] : null;

  Object.assign(g.userData, {
    span, semiSpan: semi, area, referenceArea: p.area, mirror,
    aspectRatio: span * span / p.area,
    taperRatio: cTip / (cKink ?? cRoot),
    rootChord: cRoot, kinkChord: cKink, tipChord: cTip,
    /** Which chord was solved for so the area came out. The other two are inputs. */
    derivedChord: derived,
    sweep: p.sweep, dihedral: p.dihedral,
    mac, yMac: yNum / S, xMacLE: xNum / S, xMacQuarter: xNum / S + 0.25 * mac,
    rootThickness: thicknessOf(rootFoil), tipThickness: thicknessOf(tipFoil),
    sections: { root: rootFoil.name, kink: kinkFoil?.name ?? null, tip: tipFoil.name },
    /** Winglet as built: how far up and out it reaches, and its own chord. */
    winglet: wingletTop && {
      height: wingletTop.y - panel[panel.length - 1].y,
      outboard: wingletTop.x - panel[panel.length - 1].x,
      tipChord: wingletTop.chord, cant: p.wingletCant,
    },
    frames: half, planform: p, skinMesh: mesh,
    /** Chord, leading edge and twist at any fraction of semispan. */
    at: (t) => ({ chord: chordAt(t), xLE: leAt(t),
                  y: t * semi * Math.tan(p.dihedral * DEG), twist: twistAt(t) }),
  });
  return g;
}

/** A wing: cranked, cambered, washed out, with a winglet. */
export function wing(o = {}) { return liftingSurface({ ...WING, ...o }); }

/** A horizontal tail: the same loft with no crank and symmetric sections. */
export function horizontalTail(o = {}) { return liftingSurface({ ...TAIL, ...o }); }

// No fin here yet on purpose. A vertical tail's span is a height, its area is
// one surface not two, and its aspect ratio is defined on those -- so it is a
// different set of conventions rather than a wing with mirror off, and
// pretending otherwise would put three numbers in userData that quietly mean
// something else. `liftingSurface({ mirror: false })` builds the geometry.
export const surfaces = { wing, horizontalTail };
