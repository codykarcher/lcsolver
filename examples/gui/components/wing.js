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
import { asSection, sectionPoints, thicknessOf } from './airfoil.js';

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
 *     not there;
 *   - `etaRoot` holds the chord CONSTANT from the centreline out to the side of
 *     the body, which is the wing box carrying through the fuselage. With a
 *     straight leading edge that puts the whole of it into the trailing edge,
 *     which is what a 737's inboard trailing edge is. It is worth having rather
 *     than tapering from the centreline: on the sizing decks this file is fed,
 *     ignoring it costs 2.6% of the reference area.
 */
const WING = {
  span:         34.1,  // tip to tip
  rootChord:     6.0,  // at the centreline, in the same units as the span
  crankRatio:   0.73,  // crank chord / ROOT chord
  tipRatio:     0.27,  // tip chord / CRANK chord
  taperRatio:   null,  // CRANKLESS surfaces only: tip chord / root chord
  sweep:        27.0,  // degrees, at the LEADING EDGE
  // Locked. Still parameters -- a different aircraft can pass its own -- but
  // not things the deck sets or the page exposes.
  dihedral:      3.0,  // degrees
  twistRoot:     0.0,  // degrees, positive leading edge up
  twistKink:    null,  // null interpolates between root and tip
  twistTip:     -3.0,  // negative is washout
  twistAxis:    0.25,  // chord fraction the sections are twisted about
  etaRoot:      0.00,  // constant-chord carry-through out to here, of semispan
  kink:         0.35,  // fraction of semispan, or null for a plain trapezoid
  root: '2412', kinkFoil: null, tip: '2410',   // kinkFoil null means blended
  symmetric:   false,  // force zero camber, whatever section was named
  // Thickness at the three stations, as a fraction of the local chord. Null
  // takes whatever the named aerofoil already is. Held apart from the aerofoil
  // because it is a structural number as much as an aerodynamic one -- the
  // spar depth is set here -- and because a wing routinely runs one shape at
  // several thicknesses.
  rootThickness: 0.13,
  crankThickness: 0.11,
  tipThickness:  0.10,
  thickness:    null,  // one t/c for the whole surface; overrides the three


  nChord:         80,  // points around each section
  nInner:          8,  // spanwise stations, root to crank
  nOuter:         16,  // crank to tip
};

/** A plain swept trapezoid: no crank, symmetric sections. */
const TAIL = {
  ...WING,
  // A tail has no crank, so it takes the plain taper ratio -- tip over root --
  // and one thickness for the whole surface. Neither is a special case in the
  // loft; they are the crankless spellings of the same two ideas, and having
  // them named plainly is worth more than reusing a chained ratio that happens
  // to reduce to the same thing.
  span:         12.8, rootChord: 3.94,
  taperRatio:   0.30, thickness: 0.10,
  sweep:        32.0, dihedral: 5.0,
  twistTip:      0.0, kink: null,
  // Symmetric by convention. A tail that has to work both ways up has no
  // business being cambered, so the section is NACA 00xx and the thickness
  // above sets the xx -- which is exact, because the 4-digit thickness
  // distribution scales linearly with t.
  symmetric:    true,
  root: '0010', tip: '0010',
  nInner:          2, nOuter: 20,
};

/**
 * The three chords, from one length and two ratios. Nothing is solved for.
 *
 * Without a crank, `taperRatio` is the plain tip-over-root and is what a tail
 * uses. `tipRatio` still works there as a fallback, measured off the root,
 * since with no crank the two mean the same thing.
 */
function chords({ rootChord, crankRatio, tipRatio, taperRatio, kink }) {
  if (kink == null) {
    return { cRoot: rootChord, cKink: null,
             cTip: rootChord * (taperRatio ?? tipRatio) };
  }
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

/** Piecewise-linear through a list of [station, value], stations ascending. */
function through(stops, t) {
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, va] = stops[i], [b, vb] = stops[i + 1];
    if (t <= b) return b - a < 1e-12 ? vb : va + (vb - va) * (t - a) / (b - a);
  }
  return stops[stops.length - 1][1];
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

  // `symmetric` zeroes the camber of whatever was named rather than swapping in
  // a different aerofoil, so the thickness distribution is preserved and the
  // convention cannot be defeated by naming a cambered section.
  const flat = (f) => (p.symmetric
    ? { ...f, camber: f.camber.map(() => 0), camberMax: 0,
        name: `NACA 00${String(Math.round(thicknessOf(f) * 100)).padStart(2, '0')}` }
    : f);
  const rootFoil = flat(asSection(p.root, p.nChord));
  const tipFoil = flat(asSection(p.tip, p.nChord));
  const kinkFoil = p.kinkFoil == null ? null : flat(asSection(p.kinkFoil, p.nChord));
  // Thickness runs as its own spanwise distribution, independent of which
  // aerofoil is where. So a t/c at the crank does not require a crank aerofoil,
  // and changing an aerofoil does not silently change the thickness.
  // A single `thickness` beats the three, which is how a tail is described: one
  // number for the whole surface rather than a distribution.
  const tRoot = p.thickness ?? p.rootThickness ?? thicknessOf(rootFoil);
  const tTip = p.thickness ?? p.tipThickness ?? thicknessOf(tipFoil);
  const tKink = p.thickness ?? p.crankThickness
    ?? (kinkFoil ? thicknessOf(kinkFoil) : null);
  const thickAt = (t) => alongSpan(t, p.kink, tRoot, tKink, tTip);

  // Chord by station rather than by the root/crank/tip rule, because the carry-
  // through adds a fourth: constant to etaRoot, then the panels.
  const chordStops = [[0, cRoot], [Math.max(0, p.etaRoot ?? 0), cRoot]];
  if (p.kink != null) chordStops.push([p.kink, cKink]);
  chordStops.push([1, cTip]);
  const chordAt = (t) => through(chordStops, t);
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
  const push = (x, y, zLE, chord, twist, foil, thickness) =>
    half.push({ x, y, zLE, chord, twist, foil, thickness });

  // A station lands exactly on every planform break -- the side of body and the
  // crank -- so each stays a sharp corner instead of being rounded off over one
  // panel and put in the wrong place as well.
  const breaks = [0, ...(p.etaRoot > 0 ? [p.etaRoot] : []),
                  ...(p.kink != null ? [p.kink] : []), 1];
  const etas = [0];
  for (let b = 1; b < breaks.length; b++) {
    const from = breaks[b - 1], to = breaks[b];
    const n = b === breaks.length - 1 ? p.nOuter : Math.max(2, Math.round(p.nInner / 2));
    for (let i = 1; i <= n; i++) etas.push(from + (to - from) * i / n);
  }
  for (const t of etas) {
    push(t * semi, t * semi * Math.tan(p.dihedral * DEG), leAt(t),
         chordAt(t), twistAt(t), foilAt(t), thickAt(t));
  }

  const frames = mirror
    ? [...half.slice(1).map((f) => ({ ...f, x: -f.x })).reverse(), ...half]
    : half;

  /* ---- loft ----------------------------------------------------------- */
  const M = 2 * p.nChord - 2;
  const pos = [], idx = [];
  for (const fr of frames) {
    // Blend camber and half-thickness term for term -- the two sections share a
    // chordwise distribution, so this is exact -- then take the result to the
    // wanted t/c and only then build surface points. Blending POINTS instead
    // would mix camber into thickness and vice versa.
    const { a, b, f } = fr.foil;
    const blended = {
      x: a.x, n: a.n,
      camber: a.camber.map((c, i) => c + (b.camber[i] - c) * f),
      half: a.half.map((h, i) => h + (b.half[i] - h) * f),
    };
    const own = 2 * Math.max(...blended.half);
    const pts = sectionPoints(blended, fr.thickness == null ? 1 : fr.thickness / own);

    const eps = fr.twist * DEG, ce = Math.cos(eps), se = Math.sin(eps);
    for (let i = 0; i < M; i++) {
      const dq = pts[i][0] - p.twistAxis, v0 = pts[i][1];
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
    // Declared on every surface, not only on fins. A field that exists only
    // sometimes is a field every caller has to guard, and the guard is easy to
    // get subtly wrong -- the same lesson as the fuselage's vessel factor.
    // Every fin-specific field belongs here, not just the first one I thought
    // of: adding cant meant adding three more, and check_pages caught all three.
    isFin: false, height: null, cant: 0, tipRise: null, tipOffset: null,
    /** Derived, both of them: b^2 / S on the full reference area. */
    aspectRatio: span * span / refArea,
    rootChord: cRoot, kinkChord: cKink, tipChord: cTip,
    /**
     * The conventional taper ratio, tip over ROOT -- derived, and not one of the
     * inputs. On a cranked wing it describes a straight trapezoid that does not
     * exist, which is why it is not what the wing is built from.
     */
    taperRatio: cTip / cRoot, etaRoot: p.etaRoot ?? 0,
    crankRatio: cKink ? cKink / cRoot : null, tipRatio: cTip / (cKink ?? cRoot),
    constantThickness: p.thickness ?? null,
    sweep: p.sweep, dihedral: p.dihedral,
    mac, yMac: yNum / S, xMacLE: xNum / S, xMacQuarter: xNum / S + 0.25 * mac,
    /** Thickness as built, at the three stations. */
    rootThickness: tRoot, crankThickness: tKink, tipThickness: tTip,
    thicknessAt: thickAt,
    symmetric: p.symmetric,
    sections: p.symmetric
      // Named from the thickness actually built, not from whatever base section
      // the camber was stripped off.
      ? (() => {
          const nm = (t) => `NACA 00${String(Math.round(t * 100)).padStart(2, '0')}`;
          return { root: nm(tRoot), kink: tKink == null ? null : nm(tKink), tip: nm(tTip) };
        })()
      : { root: rootFoil.name, kink: kinkFoil?.name ?? null, tip: tipFoil.name },
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

/** A fin: one surface, standing up, symmetric. */
const FIN = {
  ...TAIL,
  height:        6.2,  // ROOT to tip, not a span -- there is only one of it
  rootChord:     5.0,
  taperRatio:   0.32,
  sweep:        42.0,
  thickness:    0.10,
  cant:          0.0,  // degrees from VERTICAL; positive leans the tip to +X
  dihedral:      0.0, twistRoot: 0.0, twistTip: 0.0,
  kink:         null, symmetric: true,
  nInner:          2, nOuter: 20,
};

/**
 * A vertical tail.
 *
 * Held apart from the other two because its CONVENTIONS differ, not because the
 * loft does. A fin's span is a HEIGHT measured root to tip; its area is ONE
 * surface, not two; and its aspect ratio is height squared over that area.
 * Mirror the same shape and read it as a wing -- span 2h, area 2S -- and the
 * aspect ratio comes out at 2h^2/S, TWICE the fin's 1.879 against 3.758 on the
 * default. Building it as `liftingSurface({ mirror: false })` and reading the
 * wing's userData off it would put three numbers there that quietly mean
 * something else, and an aspect ratio out by a factor of two is exactly the
 * sort of error that survives a review, because it still looks like a number.
 *
 * So the loft runs unchanged and the things that differ are handled here: the
 * span passed down is twice the height, so the half-surface the loft builds
 * runs root to tip, and the group is turned about the axis so that span runs UP
 * rather than outboard.
 *
 * `cant` tilts it off vertical, positive leaning the tip to starboard, and it
 * is simply a smaller turn -- pi/2 minus the cant. It is deliberately NOT a
 * dihedral: dihedral moves the sections and leaves them streamwise, which is
 * right for a wing and wrong here, because a canted fin's sections lean over
 * with it. Turning the whole surface is what actually happens to a canted fin
 * and needs nothing in the loft.
 *
 * Area, aspect ratio and MAC are properties of the surface and do not change
 * with cant. What does change is how much of the height is HEIGHT: a fin canted
 * 30 degrees reaches only cos(30) of its span above the root, and both numbers
 * are reported because confusing them is easy and quiet.
 *
 * The geometry accessors in userData -- `at`, `frames` -- are in the loft's own
 * frame, before that quarter turn. The mesh carries the rotation.
 */
export function verticalTail({ height, ...o } = {}) {
  const p = { ...FIN, ...o };
  // Written back, not just used. `height` is destructured out of the argument
  // list, so without this the planform records the DEFAULT while the surface is
  // built to the override -- and anything reading planform.height afterwards
  // gets a number that was never used.
  const h = height ?? p.height;
  p.height = h;
  const g = liftingSurface({ ...p, span: 2 * h, mirror: false });
  const cant = (p.cant ?? 0) * DEG;
  g.rotation.z = Math.PI / 2 - cant;

  const u = g.userData;
  const area = u.area;                       // one surface: mirror was off
  Object.assign(u, {
    isFin: true, height: h, area, referenceArea: area,
    /** h^2 / S on ONE surface -- a fin's definition, not a wing's. */
    aspectRatio: h * h / area,
    /** Height above the root at which the MAC sits, along the span. */
    heightMac: u.yMac,
    cant: p.cant ?? 0,
    /** Where the tip ends up: canting spends span on lateral reach. */
    tipRise: h * Math.cos(cant),
    tipOffset: h * Math.sin(cant),
  });
  return g;
}

/**
 * The three default sets, so a caller can start from any of them without
 * restating them. A page that hard-codes its own copy is a second source of
 * truth for the same numbers, which is the failure check_defaults exists to
 * catch.
 */
export const defaults = { wing: WING, tail: TAIL, fin: FIN };

export const surfaces = { wing, horizontalTail, verticalTail };
