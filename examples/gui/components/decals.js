/**
 * Artwork applied to a lifting surface.
 *
 * A tail logo is not a texture on the fin. It is a separate patch of skin
 * lying ON the fin, a few millimetres proud, and that distinction is what makes
 * it work with everything else here:
 *
 *   - the paint scheme still drives the fin underneath, so artwork with an
 *     alpha channel composites over navy or crimson or bare alloy without
 *     anything being rebaked;
 *   - the fin keeps its own material, so the wireframe and translucency
 *     toggles reach it as before;
 *   - the artwork can be swapped, moved or removed without touching the loft.
 *
 * The patch conforms to the surface rather than floating flat near it. Its
 * nodes are read from the loft grid the fin was actually built from, so on a
 * 10% section the artwork follows the crown of the aerofoil instead of sinking
 * through it near the leading edge.
 *
 * It is laid out as a TRUE RECTANGLE in the fin's own plane -- upright, square
 * to the waterline -- and then projected onto the surface. Laying it out in
 * (chord fraction, span fraction) instead would be less code and wrong: a fin
 * is swept and tapered, so a rectangle in those coordinates is a trapezoid in
 * space, and the logo would come out leaning back and wider at the bottom. An
 * airline paints artwork upright.
 */
import * as THREE from 'three';
import { all as MATERIALS } from './materials.js';

/**
 * The two chordwise chains of a loft ring, each running leading edge to
 * trailing edge.
 *
 * Index `nChord - 1` is the leading edge and index 0 the trailing edge, and
 * BOTH chains end on that same index 0 -- the section is closed at the trailing
 * edge, so there is one point there, not two. Getting this wrong puts a
 * half-texel seam down the back of every decal that reaches the trailing edge.
 */
function chains(loft) {
  const { nChord, m } = loft;
  const a = [], b = [];
  for (let j = nChord - 1; j >= 0; j--) a.push(j);          // LE -> TE, first face
  b.push(nChord - 1);
  for (let j = nChord; j < m; j++) b.push(j);               // LE -> TE, second face
  b.push(0);
  return [a, b];
}

/** Vertex `j` of the frame at grid row `i`, in the loft's own frame. */
function node(loft, i, j, out = new THREE.Vector3()) {
  const k = 3 * (i * loft.m + j);
  return out.set(loft.positions[k], loft.positions[k + 1], loft.positions[k + 2]);
}

/**
 * A ring of `m` points at an arbitrary span fraction, interpolated between the
 * two frames that bracket it.
 *
 * Linear between frames is not an approximation of the surface -- it IS the
 * surface, because that is exactly what the triangles spanning those two frames
 * describe. A spline through the frames would be smoother and would leave the
 * artwork off the skin.
 */
function ringAt(loft, eta) {
  const e = loft.etas;
  let i = 0;
  while (i < e.length - 2 && e[i + 1] < eta) i++;
  const span = e[i + 1] - e[i];
  const f = span > 1e-12 ? Math.min(1, Math.max(0, (eta - e[i]) / span)) : 0;
  const ring = [];
  const p = new THREE.Vector3(), q = new THREE.Vector3();
  for (let j = 0; j < loft.m; j++) {
    node(loft, i, j, p); node(loft, i + 1, j, q);
    ring.push(p.clone().lerp(q, f));
  }
  return ring;
}

/**
 * The point on one face of the section at a given fraction of the local chord.
 *
 * The chord fraction is measured from the ring's own leading and trailing edge
 * points, not from the planform's numbers, so twist and thickness are already
 * accounted for and nothing has to agree with anything.
 */
function facepoint(ring, chain, u) {
  // Local z runs NEGATIVE aft, so distance aft of the leading edge is a
  // decreasing z. Normalising by the ring's own extent makes this independent
  // of where the section sits.
  const zLE = ring[chain[0]].z, zTE = ring[chain[chain.length - 1]].z;
  const denom = zLE - zTE;
  if (Math.abs(denom) < 1e-12) return ring[chain[0]].clone();
  const frac = (k) => (zLE - ring[chain[k]].z) / denom;
  let k = 0;
  while (k < chain.length - 2 && frac(k + 1) < u) k++;
  const f0 = frac(k), f1 = frac(k + 1);
  const t = f1 - f0 > 1e-12 ? (u - f0) / (f1 - f0) : 0;
  return ring[chain[k]].clone().lerp(ring[chain[k + 1]], t);
}

/**
 * Artwork on both faces of a lifting surface.
 *
 * Placement, all in the surface's own terms so that resizing the fin carries
 * the logo with it rather than leaving it behind:
 *
 *   height     fraction of the fin's height at the artwork's CENTRE
 *   chord      fraction of the local chord at that height, for its centre
 *   size       the artwork's height, as a fraction of the fin's height
 *   aspect     the image's width over its height
 *
 * `aspect` is not optional and is not guessed from the texture, because a
 * texture may not have loaded when this runs. A caller that has the image knows
 * its proportions; this would only find out later and silently build a square.
 *
 * Returns a group of two meshes with the fit recorded in `userData`, so that a
 * logo running off the trailing edge is a number a check can read rather than
 * something to notice in a screenshot.
 */
export function surfaceArt(surface, {
  texture,
  height = 0.62,
  chord = 0.55,
  size = 0.34,
  aspect = 1,
  offset = 0.006,
  fit = true,
  margin = 0.04,
  nu = 28,
  nv = 20,
  name = 'art',
} = {}) {
  const u = surface.userData;
  const loft = u.loft;
  if (!loft) throw new Error('surfaceArt: the surface exposes no loft grid');
  const [faceA, faceB] = chains(loft);

  // The fin's height is its semispan; a mirrored wing's is half its span. Both
  // are `semiSpan`, which is why that is what this reads.
  const semi = u.semiSpan;
  const hMid = Math.min(1, Math.max(0, height));
  const xMid = hMid * semi;

  /**
   * How far a given size runs past the leading or trailing edge, in metres.
   *
   * Checked at every row rather than at the centre, because the binding row is
   * the TOP one on a tapered fin -- the chord there is shortest -- and a logo
   * sized on the centre chord alone would hang off the back at the top while
   * looking correct where it was measured.
   */
  const overrunOf = (sz, cf) => {
    const halfH = (sz * semi) / 2;
    const w = 2 * halfH * aspect;
    const mid = u.at(hMid);
    const sMid = mid.xLE + cf * mid.chord;
    let worst = 0;
    for (let iv = 0; iv < nv; iv++) {
      const x = (xMid - halfH) + 2 * halfH * (iv / (nv - 1));
      const eta = Math.min(1, Math.max(0, x / semi));
      const g = u.at(eta);
      worst = Math.max(worst, (g.xLE) - (sMid - w / 2),
                              (sMid + w / 2) - (g.xLE + g.chord));
    }
    return worst;
  };

  /** The largest size that lands on the fin at a given fore-aft anchor. */
  const fittedSize = (cf) => {
    if (!fit || overrunOf(size, cf) <= 0) return size;
    let lo = 0, hi = size;
    for (let i = 0; i < 40; i++) {
      const m = (lo + hi) / 2;
      if (overrunOf(m, cf) > 0) hi = m; else lo = m;
    }
    return lo * (1 - margin);
  };

  /**
   * Where fore and aft the artwork sits.
   *
   * `'auto'` puts it wherever it can be LARGEST, which is not the middle of the
   * chord and is not the same place for every mark. The fin's leading edge is
   * swept a third of a turn back, so the usable box narrows towards the top on
   * the front side; a tall monogram wants to sit aft of centre to clear it,
   * while a wordmark six times wider than it is tall is limited by the chord at
   * its own row and wants the middle. Solving for it means the catalogue does
   * not carry six hand-tuned numbers that quietly go wrong the next time the
   * solve changes the tail's sweep or taper.
   */
  let anchor = chord, used;
  if (chord === 'auto') {
    anchor = 0.5; used = 0;
    for (let i = 0; i <= 90; i++) {
      const cf = 0.25 + (0.50 * i) / 90;
      const sz = fittedSize(cf);
      if (sz > used) { used = sz; anchor = cf; }
    }
  } else {
    used = fittedSize(anchor);
  }
  const shrunk = used < size - 1e-12;

  const half = (used * semi) / 2;
  const x0 = xMid - half, x1 = xMid + half;

  // Width comes from the image's proportions, in METRES, so the artwork is
  // never stretched. Where it sits is fixed at the centre height and then held
  // there for every row -- that is what keeps the rectangle upright on a swept
  // fin instead of shearing with the leading edge.
  const w = 2 * half * aspect;
  const mid = u.at(hMid);
  const sMid = mid.xLE + anchor * mid.chord;     // aft of the surface origin
  const s0 = sMid - w / 2, s1 = sMid + w / 2;

  const group = new THREE.Group();
  group.name = name;
  let worstU = 0;                                 // how far off the chord we ran

  // Which way round the artwork reads is decided in WORLD terms, not in the
  // loft's. A fin's group carries a quarter turn, so its local +y face comes
  // out on the port side, and taking the local frame at face value puts the
  // logo on backwards -- on both faces, since each gets the other's answer.
  //
  // The rule, once the side is known: the aeroplane's nose is +z, and a viewer
  // standing off the starboard side sees +z to their LEFT. So on a
  // starboard-facing surface the image's left edge belongs at the FORWARD end,
  // and on a port-facing one at the aft end.
  surface.updateWorldMatrix(true, false);
  const toWorld = (v) => v.clone().transformDirection(surface.matrixWorld);

  for (const [side, chain] of [[1, faceA], [-1, faceB]]) {
    // The outward direction of this face at the artwork's centre, kept in BOTH
    // frames because the two questions it answers live in different ones.
    // Which side of the aeroplane the face is on is a world question; which way
    // its triangles wind is a local one, since that is the frame the positions
    // are in. Using the world vector for both is what culled the port face.
    const midRing = ringAt(loft, hMid);
    const outward = facepoint(midRing, chain, anchor)
      .sub(facepoint(midRing, side > 0 ? faceB : faceA, anchor));
    const faceStarboard = toWorld(outward).x > 0;
    const pos = [], uv = [], idx = [];
    for (let iv = 0; iv < nv; iv++) {
      const x = x0 + (x1 - x0) * (iv / (nv - 1));
      const eta = Math.min(1, Math.max(0, x / semi));
      const ring = ringAt(loft, eta);
      const geom = u.at(eta);
      for (let iu = 0; iu < nu; iu++) {
        const fu = iu / (nu - 1);
        const s = s0 + (s1 - s0) * fu;
        const cu = (s - geom.xLE) / geom.chord;
        worstU = Math.max(worstU, cu > 1 ? cu - 1 : cu < 0 ? -cu : 0);
        const c = Math.min(1, Math.max(0, cu));
        const p = facepoint(ring, chain, c);
        // Outward is the direction from the OTHER face to this one. On a
        // section that is the surface normal by construction, with a sign that
        // cannot come out backwards -- which a cross product of two grid
        // tangents can, and does, the moment the grid winds the other way.
        const other = facepoint(ring, side > 0 ? faceB : faceA, c);
        const n = p.clone().sub(other);
        const len = n.length();
        if (len > 1e-9) p.addScaledVector(n, offset / len);
        pos.push(p.x, p.y, p.z);
        // `fu` runs forward to aft. On the starboard face that is left to right
        // as the image is read; on the port face it is right to left.
        uv.push(faceStarboard ? fu : 1 - fu, iv / (nv - 1));
      }
    }
    for (let iv = 0; iv < nv - 1; iv++) {
      for (let iu = 0; iu < nu - 1; iu++) {
        const a = iv * nu + iu, b = a + 1, c = a + nu, d = c + 1;
        if (side > 0) idx.push(a, c, b, b, c, d);
        else idx.push(a, b, c, b, d, c);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    // A fin patch is nearly flat, so the face's outward at its centre stands
    // for the whole of it. Derived here too, though this one was already right
    // -- an assumption that happens to hold is still an assumption.
    windOutward(g, () => outward);
    const mesh = new THREE.Mesh(g, artMaterial(texture));
    // Named for the side it actually ends up on, which is not the sign of the
    // local face -- that was the whole mistake this now derives its way out of.
    mesh.name = `${name}${faceStarboard ? 'Starboard' : 'Port'}`;
    group.add(mesh);
  }

  Object.assign(group.userData, {
    isArt: true, height: hMid, chord: anchor, requestedChord: chord, aspect,
    /** What was asked for, and what it had to become to land on the fin. */
    size: used, requestedSize: size, shrunk,
    /** Metres, on the surface, so a caller can report the size it got. */
    widthMetres: w, heightMetres: 2 * half,
    /**
     * How far the artwork ran past the leading or trailing edge, as a fraction
     * of the local chord. Zero means it fits. Anything above zero means nodes
     * were clamped onto the edge and the logo is squashed against it.
     */
    overrun: worstU,
    fits: worstU < 1e-9,
  });
  return group;
}

/** Alias kept readable at the call site: a fin is what this is usually for. */
export const finArt = surfaceArt;

/**
 * The artwork on hand, with where each piece wants to sit.
 *
 * `aspect` is the image file's own width over its height, recorded here rather
 * than measured at load: the geometry is built before the texture arrives, and
 * a decal that guessed square and corrected later would resize itself visibly
 * one frame after appearing.
 *
 * Fore and aft is solved rather than chosen, and the size each asks for is an
 * upper bound that the fit reduces to whatever the tail actually allows -- so
 * the only per-mark number here is the one the image file itself decides, its
 * aspect ratio.
 *
 * `height` is 0.40 rather than the geometric half. A fin tapers, so the mark
 * that fits at mid height is smaller than the one that fits below it, and
 * artwork centred on the geometry reads as sitting high -- the eye centres on
 * the visible panel, which is bottom-heavy, not on the span.
 */
export const TAIL_HEIGHT = 0.40;

export const TAIL_ART = {
  'none':          null,
  'beach black':  { file: 'beach-black.png', aspect: 929 / 149,   height: TAIL_HEIGHT, chord: 'auto', size: 0.9 },
  'beach gold':   { file: 'beach-gold.png',  aspect: 929 / 149,   height: TAIL_HEIGHT, chord: 'auto', size: 0.9 },
  'LB gold':      { file: 'lb-gold.png',     aspect: 1181 / 1272, height: TAIL_HEIGHT, chord: 'auto', size: 0.9 },
  'LB black':     { file: 'lb-black.png',    aspect: 1181 / 1272, height: TAIL_HEIGHT, chord: 'auto', size: 0.9 },
  'mitsubishi':   { file: 'mhi.png',         aspect: 156 / 109,   height: TAIL_HEIGHT, chord: 'auto', size: 0.9 },
  'MIT':          { file: 'mit.png',         aspect: 1400 / 724,  height: TAIL_HEIGHT, chord: 'auto', size: 0.9 },
};

export const artNames = Object.keys(TAIL_ART);

/**
 * The titles along the body, in order from the nose back.
 *
 * All at one height, which is how an airline sets a row of marks: cap heights
 * are matched and the widths fall out of each logo's own proportions, so the
 * wordmark ends up long and the roundel small rather than all three being
 * boxed to the same width and each coming out a different visual weight.
 */
export const FUSELAGE_TITLES = [
  { file: 'mhi.png',        aspect: 156 / 109 },
  { file: 'mit.png',        aspect: 1400 / 724 },
  { file: 'beach-gold.png', aspect: 929 / 149 },
];

/**
 * Where that row sits.
 *
 * `startNoseLengths` is measured in NOSE LENGTHS rather than metres so the row
 * stays put relative to the door it is meant to sit behind: both are referred
 * to the same length, so a solve that changes the nose moves them together.
 *
 * `y` is a guess, and deliberately a stated one. The window line is at
 * `R sin 20deg` and a window is about 0.165 tall, so the glass reaches roughly
 * 0.72 on this body; 1.15 puts the bottom of the artwork clear above it with
 * room left below the crown.
 */
export const TITLE_LAYOUT = {
  /**
   * The row starts at the front of the WINDOW LINE, so it is referred to the
   * cabin rather than to the nose. Both move when the solve changes, but they
   * do not move together -- the shell can grow without the nose doing anything
   * -- and it is the windows the titles have to sit alongside.
   *
   * `startNoseLengths` is the fallback for a caller with no deck to hand.
   */
  startAtWindows: true,
  startNoseLengths: 2.0,
  /**
   * Clear of the glass by about twice a window's own corner radius. A 0.32 m
   * window centred at 20 degrees reaches y 0.78; at 1.25 the titles came down
   * to 0.89, which read as tight against them rather than as a band above.
   */
  y: 1.36,
  height: 0.9,
  gap: 1.2,
};

/**
 * The row of titles as one group, packed nose to tail.
 *
 * Each mark's forward edge is set from the previous one's aft edge, so the gaps
 * are even in METRES on the skin whatever the marks' proportions are. Spacing
 * them by centre instead would bunch the wide wordmark against its neighbour.
 */
export function fuselageTitles(fuselage, { textureFor, deck, titles = FUSELAGE_TITLES,
                                           layout = {}, name = 'titles' } = {}) {
  const L = { ...TITLE_LAYOUT, ...layout };
  const noseLength = fuselage.userData.noseLength;
  const group = new THREE.Group();
  group.name = name;
  // z decreases aft, so the row starts at the largest z and works back.
  const start = (L.startAtWindows && deck)
    ? windowLayout(deck).zFwd
    : -L.startNoseLengths * noseLength;
  let zFwd = start;
  const placed = [];
  for (const t of titles) {
    const w = L.height * t.aspect;
    const piece = fuselageArt(fuselage, {
      texture: textureFor ? textureFor(t.file) : null,
      z: zFwd - w / 2, y: L.y, height: L.height, aspect: t.aspect,
      name: t.file.replace(/\.png$/, ''),
    });
    group.add(piece);
    placed.push(piece.userData);
    zFwd -= w + L.gap;
  }
  Object.assign(group.userData, {
    isArt: true, pieces: placed, startZ: start,
    /** Where the row starts and ends, so a caller can say whether it fits. */
    zRange: [zFwd + L.gap, start],
    fits: placed.every((p) => p.fits),
  });
  return group;
}

/**
 * Turn a patch's triangles to face outward, whichever way its grid happened to
 * wind.
 *
 * Worked out from the geometry rather than reasoned about, because reasoning
 * about it is exactly what fails: the winding that faces outward depends on how
 * the two grid directions are ordered AND on which way each of them runs on the
 * surface, and the second face of a body reverses one of those. Get it wrong
 * and the artwork is not subtly off, it is invisible -- backface culling
 * removes it completely, which looks like the decal never being created at all.
 *
 * Voted over every triangle rather than decided on the first, so one sliver at
 * a corner cannot flip the whole patch.
 */
function windOutward(geo, outwardAt) {
  const pos = geo.getAttribute('position'), idx = geo.getIndex();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const ab = new THREE.Vector3(), ac = new THREE.Vector3(), n = new THREE.Vector3();
  let vote = 0;
  for (let t = 0; t < idx.count; t += 3) {
    a.fromBufferAttribute(pos, idx.getX(t));
    b.fromBufferAttribute(pos, idx.getX(t + 1));
    c.fromBufferAttribute(pos, idx.getX(t + 2));
    n.crossVectors(ab.subVectors(b, a), ac.subVectors(c, a));
    vote += n.dot(outwardAt(a.add(b).add(c).multiplyScalar(1 / 3))) > 0 ? 1 : -1;
  }
  if (vote >= 0) return false;
  const arr = idx.array;
  for (let t = 0; t < arr.length; t += 3) {
    const tmp = arr[t + 1]; arr[t + 1] = arr[t + 2]; arr[t + 2] = tmp;
  }
  idx.needsUpdate = true;
  geo.computeVertexNormals();
  return true;
}

/**
 * The angle at which a section passes through a given world height.
 *
 * Bisected on the upper half, where height rises monotonically with angle for
 * every section here. Solved rather than taken as `asin(y / r)` because that is
 * only true of a circle, and the D8's section is not one -- its height at an
 * angle depends on the section shape as well as the radius.
 */
function angleAtHeight(fuse, z, y) {
  let lo = -Math.PI / 2, hi = Math.PI / 2;
  const yAt = (t) => fuse.surfaceAt(z, t).y;
  if (y <= yAt(lo) || y >= yAt(hi)) return null;
  for (let i = 0; i < 48; i++) {
    const m = (lo + hi) / 2;
    if (yAt(m) < y) lo = m; else hi = m;
  }
  return (lo + hi) / 2;
}

/**
 * Step around a section by ARC LENGTH from a starting angle.
 *
 * Equal steps in ANGLE would be the obvious thing and would stretch the
 * artwork as it climbs towards the crown, because equal angles do not subtend
 * equal distances on a section that is not a circle -- and even on one, equal
 * steps in HEIGHT do not. A logo is applied to the developed skin, so the
 * spacing that has to be even is the one measured along it.
 *
 * This is the same lesson as the fuselage's own rings, which are resampled at
 * even arc length for exactly this reason.
 */
function angleAtArc(fuse, z, th0, arc, steps = 160) {
  if (Math.abs(arc) < 1e-12) return th0;
  const dir = Math.sign(arc), want = Math.abs(arc);
  let th = th0, acc = 0;
  let prev = fuse.surfaceAt(z, th);
  // A fixed angular step, refined by walking: the section's radius is bounded,
  // so a step of a degree is far finer than the artwork's own grid.
  const dth = (dir * Math.PI) / 720;
  for (let i = 0; i < steps * 4; i++) {
    const next = fuse.surfaceAt(z, th + dth);
    const d = Math.hypot(next.x - prev.x, next.y - prev.y);
    if (acc + d >= want) return th + dth * ((want - acc) / d);
    acc += d; th += dth; prev = next;
  }
  return th;
}

/**
 * Artwork on the side of a fuselage, both sides.
 *
 * Placed at a world HEIGHT rather than at an angle, for the same reason the
 * window row is: a cabin floor is level, so anything referenced to it has to
 * stay level too. Place by angle and the band climbs with the section centre
 * through the tailcone, arcing up the side of the aeroplane in a way no real
 * one does.
 *
 *   z        station of the artwork's CENTRE, negative aft
 *   y        world height of its centre
 *   height   its height on the skin, in metres of arc
 *   aspect   the image's width over its height
 */
export function fuselageArt(fuselage, {
  texture,
  z = -12,
  y = 1.15,
  height = 0.9,
  aspect = 1,
  offset = 0.008,
  nu = 34,
  nv = 18,
  name = 'art',
} = {}) {
  const fu = fuselage.userData;
  const w = height * aspect;
  // z decreases aft, so the FORWARD edge is the larger z.
  const zFwd = z + w / 2, zAft = z - w / 2;

  const group = new THREE.Group();
  group.name = name;
  let clipped = 0;

  for (const side of [1, -1]) {
    const pos = [], uv = [], idx = [];
    for (let iv = 0; iv < nv; iv++) {
      const fv = iv / (nv - 1);
      for (let iu = 0; iu < nu; iu++) {
        const fu2 = iu / (nu - 1);
        const zs = zFwd + (zAft - zFwd) * fu2;
        // The band's centre angle is found afresh at every station, which is
        // what keeps it level rather than parallel to the section.
        const thC = angleAtHeight(fu, zs, y);
        if (thC == null) clipped++;
        const base = thC ?? Math.PI / 2;
        const arc = (fv - 0.5) * height;
        const th = angleAtArc(fu, zs, base, arc);
        const p = fu.surfaceAt(zs, side > 0 ? th : Math.PI - th);
        const n = fu.normalAt(zs, side > 0 ? th : Math.PI - th);
        pos.push(p.x + n.x * offset, p.y + n.y * offset, p.z + n.z * offset);
        // Starboard is +x. A viewer there sees the nose (+z) to their LEFT, so
        // the image's left edge belongs at the forward end -- which is fu2 = 0.
        // To port it is the other way round. The same rule as the fin, stated
        // once more here because the surface, not the rule, is what differs.
        uv.push(side > 0 ? fu2 : 1 - fu2, fv);
      }
    }
    for (let iv = 0; iv < nv - 1; iv++) {
      for (let iu = 0; iu < nu - 1; iu++) {
        const a = iv * nu + iu, b = a + 1, c = a + nu, d = c + 1;
        if (side > 0) idx.push(a, c, b, b, c, d);
        else idx.push(a, b, c, b, d, c);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    // Turned to face out from the body's own normal, not from an assumed
    // ordering of the grid -- getting this wrong culled the whole row.
    windOutward(g, (p) => {
      const s = fu.shapeAt(p.z);
      return fu.normalAt(p.z, Math.atan2(p.y - s.yc, p.x));
    });
    const mesh = new THREE.Mesh(g, artMaterial(texture));
    mesh.name = `${name}${side > 0 ? 'Starboard' : 'Port'}`;
    group.add(mesh);
  }

  Object.assign(group.userData, {
    isArt: true, z, y, aspect,
    widthMetres: w, heightMetres: height,
    zRange: [zAft, zFwd],
    /** Stations where the band ran off the top of the body. Should be zero. */
    clipped, fits: clipped === 0,
  });
  return group;
}

/**
 * The cabin windows.
 *
 * Where they go is not a styling choice: the solve says how many rows there
 * are and where the pressure shell starts and stops, and a window belongs at
 * every row inside it. So the row count, the extent and the pitch are all read
 * from the deck, and the only numbers here are the ones about the window
 * itself -- how big it is and how high up the side it sits.
 *
 * One window per SEAT ROW, at the pitch the solve implies. A real aeroplane
 * carries windows at the FRAME pitch instead, which is finer than the seat
 * pitch and gives roughly five windows for every three rows -- but frame
 * spacing is not something this solve knows, and inventing a number for it
 * would put windows where nothing in the model asked for them. Pass a `pitch`
 * to override if the denser row is wanted.
 */
export const WINDOW = {
  /** Fore-aft and vertical size on the skin, matching textures/window.png. */
  width: 0.23,
  height: 0.32,
  /** Degrees above the section centre. Windows sit a fixed height above a
   *  level floor, so this is converted to a height once and held there. */
  angle: 20,
  lift: 0.004,
};

/**
 * Where the row of windows lands, as numbers, without building anything.
 *
 * Split out because two things need it and only one of them makes windows: the
 * titles are asked to start at the front of the WINDOW LINE, and computing
 * that separately in each place is how the two quietly drift apart.
 */
export function windowLayout(deck, { pitch, width = WINDOW.width } = {}) {
  const zStart = -deck.cabinStart, zEnd = -deck.cabinEnd;
  const step = pitch ?? deck.seatPitch;
  const rows = pitch == null
    ? Math.round(deck.cabinRows)
    : Math.max(0, Math.floor(Math.abs(zEnd - zStart) / step + 1e-9));
  const used = rows * step;
  const z0 = zStart - (Math.abs(zEnd - zStart) - used) / 2 - step / 2;
  return {
    rows, step, z0,
    /** Forward and aft EDGES of the glass, not the centres. */
    zFwd: z0 + width / 2,
    zAft: z0 - (rows - 1) * step - width / 2,
  };
}

export function cabinWindows(fuselage, {
  texture, deck, pitch, name = 'windows',
  width = WINDOW.width, height = WINDOW.height, angle = WINDOW.angle,
  lift = WINDOW.lift,
} = {}) {
  if (!deck) throw new Error('cabinWindows: needs the deck for the cabin extent');
  const group = new THREE.Group();
  group.name = name;

  // The row COUNT is read, not recomputed. Dividing the shell by the pitch
  // gives 29.999999999999996 on this solve -- the pitch was derived from those
  // same two numbers -- and flooring that quietly drops the last row.
  const { rows, step, z0 } = windowLayout(deck, { pitch, width });

  // Held at a fixed HEIGHT rather than a fixed angle, for the same reason the
  // row of titles is: the floor is level, so a window line referenced to it
  // cannot climb with the section.
  const y = fuselage.userData.shapeAt(-deck.cabinStart).r
          * Math.sin(angle * Math.PI / 180);

  for (let i = 0; i < rows; i++) {
    const w = fuselageArt(fuselage, {
      texture, z: z0 - i * step, y, height, aspect: width / height,
      offset: lift, nu: 6, nv: 6, name: `window${i}`,
    });
    group.add(w);
  }

  Object.assign(group.userData, {
    isArt: true, rows, pitch: step, y, width, height,
    zRange: [z0 - (rows - 1) * step, z0],
    /** Declared so a page can say what it is showing without recomputing it. */
    passengers: deck.passengers, seatsAbreast: deck.seatsAbreast,
    perRow: 1,
  });
  return group;
}

/**
 * The windscreen: a band cut in SIDE VIEW and wrapped onto the nose.
 *
 * Two horizontal lines at fractions of the fuselage's height and a station a
 * fraction of the nose back, and everything on the nose between them is glass.
 * No artwork, no panes drawn into a texture -- the shape comes from where those
 * planes actually cut the body, which is why it looks like it belongs on the
 * nose rather than like a decal applied to one.
 *
 * The band changes topology along its length, and handling that is the whole
 * job. Aft, the crown stands above the upper line and the glass is two separate
 * strips down the sides. Forward of about a quarter of the nose the section has
 * shrunk enough that the crown drops BELOW the upper line, and the band closes
 * over the top into one wrap-around screen. Further forward still the crown
 * falls below the LOWER line and there is no glass at all, which is what tapers
 * the screen to a point at the front.
 *
 * Built as two mirrored patches whose upper edge is the lower of the upper line
 * and the crown. Where the crown binds, both patches end exactly on it and abut
 * with no seam; where the line binds, they are properly separate. That is the
 * whole topology change, handled by a `min` rather than by two cases -- and it
 * leaves no degenerate triangles, which clamping a fixed grid to the crown
 * would have produced all along the wrap.
 */
export const WINDSCREEN = {
  /** Fractions of the body's height, measured up from the keel. */
  low: 0.60,
  high: 0.80,
  /** How far back along the NOSE the glass runs. */
  backFraction: 0.50,
  lift: 0.004,
  colour: 0x12161c,
  /**
   * Proportions of the glazed length, NOSE FIRST, and the post between each
   * pair in metres. Three a side is what an airliner carries: the windscreen,
   * the sliding window, and the quarter light behind it.
   */
  panes: [0.42, 0.31, 0.27],
  post: 0.07,
};

export function windscreen(fuselage, {
  low = WINDSCREEN.low, high = WINDSCREEN.high,
  backFraction = WINDSCREEN.backFraction, lift = WINDSCREEN.lift,
  colour = WINDSCREEN.colour, panes = WINDSCREEN.panes, post = WINDSCREEN.post,
  nz = 64, nv = 14, name = 'windscreen',
} = {}) {
  const fu = fuselage.userData;
  const noseLength = fu.noseLength;

  // The two lines are fractions of the BARREL's height, taken from the body
  // itself rather than from twice a radius, so a section that is not a circle
  // still gives its own keel and crown.
  const zRef = fu.cabinZ[0] - 0.05 * (fu.cabinZ[0] - fu.cabinZ[1]);
  const keel = fu.keelAt(zRef), crown = fu.crownAt(zRef);
  const H = crown - keel;
  const yLo = keel + low * H, yHi = keel + high * H;

  const zBack = -backFraction * noseLength;
  /** The station forward of which the crown has dropped below a given line. */
  const crownCrosses = (target) => {
    let a = zBack, b = 0;
    for (let i = 0; i < 60; i++) {
      const m = (a + b) / 2;
      if (fu.crownAt(m) > target) a = m; else b = m;
    }
    return a;
  };
  // Forward end: where the crown drops to the lower line, so the glass runs out
  // on its own rather than at a station picked by hand.
  const zNose = crownCrosses(yLo);
  // And the CORNER: where the crown drops past the upper line and the two side
  // strips merge into one wrap. The boundary genuinely turns there -- it stops
  // following a level plane and starts following the crown ridge -- so a
  // station has to land exactly on it. Without one the mesh chords straight
  // across the turn and takes a wedge out of the glass beside the centreline.
  // Same rule as a wing putting a station on every planform break.
  // z runs NEGATIVE aft, so a corner between the two ends is GREATER than the
  // aft station and LESS than the forward one.
  const zCorner = crownCrosses(yHi);
  const hasCorner = zCorner > zBack + 1e-9 && zCorner < zNose - 1e-9;

  /** Angle of the upper edge: the upper line, or the crown where that is lower. */
  const upperAngle = (z) => (fu.surfaceAt(z, Math.PI / 2).y <= yHi
    ? Math.PI / 2
    : angleAtHeight(fu, z, yHi));

  /**
   * The panes, as z ranges, laid out from the NOSE back.
   *
   * `panes` are proportions of the glazed length and are given nose-first,
   * because that is the order they are named on an aeroplane -- the windscreen,
   * then the sliding window, then the quarter light. The posts between them are
   * a width in METRES rather than a proportion: a post is a piece of structure
   * a hand's breadth across whatever the aeroplane's size, and one expressed as
   * a fraction would thicken with the nose.
   */
  const glazed = Math.abs(zNose - zBack) - post * (panes.length - 1);
  const weight = panes.reduce((a, b) => a + b, 0);
  const ranges = [];
  {
    let z = zNose;                              // start at the nose, work aft
    for (const share of panes) {
      const len = (glazed * share) / weight;
      ranges.push([z - len, z]);                // [aft, forward], z negative aft
      z -= len + post;
    }
  }

  /**
   * One pane, as a patch between two stations.
   *
   * Stations are bunched at BOTH ends -- each pane's edges are where its shape
   * is cut off, and a distribution that refined only one would leave a facet at
   * the other -- and the corner is inserted if it falls inside this pane, so no
   * pane can chord across the turn.
   */
  const cosine = (t) => (1 - Math.cos(Math.PI * t)) / 2;
  const paneStations = (zAft, zFwd, n) => {
    const out = [];
    for (let i = 0; i < n; i++) out.push(zAft + (zFwd - zAft) * cosine(i / (n - 1)));
    if (hasCorner && zCorner > zAft + 1e-9 && zCorner < zFwd - 1e-9) {
      // The corner itself, and a geometric fan either side of it. The pane's
      // own ends are already refined by the cosine, but the turn can fall
      // anywhere inside a pane -- here it lands 22 mm from pane 1's aft edge --
      // and it is the sharpest feature the band has.
      out.push(zCorner);
      for (let d = (zFwd - zAft) * 0.05; d > 1e-4; d *= 0.4) {
        if (zCorner - d > zAft) out.push(zCorner - d);
        if (zCorner + d < zFwd) out.push(zCorner + d);
      }
      out.sort((a, b) => a - b);
    }
    return out;
  };

  const group = new THREE.Group();
  group.name = name;
  const material = glazingMaterial(colour);
  let wraps = 0;
  const built = [];

  for (const side of [1, -1]) {
    ranges.forEach(([zAft, zFwd], pane) => {
      // Stations shared out by LENGTH, not equally: the panes are different
      // sizes, and an equal share leaves the longest one coarsest.
      const share = Math.abs(zFwd - zAft) / glazed;
      const stations = paneStations(zAft, zFwd, Math.max(8, Math.round(nz * share)));
      const pos = [], idx = [];
      for (const z of stations) {
        const thLo = angleAtHeight(fu, z, yLo) ?? Math.PI / 2;
        const thHi = upperAngle(z) ?? Math.PI / 2;
        if (side > 0 && thHi >= Math.PI / 2 - 1e-9) wraps++;
        for (let iv = 0; iv < nv; iv++) {
          const th = thLo + (thHi - thLo) * (iv / (nv - 1));
          const t = side > 0 ? th : Math.PI - th;
          const p = fu.surfaceAt(z, t), n = fu.normalAt(z, t);
          pos.push(p.x + n.x * lift, p.y + n.y * lift, p.z + n.z * lift);
        }
      }
      for (let iz = 0; iz < stations.length - 1; iz++) {
        for (let iv = 0; iv < nv - 1; iv++) {
          const p0 = iz * nv + iv, p1 = p0 + 1, p2 = p0 + nv, p3 = p2 + 1;
          idx.push(p0, p2, p1, p1, p2, p3);
        }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      g.setIndex(idx);
      g.computeVertexNormals();
      windOutward(g, (q) => {
        const s = fu.shapeAt(q.z);
        return fu.normalAt(q.z, Math.atan2(q.y - s.yc, q.x));
      });
      const mesh = new THREE.Mesh(g, material);
      mesh.name = `${name}${side > 0 ? 'Starboard' : 'Port'}${pane + 1}`;
      mesh.userData = { pane: pane + 1, side, zRange: [zAft, zFwd], stations, nv };
      group.add(mesh);
      if (side > 0) built.push({ pane: pane + 1, zRange: [zAft, zFwd], stations: stations.length });
    });
  }

  Object.assign(group.userData, {
    isArt: true, low, high, yLow: yLo, yHigh: yHi, lift,
    bodyHeight: H, keel, crown,
    zRange: [zBack, zNose],
    /** Where the two side strips merge into one wrap, and whether there is one. */
    zCorner: hasCorner ? zCorner : null, hasCorner,
    /** The panes as built, nose-first, and the posts between them. */
    panes: built, post, paneCount: panes.length, nv,
    /** Stations at which the band closes over the crown rather than at yHigh. */
    wrapStations: wraps,
  });
  return group;
}

/** Dark glass. Its own material, because it carries no texture to key on. */
const glazings = new Map();
function glazingMaterial(colour) {
  if (glazings.has(colour)) return glazings.get(colour);
  const m = new THREE.MeshStandardMaterial({
    color: new THREE.Color(colour), roughness: 0.14, metalness: 0.10,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
  });
  m.name = `glazing ${colour.toString(16)}`;
  glazings.set(colour, m);
  MATERIALS[`glazing${glazings.size}`] = m;
  return m;
}

/**
 * The material for one piece of artwork.
 *
 * Biased toward the camera by the same amounts the fuselage decals use: a few
 * millimetres of standoff on a body tens of metres long is far below the depth
 * buffer's resolution there, and without the bias the artwork and the skin win
 * the depth test in alternating patches as the camera moves.
 *
 * `alphaTest` is small rather than a half: it rejects the texels that are fully
 * clear, so they stop writing depth, while leaving the soft edge of the logo
 * to blend. A large alphaTest would give crisp edges and a visible stair-step
 * on every curve of the artwork.
 */
const materials = new Map();
function artMaterial(texture) {
  if (materials.has(texture)) return materials.get(texture);
  const m = new THREE.MeshStandardMaterial({
    map: texture ?? null, transparent: true, alphaTest: 0.02,
    roughness: 0.44, metalness: 0.0,
    polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4,
  });
  m.name = `art ${texture?.name ?? materials.size}`;
  materials.set(texture, m);
  // Registered so the page's wireframe toggle reaches it, the same way the
  // fuselage's decal materials are.
  MATERIALS[`art${materials.size}`] = m;
  return m;
}
