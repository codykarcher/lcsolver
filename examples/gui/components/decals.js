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
  const overrunOf = (sz) => {
    const halfH = (sz * semi) / 2;
    const w = 2 * halfH * aspect;
    const mid = u.at(hMid);
    const sMid = mid.xLE + chord * mid.chord;
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

  // Auto-fit. A catalogue can then ask for a generous size and get the largest
  // one that actually lands on the fin, instead of every entry having to be
  // hand-tuned against whatever the current solve made the tail.
  let used = size, shrunk = false;
  if (fit && overrunOf(used) > 0) {
    let lo = 0, hi = used;
    for (let i = 0; i < 40; i++) {
      const midSz = (lo + hi) / 2;
      if (overrunOf(midSz) > 0) hi = midSz; else lo = midSz;
    }
    used = lo * (1 - margin);
    shrunk = true;
  }

  const half = (used * semi) / 2;
  const x0 = xMid - half, x1 = xMid + half;

  // Width comes from the image's proportions, in METRES, so the artwork is
  // never stretched. Where it sits is fixed at the centre height and then held
  // there for every row -- that is what keeps the rectangle upright on a swept
  // fin instead of shearing with the leading edge.
  const w = 2 * half * aspect;
  const mid = u.at(hMid);
  const sMid = mid.xLE + chord * mid.chord;      // aft of the surface origin
  const s0 = sMid - w / 2, s1 = sMid + w / 2;

  const group = new THREE.Group();
  group.name = name;
  let worstU = 0;                                 // how far off the chord we ran

  for (const [side, chain] of [[1, faceA], [-1, faceB]]) {
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
        // Mirrored on the second face. Both faces are seen from opposite sides,
        // so identical UVs would put the logo on backwards on one of them.
        uv.push(side > 0 ? fu : 1 - fu, iv / (nv - 1));
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
    const mesh = new THREE.Mesh(g, artMaterial(texture));
    mesh.name = `${name}${side > 0 ? 'Starboard' : 'Port'}`;
    group.add(mesh);
  }

  Object.assign(group.userData, {
    isArt: true, height: hMid, chord, aspect,
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
 * Placement differs per mark because their proportions do. A tall monogram
 * centres on the fin and can be large; a wordmark six times wider than it is
 * tall has to sit low, where the chord is longest, and is still limited by it.
 * Sizes here are what each mark WANTS -- `surfaceArt` shrinks any that will not
 * fit the tail the current solve produced, so these need no upkeep when the
 * fin changes.
 */
export const TAIL_ART = {
  'none':          null,
  'beach black':  { file: 'beach-black.png', aspect: 929 / 149,   height: 0.42, chord: 0.56, size: 0.30 },
  'beach gold':   { file: 'beach-gold.png',  aspect: 929 / 149,   height: 0.42, chord: 0.56, size: 0.30 },
  'LB gold':      { file: 'lb-gold.png',     aspect: 1181 / 1272, height: 0.60, chord: 0.60, size: 0.46 },
  'LB black':     { file: 'lb-black.png',    aspect: 1181 / 1272, height: 0.60, chord: 0.60, size: 0.46 },
  'mitsubishi':   { file: 'mhi.png',         aspect: 156 / 109,   height: 0.58, chord: 0.58, size: 0.42 },
  'MIT':          { file: 'mit.png',         aspect: 1400 / 724,  height: 0.55, chord: 0.58, size: 0.38 },
};

export const artNames = Object.keys(TAIL_ART);

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
    map: texture, transparent: true, alphaTest: 0.02,
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
