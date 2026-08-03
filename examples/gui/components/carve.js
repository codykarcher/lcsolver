/**
 * Cutting a duct out of a finished body.
 *
 * This exists because of one hard limit in `fuselage.js`: a section there is a
 * single radius per angle about the body's own centre, and a duct is not
 * expressible that way. A duct has a wall with void on both sides, so a ray out
 * of the section crosses skin, opening, then skin again -- two spans of
 * material on one ray -- and one radius cannot say that. Every way of asking
 * the section for it was tried and measured: a trough over a deepened floor
 * left 54 rays that needed two spans, a straight prismatic U left 168 to 294
 * from every candidate origin, and shortening the walls to 141 mm still left
 * 46. Only walls of no height at all -- a flat lid, not a duct -- came out
 * clean.
 *
 * But the limit belongs to the SECTION, not to geometry. A mesh has no such
 * constraint: nothing ever asks a triangle for its radius. So the body is built
 * exactly as it was, and the duct is taken out of the mesh afterwards. Outside
 * the cut not one parameter, station, or vertex moves -- which is the whole
 * point, because that shape was hard-won and this must not disturb it.
 *
 * Two surfaces, two cuts, one clipper. The skin is cut on the duct's surface;
 * the duct's floor is cut on the body's. They meet on the curve where the two
 * surfaces cross, which both cuts converge to from their own side.
 */
import * as THREE from 'three';

/**
 * Where a segment crosses a surface.
 *
 * Interpolating the field linearly gets the crossing only if the field is
 * linear along the segment, and none of these are. The duct's corner is a
 * circular arc that turns through vertical, so the field's gradient changes by
 * an order of magnitude across one skin triangle, and the linear guess lands
 * well off the surface -- 34 mm off at the worst, which is a 34 mm crack
 * between the skin's cut edge and the duct's own rim. It does not improve with
 * a finer sheet, because the sheet was never the problem.
 *
 * So the guess is only a bracket, and the crossing is then solved for.
 */
function crossingT(a, b, f, da) {
  let lo = 0, hi = 1, flo = da;
  for (let i = 0; i < 40; i++) {
    const m = (lo + hi) / 2;
    const fm = f(a.clone().lerp(b, m));
    if ((fm > 0) === (flo > 0)) { lo = m; flo = fm; } else hi = m;
  }
  return (lo + hi) / 2;
}

/**
 * A vertex, carrying its normal as well as its position.
 *
 * The normal has to travel with it. Clipping produces an unindexed mesh -- the
 * skin's 8,961 shared vertices become 50,286 unshared ones -- and recomputing
 * normals on that gives one per FACE, so a body that was smooth comes back
 * visibly faceted over its whole length. Nothing about the shape changed; only
 * the shading did, which is worse, because it looks like the carve wrecked the
 * mesh when the mesh is exactly right.
 *
 * Interpolated at the crossing by the same parameter as the position, so a cut
 * edge shades continuously with the surface it was cut from.
 */
const vert = (p, n) => ({ p, n });
const lerpVert = (a, b, t) => vert(
  a.p.clone().lerp(b.p, t),
  a.n.clone().lerp(b.n, t).normalize());

/** Read a mesh's triangles as vertices carrying their normals. */
function readTriangles(geometry) {
  const pos = geometry.getAttribute('position');
  const nrm = geometry.getAttribute('normal');
  const index = geometry.getIndex();
  const count = index ? index.count : pos.count;
  const at = index ? (i) => index.getX(i) : (i) => i;
  const tris = [];
  for (let t = 0; t + 2 < count; t += 3) {
    const tri = [];
    for (let k = 0; k < 3; k++) {
      const i = at(t + k);
      tri.push(vert(
        new THREE.Vector3().fromBufferAttribute(pos, i),
        nrm ? new THREE.Vector3().fromBufferAttribute(nrm, i) : new THREE.Vector3()));
    }
    tris.push(tri);
  }
  return { tris, hasNormals: !!nrm };
}

/** Build a geometry from triangles of vertices, keeping their normals. */
function fromTriangles(tris, hasNormals) {
  const pos = [], nrm = [];
  for (const t of tris) {
    for (const v of t) {
      pos.push(v.p.x, v.p.y, v.p.z);
      nrm.push(v.n.x, v.n.y, v.n.z);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  if (hasNormals) g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  else g.computeVertexNormals();
  return g;
}

/**
 * Cut one triangle on one surface: the part where `field` is negative, and the
 * part where it is positive.
 *
 * A triangle wholly on one side passes through vertex for vertex -- no
 * resampling, no drift. One that straddles is split at the field's own zero, so
 * the edge follows the cutting surface rather than whichever triangle happened
 * to lie across it.
 *
 * `field` has to be signed and continuous, not a predicate: a predicate can
 * only put the boundary on whichever vertex happened to test true, which leaves
 * an edge as ragged as the mesh is coarse.
 */
function cutTriangle(tri, field) {
  const d = tri.map((v) => field(v.p));
  const up = d.filter((x) => x > 0).length;
  if (up === 3) return { neg: [], pos: [tri] };
  if (up === 0) return { neg: [tri], pos: [] };
  // Rotate so the odd vertex out is first, which makes both cases one shape.
  const o = up === 1 ? d.findIndex((x) => x > 0) : d.findIndex((x) => x <= 0);
  const p = [tri[o], tri[(o + 1) % 3], tri[(o + 2) % 3]];
  const q = [d[o], d[(o + 1) % 3], d[(o + 2) % 3]];
  const m1 = lerpVert(p[0], p[1], crossingT(p[0].p, p[1].p, field, q[0]));
  const m2 = lerpVert(p[0], p[2], crossingT(p[0].p, p[2].p, field, q[0]));
  const corner = [[p[0], m1, m2]];                 // the odd vertex out
  const quad = [[m1, p[1], p[2]], [m1, p[2], m2]];
  return up === 1 ? { neg: quad, pos: corner } : { neg: corner, pos: quad };
}

/** Keep only the part of a mesh where `field` is positive. */
export function clipTriangles(geometry, field) {
  const { tris, hasNormals } = readTriangles(geometry);
  const keep = [];
  let cut = 0;
  for (const tri of tris) {
    const r = cutTriangle(tri, field);
    if (r.neg.length && r.pos.length) cut++;
    keep.push(...r.pos);
  }
  const g = fromTriangles(keep, hasNormals);
  g.userData.clip = { kept: keep.length, cut };
  return g;
}

/**
 * Remove from a mesh everything inside a region, where the region is the
 * intersection of several smooth surfaces' interiors.
 *
 * Given as a LIST of surfaces rather than as one field, and this is not a
 * convenience. Combining them with `min` first and clipping on that once puts
 * the cut in the wrong place: `min` has a crease wherever two surfaces trade
 * places, and a triangle straddling that crease has no linear crossing to find,
 * so the interpolated point lands well inside the region. Measured on the
 * corner where the duct's floor meets its side wall, the cut edge came out
 * 114 mm inboard of the wall it was supposed to be on, leaving a real hole.
 *
 * Splitting one surface at a time avoids it: each cut is against a single
 * smooth field, so each crossing is where it should be, and the pieces that
 * survive early cuts are set aside rather than tested again. The creases then
 * appear on their own, as the seams between pieces.
 */
export function carveOut(geometry, fields) {
  const { tris, hasNormals } = readTriangles(geometry);
  const out = [];
  let whole = 0, split = 0, dropped = 0;

  for (const tri of tris) {
    const d = fields.map((f) => tri.map((v) => f(v.p)));

    // Clear of the region entirely -- every vertex outside the same surface --
    // so it passes through UNTOUCHED. Not merely unchanged in shape: the same
    // three vertices, in the same order, unsplit. Splitting a triangle far
    // from the duct is invisible but it is still a change to the body, and
    // the whole premise here is that nothing outside the cut moves.
    if (d.some((row) => row.every((x) => x <= 0))) { whole++; out.push(tri); continue; }
    if (d.every((row) => row.every((x) => x > 0))) { dropped++; continue; }

    // On the boundary: cut against one surface at a time, setting aside what
    // each cut puts outside, and carrying only what is still in question.
    split++;
    let rest = [tri];
    for (const f of fields) {
      const next = [];
      for (const piece of rest) {
        const r = cutTriangle(piece, f);
        out.push(...r.neg);                        // outside this surface: kept
        next.push(...r.pos);                       // inside it: still in question
      }
      rest = next;
      if (!rest.length) break;
    }
  }

  const g = fromTriangles(out, hasNormals);
  g.userData.clip = { kept: out.length, whole, split, dropped };
  return g;
}

/**
 * The duct, as a solid.
 *
 * A flat floor with its corners rounding up at the radius of what it holds --
 * so the walls cannot be the wrong curve for the engines, because they ARE the
 * engines' curve -- running from the trailing edge, where it is deepest, up to
 * the crown at the end of the cabin, where it has no depth left and the body is
 * untouched. Everything above that floor and between those walls is void.
 *
 * Given as a signed depth rather than as a surface. That is all the clipping
 * needs, and it frees the duct from having to be star-shaped about anything,
 * which is the entire reason this works where the section could not.
 */
export function ductVolume({ floor, halfWidth, cornerR, fromX, toX, deepFrom, crown }) {
  const flat = Math.max(halfWidth - cornerR, 0);
  const deep = deepFrom ?? toX;
  /** The floor's height at a lateral offset: flat across, then rounding up. */
  const lift = (x) => {
    const ax = Math.abs(x);
    if (ax <= flat) return 0;
    const dx = Math.min(ax - flat, cornerR);
    return cornerR - Math.sqrt(Math.max(0, cornerR * cornerR - dx * dx));
  };
  /**
   * How far the floor has dropped, at a station.
   *
   * Level from `deepFrom` aft -- the engines lie along there and the floor has
   * to be clear beneath the whole of them, not just under their tail. Forward
   * of that it climbs to the crown at the cabin, and then keeps climbing, out
   * through the roof and clear of the body, which is what closes the duct at
   * the front.
   *
   * The climb does NOT ease in. A smoothstep leaves the roof tangentially,
   * which sounds better and is much worse: the body's own crown falls away at
   * a slope of 0.1 there, so a floor leaving flat runs within millimetres of
   * the skin for a third of a metre, and the two surfaces cross at a grazing
   * angle over that whole stretch instead of at a curve. The carve came out as
   * a hairline slot right across the crown that nothing could close -- 227 mm
   * from the nearest edge of anything. Leaving at a slope of 0.55 instead, five
   * times the roof's own, the surfaces cross cleanly and the opening starts at
   * a point and widens.
   *
   * Cubic, with that slope at the start and level at the end, so the join into
   * the deep part -- where smoothness is actually visible -- is still smooth.
   */
  const span = Math.max(deep - fromX, 1e-9);
  const drop = floor - crown;
  const k = 1;                          // initial slope, in units of drop/span
  const floorAt = (x) => {
    const f = (x - fromX) / span;
    if (f >= 1) return floor;
    if (f <= 0) return crown + drop * k * f;      // climbing out through the roof
    return crown + drop * ((k - 2) * f ** 3 + (3 - 2 * k) * f ** 2 + k * f);
  };
  return {
    fromX, toX, deepFrom: deep, floor, halfWidth, cornerR, crown, flat, lift, floorAt,
    /**
     * The three surfaces that bound the duct, each on its own and each smooth.
     * Kept separate because anything cutting geometry has to cut on one at a
     * time -- see `carveOut` for what combining them first does to the edge.
     */
    faces: [
      (p) => p.y - (floorAt(-p.z) + lift(p.x)),   // above the floor
      (p) => halfWidth - Math.abs(p.x),           // inboard of the walls
    ],
    /** Inside both at once. For asking about a point, not for cutting. */
    depth(p) {
      return Math.min(
        p.y - (floorAt(-p.z) + lift(p.x)),
        halfWidth - Math.abs(p.x),
      );
    },
  };
}

/**
 * The duct's own surface, over just the part of it that is inside the body.
 *
 * Without this the body is a shell with a hole in it, and you look through the
 * opening straight at the back of the far wall. With it the duct reads as a
 * duct: a floor sweeping down out of the cabin roof and two walls rounding up
 * to hold the engines.
 *
 * Built on the volume's own definition and then clipped on the body's, so it
 * ends exactly where the skin's cut began -- both cuts run to the curve where
 * the two surfaces meet.
 */
export function ductSurface(duct, depthInside, { nx = 160, nu = 72, wallTop } = {}) {
  const pos = [], idx = [];
  /**
   * One station's profile: down one wall, round the corner, across the floor,
   * and back up the other.
   *
   * The WALLS are here as well as the floor, which they were not at first, and
   * that is the whole difference between a closed shell and a leaking one. The
   * duct is bounded laterally as well as from below, so where the body runs
   * wider than the duct there is a vertical strip of material between the top
   * of the corner round and the skin. Meshing only the floor left that strip
   * with no face on it: 50 of 238 cut edges had nothing to close them, the
   * worst 450 mm from anything.
   *
   * Sampled by ARC LENGTH along the profile rather than by segment, so the
   * quads stay near-square instead of bunching at the corners.
   */
  const top = wallTop ?? (duct.crown + 0.1);
  const profile = (x) => {
    const y0 = duct.floorAt(x);
    const wallH = Math.max(top - (y0 + duct.cornerR), 0);
    const arc = (Math.PI / 2) * duct.cornerR;
    const half = wallH + arc + duct.flat;          // one side, floor centre out
    const at = (s) => {                            // s in [-half, half]
      const side = s < 0 ? -1 : 1, a = Math.abs(s);
      if (a <= duct.flat) return [side * a, y0];
      if (a <= duct.flat + arc) {
        const th = (a - duct.flat) / duct.cornerR;
        return [side * (duct.flat + duct.cornerR * Math.sin(th)),
                y0 + duct.cornerR * (1 - Math.cos(th))];
      }
      return [side * duct.halfWidth, y0 + duct.cornerR + (a - duct.flat - arc)];
    };
    return { at, half };
  };
  // Starts FORWARD of the cabin, where the floor is already climbing out
  // through the roof. Nothing there survives the clip, but the sheet has to
  // reach past the point where the two surfaces cross or the cut edge runs off
  // the end of it.
  const x0 = duct.fromX - 0.5;
  /** A point on the sheet, from station and normalised position around it. */
  const point = (x, t) => {
    const { at, half } = profile(x);
    const [px, py] = at(half * (2 * t - 1));
    return new THREE.Vector3(px, py, -x);
  };
  /**
   * The sheet's normal, from the surface itself rather than from its triangles.
   *
   * Face normals would face the duct exactly as well and shade it as a set of
   * flat panels, which is the same fault the skin had: the arc across the
   * corners would read as a row of chamfers instead of a round.
   */
  const nrm = [];
  const normalAt = (x, t) => {
    const h = 1e-4;
    const ds = point(x, Math.min(1, t + h)).sub(point(x, Math.max(0, t - h)));
    const dx = point(x + h, t).sub(point(x - h, t));
    const n = new THREE.Vector3().crossVectors(ds, dx).normalize();
    // Into the void: up off the floor, inboard off the walls.
    const p = point(x, t);
    if (n.dot(new THREE.Vector3(-Math.sign(p.x) * 0.3, 1, 0)) < 0) n.negate();
    return n;
  };
  for (let i = 0; i <= nx; i++) {
    const x = x0 + (duct.toX - x0) * (i / nx);
    const { at, half } = profile(x);
    for (let j = 0; j <= nu; j++) {
      const t = j / nu;
      const [px, py] = at(half * (2 * t - 1));
      pos.push(px, py, -x);
      const n = normalAt(x, t);
      nrm.push(n.x, n.y, n.z);
    }
  }
  const row = nu + 1;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nu; j++) {
      const a = i * row + j, b = a + 1, c = a + row, e = c + 1;
      idx.push(a, c, b, b, c, e);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  g.setIndex(idx);
  // Keep only what lies within the body. Elsewhere the floor has nothing to be
  // the floor OF, and would read as a shelf hanging in the open air.
  const clipped = clipTriangles(g, (p) => depthInside(p.x, p.y, p.z));
  return faceUp(clipped);
}

/**
 * Wind a sheet so its visible side faces up, into the duct.
 *
 * Votes area-weighted face normals against +y and flips the lot if they lose,
 * for the same reason the fin art needed it: backface culling makes a wrongly
 * wound patch INVISIBLE rather than wrong-looking, so a silent flip reads as
 * "the carve did nothing" and sends you looking in the wrong place.
 *
 * Winding only. The sheet's normals are its own, taken from the surface, and
 * recomputing them from the faces here would undo that and facet it.
 */
function faceUp(geo) {
  const pos = geo.getAttribute('position');
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = new THREE.Vector3();
  let vote = 0;
  for (let i = 0; i + 2 < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i);
    b.fromBufferAttribute(pos, i + 1);
    c.fromBufferAttribute(pos, i + 2);
    n.crossVectors(b.sub(a), c.sub(a));
    vote += n.y;
  }
  if (vote < 0) {
    const arr = pos.array;
    for (let i = 0; i + 8 < arr.length; i += 9) {
      for (let k = 0; k < 3; k++) {
        const t = arr[i + 3 + k]; arr[i + 3 + k] = arr[i + 6 + k]; arr[i + 6 + k] = t;
      }
    }
    pos.needsUpdate = true;
  }
  return geo;
}

/**
 * Cut the duct out of a built fuselage, in place.
 *
 * Replaces the skin's geometry with the clipped one and adds the duct's floor
 * beside it, both as children of the same group, so everything mounted to the
 * body -- decals, windows, the fins, the engines -- is untouched and still sits
 * where it did.
 */
/** The edges of a mesh used by only one triangle -- where it is open. */
export function openEdges(geometry) {
  const pos = geometry.getAttribute('position');
  const index = geometry.getIndex();
  const n = index ? index.count : pos.count;
  const at = index ? (i) => index.getX(i) : (i) => i;
  const v = new THREE.Vector3();
  const key = (i) => {
    v.fromBufferAttribute(pos, at(i));
    return `${v.x.toFixed(5)},${v.y.toFixed(5)},${v.z.toFixed(5)}`;
  };
  const uses = new Map(), ends = new Map();
  for (let t = 0; t + 2 < n; t += 3) {
    const k = [key(t), key(t + 1), key(t + 2)];
    for (let e = 0; e < 3; e++) {
      const a = k[e], b = k[(e + 1) % 3];
      const id = a < b ? `${a}/${b}` : `${b}/${a}`;
      uses.set(id, (uses.get(id) || 0) + 1);
      if (!ends.has(id)) {
        ends.set(id, [new THREE.Vector3(...a.split(',').map(Number)),
                      new THREE.Vector3(...b.split(',').map(Number))]);
      }
    }
  }
  return [...uses].filter(([, c]) => c === 1).map(([id]) => ends.get(id));
}

/**
 * Pull a sheet's rim onto another mesh's open edge.
 *
 * The two sheets are cut on different surfaces and meshed at different
 * densities, so each approximates the curve where those surfaces cross with its
 * own polyline -- and between shared endpoints one runs as a chord where the
 * other follows the curve. The lens between them is a real crack, up to 39 mm
 * here, and it does not close by refining either sheet: the coarse one is the
 * skin, and the skin is the thing that must not be touched.
 *
 * So the sheet, which may be moved, is brought to the skin, which may not.
 * Only its rim moves, and only where there is something within `tol` to move
 * it to, which leaves the sheet's genuinely free edges alone.
 */
function snapRim(geometry, edges, tol = 0.15) {
  const pos = geometry.getAttribute('position');
  const rim = new Map();
  for (const [a, b] of openEdges(geometry)) {
    for (const q of [a, b]) rim.set(`${q.x.toFixed(5)},${q.y.toFixed(5)},${q.z.toFixed(5)}`, true);
  }
  const p = new THREE.Vector3(), ab = new THREE.Vector3(), c = new THREE.Vector3();
  let moved = 0, worst = 0;
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    if (!rim.has(`${p.x.toFixed(5)},${p.y.toFixed(5)},${p.z.toFixed(5)}`)) continue;
    let best = null, bestD = tol;
    for (const [a, b] of edges) {
      ab.subVectors(b, a);
      const l2 = ab.lengthSq();
      const t = l2 < 1e-18 ? 0
        : Math.min(1, Math.max(0, c.subVectors(p, a).dot(ab) / l2));
      c.copy(a).addScaledVector(ab, t);
      const dd = c.distanceTo(p);
      if (dd < bestD) { bestD = dd; best = c.clone(); }
    }
    if (best) { pos.setXYZ(i, best.x, best.y, best.z); moved++; worst = Math.max(worst, bestD); }
  }
  pos.needsUpdate = true;
  return { moved, worst };
}

/**
 * Cut a region out of every mesh under an object, wherever it sits.
 *
 * The fins are built in their own frame -- hung on the body's corners and
 * canted outboard -- so the duct, which is described in the aeroplane's frame,
 * has to be asked about in theirs. Each mesh's own transform does that, and the
 * geometry is then cut where it stands, with no need for anything to agree
 * about coordinates beforehand.
 *
 * `base` is where `root`'s parent sits in that frame -- identity when the root
 * hangs directly off the aeroplane. Accumulated down the tree from local
 * matrices rather than read off `matrixWorld`, because this runs while the
 * aeroplane is still being assembled and nothing has been through a render yet,
 * so the world matrices are whatever they were left at.
 */
export function carveInto(root, fields, base = new THREE.Matrix4()) {
  const out = [];
  const walk = (o, parent) => {
    o.updateMatrix();
    const m = new THREE.Matrix4().multiplyMatrices(parent, o.matrix);
    if (o.isMesh && o.geometry?.getAttribute('position')) {
      const q = new THREE.Vector3();
      const local = fields.map((f) => (p) => f(q.copy(p).applyMatrix4(m)));
      const before = o.geometry;
      o.geometry = carveOut(before, local);
      out.push(o.geometry.userData.clip);
      before.dispose();
    }
    for (const c of o.children) walk(c, m);
  };
  walk(root, base);
  return out;
}

export function carveDuct(fuse, spec) {
  const duct = ductVolume(spec);
  const skin = fuse.userData.skinMesh;
  const before = skin.geometry;
  const kept = carveOut(before, duct.faces);
  skin.geometry = kept;
  before.dispose();

  const sheet = ductSurface(duct, fuse.userData.depthInside);
  const snap = snapRim(sheet, openEdges(kept));
  const floor = new THREE.Mesh(sheet, skin.material);
  floor.name = 'ductFloor';
  fuse.add(floor);

  fuse.userData.duct = {
    ...spec, mesh: floor, snap,
    skin: kept.userData.clip,
    floorTriangles: sheet.getAttribute('position').count / 3,
  };
  return duct;
}
