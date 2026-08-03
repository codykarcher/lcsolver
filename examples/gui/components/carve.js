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
    /**
     * Asked one surface at a time, and stopped as soon as one settles it.
     *
     * Every field here is expensive -- the duct's faces flare against the body,
     * so all three ask the section where the skin is -- and evaluating all of
     * them on all three vertices before looking at any was most of what the fin
     * carve cost: nine section queries on every triangle, when the first three
     * already showed the triangle was nowhere near. Order matters for the same
     * reason, which is why the caller puts the field that rules out the most
     * first.
     */
    const d = [];
    let clear = false;
    for (const f of fields) {
      const row = [f(tri[0].p), f(tri[1].p), f(tri[2].p)];
      d.push(row);
      if (row[0] <= 0 && row[1] <= 0 && row[2] <= 0) { clear = true; break; }
    }
    if (clear) { whole++; out.push(tri); continue; }

    // Clear of the region entirely -- handled above -- passes through
    // UNTOUCHED. Not merely unchanged in shape: the same three vertices, in the
    // same order, unsplit. Splitting a triangle far from the duct is invisible
    // but it is still a change to the body, and the whole premise here is that
    // nothing outside the cut moves.
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
export function ductVolume({
  floor, halfWidth, cornerR, fromX, toX, deepFrom, crown,
  blend = 0, depthInside = null,
}) {
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
  /**
   * The lip, rounded rather than square.
   *
   * Left alone the duct meets the skin at a hard crease -- a right angle along
   * the median of the rim, and folded back on itself at 158 degrees at the
   * worst. This flares the duct outward as it nears the skin, by an amount that
   * depends only on how deep into the body a point is. The rim is exactly where
   * that depth is zero, everywhere it runs, so one profile rounds the whole lip
   * -- walls, forward closure, trailing edge -- without ever having to find the
   * rim or follow it.
   *
   * The flare has to be in the FACES, not in the sheet alone. A fillet moves
   * the line where the two surfaces meet: it is tangent to the skin a distance
   * out from the old corner, not at it. Scooping only the sheet and leaving the
   * cut where it was cannot be tangent to anything -- tried, and it turned the
   * median crease from 90 to 48 degrees by undercutting the wall, which softens
   * the shading but leaves an overhang rather than a round.
   *
   * The arc is truncated at 70 degrees instead of running to tangency. A fillet
   * that meets the skin flat meets it at a grazing angle, and trimming one
   * surface against another along a graze is exactly what left a hairline slot
   * across the crown 227 mm from anything. Stopping at 70 leaves the surfaces
   * crossing at 20 degrees -- shallow, but a crossing.
   */
  const LIP = 70 * Math.PI / 180;
  const aKnee = blend * (1 - Math.sin(LIP));      // where the arc is truncated
  const eKnee = blend * (1 - Math.cos(LIP));
  const eSkin = eKnee + Math.tan(LIP) * aKnee;    // the flare at the skin itself
  const flare = (a) => {
    if (!(blend > 0)) return 0;
    if (a >= blend) return 0;
    if (a <= aKnee) return eKnee + Math.tan(LIP) * (aKnee - Math.max(a, 0));
    const w = blend - a;
    return blend - Math.sqrt(Math.max(0, blend * blend - w * w));
  };
  const lip = depthInside ? (p) => flare(depthInside(p.x, p.y, p.z)) : () => 0;

  return {
    fromX, toX, deepFrom: deep, floor, halfWidth, cornerR, crown, blend, flat,
    lift, floorAt, flare, lip, eSkin,
    /**
     * The two surfaces that bound the duct, each on its own and each smooth.
     * Kept separate because anything cutting geometry has to cut on one at a
     * time -- see `carveOut` for what combining them first does to the edge.
     */
    faces: [
      (p) => p.y - (floorAt(-p.z) + lift(p.x)) + lip(p),   // above the floor
      (p) => halfWidth + lip(p) - Math.abs(p.x),           // inboard of the walls
    ],
    /** Inside both at once. For asking about a point, not for cutting. */
    depth(p) {
      return Math.min(
        p.y - (floorAt(-p.z) + lift(p.x)) + lip(p),
        halfWidth + lip(p) - Math.abs(p.x),
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
export function ductSurface(duct, depthInside, { edge, nx, nu, wallTop } = {}) {
  const pos = [], nrm = [], idx = [];
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
  /**
   * How high this station's walls are meshed: just past where they leave the
   * body, found by asking the body.
   *
   * A single height for the whole duct -- the cabin's crown, which is what this
   * used -- wastes almost all of the samples. The afterbody is 0.9 m lower at
   * the engines than at the cabin, so at those stations two thirds of every
   * wall was meshed in open air and thrown away by the clip, leaving three rows
   * across the part that is actually there. The lip's round then fell inside a
   * single row and did not appear at all: the sheet stepped 180 mm sideways in
   * one quad and read exactly as square as before.
   */
  const wallTopAt = (x) => {
    if (wallTop != null) return wallTop;
    if (!depthInside) return duct.crown + 0.1;
    const margin = (duct.eSkin ?? 0) + 0.10;
    const y0 = duct.floorAt(x) + duct.cornerR;
    let lo = y0, hi = duct.crown + 0.2;
    if (!(depthInside(duct.halfWidth, lo, -x) > 0)) return y0 + margin;
    for (let i = 0; i < 32; i++) {
      const m = (lo + hi) / 2;
      if (depthInside(duct.halfWidth, m, -x) > 0) lo = m; else hi = m;
    }
    return lo + margin;
  };
  /**
   * One station's profile, remembered.
   *
   * `wallTopAt` bisects against the body to find where the wall leaves it --
   * 32 section queries -- and this was being rebuilt for every point AROUND the
   * profile rather than once per station. That alone was 376,000 of the 572,000
   * queries the sheet made, two thirds of the most expensive stage in the
   * build, all of it recomputing the same number 73 times in a row.
   *
   * One slot is all it needs: the grid walks the whole profile at one station
   * before moving to the next.
   */
  let lastX = NaN, lastProfile = null;
  const profile = (x) => {
    if (x === lastX) return lastProfile;
    lastX = x; lastProfile = buildProfile(x);
    return lastProfile;
  };
  const buildProfile = (x) => {
    const y0 = duct.floorAt(x);
    const wallH = Math.max(wallTopAt(x) - (y0 + duct.cornerR), 0);
    const arc = (Math.PI / 2) * duct.cornerR;
    /**
     * The wall is sampled far more finely than its length deserves, and
     * clustered at its top.
     *
     * Straight arc length gives every part of the profile the same spacing,
     * which is right for quad aspect and wrong for what is happening here: the
     * wall is a tenth of the profile and the lip's round lives in its top few
     * centimetres, so the round fell inside a single quad and the sheet stepped
     * 180 mm sideways in one go. It shaded exactly as square as an unblended
     * cut, because as far as the mesh was concerned it was one.
     */
    // Enough extra to resolve the lip's round, and no more. Boosting it to a
    // fixed share of the whole profile -- which is what this did -- gave the
    // wall five times the sample density of the floor, so the quads went from
    // 18 mm across on the wall to 92 mm on the floor and no single grid count
    // could suit both.
    const wEff = Math.max(wallH, 0.45 * (arc + duct.flat));
    const half = wEff + arc + duct.flat;           // one side, floor centre out
    const at = (s) => {                            // s in [-half, half]
      const side = s < 0 ? -1 : 1, a = Math.abs(s);
      if (a <= duct.flat) return [side * a, y0];
      if (a <= duct.flat + arc) {
        const th = (a - duct.flat) / duct.cornerR;
        return [side * (duct.flat + duct.cornerR * Math.sin(th)),
                y0 + duct.cornerR * (1 - Math.cos(th))];
      }
      const v = Math.min(1, (a - duct.flat - arc) / Math.max(wEff, 1e-9));
      return [side * duct.halfWidth,
              y0 + duct.cornerR + wallH * (1 - (1 - v) ** 2.4)];
    };
    return { at, half };
  };
  // Runs past the duct at BOTH ends. Forward of the cabin the floor is already
  // climbing out through the roof, and aft of the trailing edge there is no
  // body left -- nothing there survives the clip either way. But the sheet has
  // to reach past the point where the two surfaces cross or the cut edge runs
  // off the end of it, and stopping dead on the trailing edge left the last
  // three edges of the seam with nothing to close them, the worst 30 mm.
  const x0 = duct.fromX - 0.5;
  const x1 = duct.toX + 0.2;

  /**
   * The grid, from a target EDGE LENGTH rather than from two fixed counts.
   *
   * Fixed counts cannot know the sheet's proportions. 160 by 72 across a duct
   * 5.7 m long and 6.7 m around the profile makes quads 36 mm by 93 mm -- two
   * and a half to one -- and that ratio IS the tessellation the eye objects to:
   * area over longest-edge-squared came out at 0.168, where a right isoceles
   * triangle, the best a quad grid can do, is 0.250.
   *
   * Derived from the geometry, so it holds at any size of aeroplane and any
   * shape of duct, which two hand-set numbers never could.
   */
  const midProfile = profile((x0 + x1) / 2);
  const target = edge ?? 0.045;
  const even = (n) => 2 * Math.max(3, Math.round(n / 2));
  nx = nx ?? even((x1 - x0) / target);
  nu = nu ?? even((2 * midProfile.half) / target);

  /**
   * The lip, blended rather than square.
   *
   * Left alone the duct meets the skin at a hard crease -- a right angle at the
   * median of the rim, and folded back on itself at 158 degrees at the worst.
   * The blend scoops the sheet away from the rim so it leaves ALONG the skin
   * and turns down into the wall over a band, which is what a fillet does.
   *
   * The scoop is a function of depth into the body, and that is what makes it
   * simple: the rim is exactly where that depth is zero, everywhere it runs,
   * so one profile softens the whole lip -- walls, forward closure, trailing
   * edge -- without ever having to find the rim or follow it.
   *
   * It also has to vanish AT the rim, which is why a bulge and not a ramp. The
   * skin's cut is on the unblended duct and must not move: the two meshes were
   * brought into agreement to 7.4 mm and a lip that shifted the sheet's edge
   * would open all of that back up.
   *
   * `sin` gives both ends for free -- zero and steep at the rim, zero and level
   * where it rejoins the wall -- and its slope at the rim is the angle the
   * crease is turned through.
   *
   * That angle is worked out AT EACH POINT, not fixed. Turning every point
   * through the same 55 degrees brought the median crease down from 90 to 56
   * and made the tail worse -- the 90th percentile went from 129 to 159 and the
   * worst from 158 to 178 -- because much of the rim was already shallow and
   * the scoop drove it straight past flat into a fold. Each point turns by what
   * it actually needs, capped, and a point that needs nothing gets nothing.
   */
  /**
   * A point of the sheet: on the duct's profile, then pushed out to wherever
   * the flared face actually is.
   *
   * SOLVED for rather than offset. The flare depends on depth into the body,
   * and moving a point changes its depth, so the displacement that satisfies it
   * is a fixed point of that relation -- and iterating it diverges near the rim,
   * where the flare's slope exceeds one. Walking in from the outside and
   * bisecting the first sign change finds the OUTERMOST root, which is the
   * fillet; a plain bisection over the whole span can land on an inner one and
   * put the sheet inside the wall it is supposed to be rounding.
   */
  const wallAt = duct.flat + (Math.PI / 2) * duct.cornerR;
  const reach = (duct.eSkin ?? 0) + 0.02;
  const point = (x, t) => {
    const { at, half } = profile(x);
    const s = half * (2 * t - 1);
    const [px, py] = at(s);
    const p = new THREE.Vector3(px, py, -x);
    if (!(reach > 0.02) || !depthInside) return p;
    // Away from the lip there is nothing to solve, and this is most of the
    // sheet: the round only reaches `blend` in from the skin, so a point deeper
    // than that is already where it belongs and the search would spend two
    // dozen section evaluations confirming it. A point well OUTSIDE the body is
    // the same story from the other side -- the solve can only push it further
    // out, and the clip throws it away regardless.
    //
    // Well outside means several grid steps outside, not merely outside. A
    // vertex just past the skin is still one END of the segment the clip
    // interpolates the rim along, so leaving it unsolved moves the rim: cutting
    // at the flare's own reach, 140 mm, took the lip's blend from 65 back
    // towards 55 degrees. Half a metre is past anything the grid can reach
    // across.
    const a0 = depthInside(px, py, -x);
    if (a0 > duct.blend || a0 < -0.5) return p;
    /**
     * Out into the material, along the face's OWN gradient.
     *
     * Along an axis was the obvious thing and is wrong on the corner rounds.
     * The round turns through ninety degrees, so near its top the surface is
     * nearly vertical and stepping in y barely moves you off it -- the solve
     * finds no crossing, returns the unflared point, and the sheet stops short
     * of where the skin was cut. That is the 55 mm gap at the body's aft
     * corner: not a resolution problem, a direction one.
     *
     * The gradient is the direction the face actually changes in, so it is
     * never degenerate on the face's own surface.
     */
    const onWall = Math.abs(s) > wallAt;
    const face = onWall ? duct.faces[1] : duct.faces[0];
    const h = 1e-5;
    const gx = (face(new THREE.Vector3(px + h, py, -x)) - face(new THREE.Vector3(px - h, py, -x))) / (2 * h);
    const gy = (face(new THREE.Vector3(px, py + h, -x)) - face(new THREE.Vector3(px, py - h, -x))) / (2 * h);
    const gl = Math.hypot(gx, gy);
    const dx = gl > 1e-9 ? -gx / gl : (onWall ? Math.sign(s) : 0);
    const dy = gl > 1e-9 ? -gy / gl : (onWall ? 0 : -1);
    const step = (u) => new THREE.Vector3(px + dx * u, py + dy * u, -x);
    const N = 24;
    let hi = reach, fHi = face(step(hi));
    for (let k = N - 1; k >= 0; k--) {
      const lo = reach * (k / N), fLo = face(step(lo));
      if ((fLo > 0) !== (fHi > 0)) {
        let a = lo, b = hi;
        for (let it = 0; it < 30; it++) {
          const m = (a + b) / 2;
          if ((face(step(m)) > 0) === (fLo > 0)) a = m; else b = m;
        }
        return step((a + b) / 2);
      }
      hi = lo; fHi = fLo;
    }
    return p;
  };
  const rows = [];
  for (let i = 0; i <= nx; i++) {
    const x = x0 + (x1 - x0) * (i / nx);
    const row = [];
    for (let j = 0; j <= nu; j++) row.push(point(x, j / nu));
    rows.push(row);
  }
  /**
   * Normals from the finished grid, by central differences on its neighbours.
   *
   * Taken from the grid rather than analytically because the blend is not
   * analytic -- it asks the body how deep it is -- and because differencing the
   * grid gives the normals of the surface that is actually there, blend
   * included, rather than of the one it started as.
   */
  for (let i = 0; i <= nx; i++) {
    for (let j = 0; j <= nu; j++) {
      const ds = rows[i][Math.min(nu, j + 1)].clone().sub(rows[i][Math.max(0, j - 1)]);
      const dx = rows[Math.min(nx, i + 1)][j].clone().sub(rows[Math.max(0, i - 1)][j]);
      const n = new THREE.Vector3().crossVectors(ds, dx);
      if (n.lengthSq() < 1e-18) n.set(0, 1, 0); else n.normalize();
      const p = rows[i][j];
      if (n.dot(new THREE.Vector3(-Math.sign(p.x) * 0.3, 1, 0)) < 0) n.negate();
      pos.push(p.x, p.y, p.z);
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
  /**
   * Keep what lies within the body -- and a little beyond it.
   *
   * Elsewhere the floor has nothing to be the floor OF and would read as a
   * shelf hanging in the open air, so it has to be trimmed. But trimming
   * exactly at the skin puts the sheet's rim on a knife edge: the lip is built
   * nearly tangent to the skin, so whether a vertex near it falls inside or out
   * turns on millimetres, and where it fell outside the sheet simply stopped
   * short of the cut. At the body's aft corner it stopped 55 mm short, and a
   * coarse skin only hid that by having few cut edges there to notice.
   *
   * So the sheet is cut slightly PROUD of the skin and `snapRim` then pulls the
   * overshoot back onto the skin's own cut. Overshoot and snap, rather than try
   * to land on a tangency.
   */
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

/**
 * Give the sheet the body's own normal where the two meet.
 *
 * The blend is in the geometry, but shading is what the eye reads as an edge,
 * and two meshes that meet along a curve shade as a crease unless their normals
 * agree there. The skin's normals at its cut are the body's, unchanged; the
 * sheet's at its rim are the fillet's. Setting the second equal to the first is
 * not a cheat -- the fillet is built tangent to the skin, truncated at 70
 * degrees of its arc, so the body's normal is what the sheet's normal is
 * approaching anyway, to within the 20 degrees that truncation left.
 *
 * Only the rim vertices, so the fillet still carries the turn.
 */
function matchRimNormals(geometry, depthInside, tol = 1e-3) {
  const pos = geometry.getAttribute('position');
  const nrm = geometry.getAttribute('normal');
  if (!nrm) return 0;
  const p = new THREE.Vector3();
  const h = 1e-4;
  let n = 0;
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    if (Math.abs(depthInside(p.x, p.y, p.z)) > tol) continue;
    const g = new THREE.Vector3(
      depthInside(p.x + h, p.y, p.z) - depthInside(p.x - h, p.y, p.z),
      depthInside(p.x, p.y + h, p.z) - depthInside(p.x, p.y - h, p.z),
      depthInside(p.x, p.y, p.z + h) - depthInside(p.x, p.y, p.z - h));
    if (g.lengthSq() < 1e-18) continue;
    g.normalize().negate();                       // the body's OUTWARD normal
    nrm.setXYZ(i, g.x, g.y, g.z);
    n++;
  }
  nrm.needsUpdate = true;
  return n;
}

export function carveDuct(fuse, spec) {
  const duct = ductVolume({ ...spec, depthInside: fuse.userData.depthInside });
  const skin = fuse.userData.skinMesh;
  const before = skin.geometry;
  const kept = carveOut(before, duct.faces);
  skin.geometry = kept;
  before.dispose();

  const sheet = ductSurface(duct, fuse.userData.depthInside,
                            { edge: spec.edge });
  const snap = snapRim(sheet, openEdges(kept));
  const matched = matchRimNormals(sheet, fuse.userData.depthInside);
  const floor = new THREE.Mesh(sheet, skin.material);
  floor.name = 'ductFloor';
  fuse.add(floor);

  fuse.userData.duct = {
    ...spec, mesh: floor, snap, rimNormals: matched,
    skin: kept.userData.clip,
    floorTriangles: sheet.getAttribute('position').count / 3,
  };
  return duct;
}
