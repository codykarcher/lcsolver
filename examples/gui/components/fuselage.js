/**
 * Fuselages.
 *
 * A fuselage here is a **lofted body**: a stack of closed sections swept along
 * a curved centreline. Three functions define one completely --
 *
 *   r(z)     the size of the section at station z
 *   yc(z)    where the section's centre sits, so that a chosen profile line
 *            -- keel forward, crown aft -- stays straight through the taper
 *   sec(th, z)  the SHAPE of the section, as a radius multiplier by angle --
 *            and, for a body whose section changes along it, by station too
 *
 * -- and everything else in this file is either one of those three for a
 * particular aeroplane, or machinery for turning them into triangles.
 *
 * Splitting the section shape out from the size distribution is the whole
 * point. A conventional jetliner and a double bubble have very nearly the same
 * r(z) and yc(z); what makes one a D8 is `sec`. So the second fuselage is a
 * section function, not a second body, and anything that works on one -- the
 * window rows, the doors, the surface-patch machinery, whatever attaches to it
 * later -- works on the other without being told about it.
 *
 * Axis convention: nose tip at the ORIGIN, body running aft along **-Z**, +Y
 * up. Same as every other component in the library, so a fuselage and an engine
 * dropped into the same scene are already in the same frame.
 */
import * as THREE from 'three';
import { skin, glass, trim, painted, cavity, decal } from './materials.js';
import { orientOutward } from './geom.js';

// Register the decal variants at load rather than on first use, so that a page
// enumerating the palette -- the wireframe toggle does -- sees them before it
// has built anything.
[glass, trim, painted].forEach(decal);

/* ---- section shapes ---------------------------------------------------- */

/**
 * A section is `sec(theta, z) -> radius multiplier`, normalised so that the
 * multiplier is 1 straight up. That normalisation is what lets `r(z)` mean
 * HALF-HEIGHT for every section, circular or not, so one size distribution and
 * one centreline law serve all of them.
 *
 * The station argument is what makes a D8 possible. A tube has one section from
 * end to end; a double bubble does not -- it starts as an ellipse, becomes two
 * lobes over the cabin, and opens into a trough at the back to take the
 * engines. So the section is a function of where you are, not a constant.
 */

/**
 * A circle. What almost every pressurised metal tube actually is -- a cylinder
 * is the only shape that carries pressure in pure tension.
 */
export const circularSection = () => 1;

/** An ellipse `w` times as wide as it is tall. */
export const ellipticalSection = (w) => (th) =>
  1 / Math.hypot(Math.sin(th), Math.cos(th) / w);

/**
 * Two overlapping circles side by side -- a double bubble.
 *
 * Unit lobes with their centres at +/- `offset` from the axis. The union is
 * star-shaped about the centre while offset < 1, so it has a closed polar form:
 * the radius to the boundary is |offset * cos| + sqrt(1 - offset^2 sin^2), the
 * larger root of the nearer lobe. That is worth having exactly rather than by
 * clipping two meshes together -- the crease down the top and bottom centreline
 * is a real feature of the shape, not an artefact, and it comes out on its own.
 *
 * No normalising constant is needed, which is a small piece of luck worth
 * pointing out: the highest point of the union is a LOBE APEX, at (+/-offset, 1),
 * so the half-height is exactly 1 already and `r(z)` keeps meaning half-height
 * for this section as it does for every other. Dividing by the height at the
 * centreline instead -- sqrt(1 - offset^2), which is the valley, not the top --
 * makes the body 12% taller than the half-height asked for.
 *
 * The top is not flat: a hump over each lobe and a valley between them, which
 * is the thing that makes a double bubble recognisable. Half-width is
 * 1 + offset, so that is the one number that grows with the bubble.
 *
 * `trough` deepens that valley -- a Gaussian notch about straight up -- which is
 * how the aft body opens to seat the nacelles.
 */
export function doubleBubbleSection({ offset = 0.45, trough = 0, troughWidth = 0.55 } = {}) {
  const o = Math.min(0.95, Math.max(0, offset));
  return (th) => {
    const s = Math.sin(th), c = Math.cos(th);
    const r = Math.abs(o * c) + Math.sqrt(Math.max(0, 1 - o * o * s * s));
    if (trough <= 0) return r;
    // Angular distance from straight up, wrapped, so the notch does not
    // reappear at the keel when theta runs past pi.
    const d = Math.atan2(Math.sin(th - Math.PI / 2), Math.cos(th - Math.PI / 2));
    const g = d / troughWidth;
    return r * (1 - trough * Math.exp(-g * g));
  };
}

/**
 * A section that changes along the body.
 *
 * `stops` are [fraction aft, section] in order. Between two stops the two
 * sections are blended with a smoothstep, so the morph has no corner at either
 * end of a transition -- which matters, because a discontinuity in the SECTION
 * is a ring-shaped crease around the whole body and is impossible to miss.
 */
export function morphSection(stops, length) {
  return (th, z) => {
    const u = Math.min(1, Math.max(0, -z / length));
    let i = 0;
    while (i < stops.length - 2 && u > stops[i + 1][0]) i++;
    const [u0, a] = stops[i], [u1, b] = stops[i + 1];
    if (u <= u0) return a(th, z);
    if (u >= u1) return b(th, z);
    const t = (u - u0) / (u1 - u0), k = t * t * (3 - 2 * t);
    return a(th, z) * (1 - k) + b(th, z) * k;
  };
}

/* ---- shape law --------------------------------------------------------- */

/**
 * Proportions of a conventional single-aisle jetliner, in diameters and radii
 * so they hold at any size. Numbers are off a 737-800 / A320: 37-38 m long,
 * 3.8-4.0 m across, so a fineness ratio right around 10.
 *
 * Both ends come from `(1 - x^A)^B`. B < 1 makes the end round rather than
 * pointed, A > 1 makes the slope vanish where the taper meets the barrel, so
 * neither end blends in with a crease.
 *
 * **Three of these are deck inputs**: fineness, noseD, and the body radius that
 * arrives beside them. Everything else is the shape of a conventional jetliner
 * and is fixed. They remain overridable through `shape` so a different aircraft
 * can be described later, but nothing routine should touch them.
 *
 * noseB is locked at exactly 1/2, which is worth knowing rather than treating
 * as one value among many: it is the only exponent at which the point has a
 * finite radius of curvature. Above it the tip is a knife edge, below it a
 * flat. At 1/2 the nose closes on a sphere of radius R^2 * noseA / (2 * lNose),
 * which the model reports -- see `noseTipRadius`.
 */
const JET = {
  fineness:   10.1,  // DECK: overall length / diameter
  noseD:      1.70,  // DECK: nose length, in diameters
  tailD:      2.90,  // tailcone length, in diameters
  noseA: 2.7, noseB: 0.50,   // locked
  tailA: 1.6, tailB: 0.75,   // locked
  tipR:       0.10,  // tailcone tip radius, in radii -- the APU exhaust
  keelHold:   0.45,  // fraction of the NOSE taper taken off the crown
  crownHold:  1.00,  // fraction of the TAILCONE taper taken off the belly
};

/**
 * Build the r(z) and yc(z) pair for the whole body: nose, barrel, tailcone.
 *
 * Both ends obey the same law, mirrored. A taper has to be spent somewhere --
 * as the section shrinks, one of the two profile lines has to come and meet the
 * other -- and the only question is which line moves:
 *
 *     tailcone   yc = +crownHold * (radius - r)     crown level, belly rises
 *     nose       yc = -keelHold  * (radius - r)     keel level, crown falls
 *
 * At hold = 1 the named line is held exactly straight and the whole taper goes
 * into the other one; at 0 the section stays centred and both lines close in
 * symmetrically. Nothing else changes between the two ends.
 *
 * Aft that is structural: the cabin ceiling and the fin root both run along the
 * top of the tube, so nothing up there is free to move, while the space under
 * the aft floor is exactly what gets given up for rotation clearance.
 *
 * Keeping either hold at or below 1 makes bulging impossible rather than merely
 * unobserved: yc -/+ r is monotonic in r over [0, radius], running from
 * -/+ hold*radius at the point to -/+ radius at the join, so the nose cannot
 * hang below the belly nor the tailcone rise above the roof.
 *
 * Returned as one object carrying `at(z)` and the stations it was cut at, so
 * that everything downstream -- skin, windows, doors, and later whatever mounts
 * to the side of it -- asks the same question of the same object and cannot
 * disagree about where the surface is.
 */
function jetShape({ length, radius, p = JET }) {
  const lNose = p.noseD * 2 * radius;
  const lTail = p.tailD * 2 * radius;
  const zNose = -lNose;                    // nose taper ends here
  const zTail = -(length - lTail);         // tailcone begins here
  const rTip  = p.tipR * radius;

  function at(z) {
    if (z > zNose) {                                     // nose
      const t = Math.min(1, Math.max(0, -z / lNose));
      // Closes to a POINT, not to an area -- but noseB < 1 gives r a vertical
      // tangent there, so crown and keel both arrive at the tip vertically and
      // the profile is round rather than pointed.
      const r = radius * Math.pow(1 - Math.pow(1 - t, p.noseA), p.noseB);
      return { r, yc: -p.keelHold * (radius - r) };
    }
    if (z > zTail) return { r: radius, yc: 0 };          // barrel
    const s = Math.min(1, Math.max(0, (zTail - z) / lTail));   // tailcone
    const r = rTip + (radius - rTip) * Math.pow(1 - Math.pow(s, p.tailA), p.tailB);
    return { r, yc: p.crownHold * (radius - r) };
  }

  return { at, length, radius, lNose, lTail, zNose, zTail, rTip };
}

/* ---- lofting ----------------------------------------------------------- */

/** The surface point at station z, angle th. th = 0 is +X, th = pi/2 is up. */
function surfacePoint(shape, sec, z, th, out = new THREE.Vector3()) {
  const { r, yc } = shape.at(z);
  const rr = r * sec(th, z);
  return out.set(rr * Math.cos(th), yc + rr * Math.sin(th), z);
}

/** The outward unit normal at (z, th), by finite difference on the surface. */
function surfaceNormal(shape, sec, z, th, out = new THREE.Vector3()) {
  const d = 1e-4;
  const p = surfacePoint(shape, sec, z, th, new THREE.Vector3());
  const a = surfacePoint(shape, sec, z, th + d, new THREE.Vector3()).sub(p);
  const b = surfacePoint(shape, sec, z + d, th, new THREE.Vector3()).sub(p);
  out.crossVectors(a, b).normalize();
  // Point it away from the section's own centre, not away from the z axis: in
  // the tailcone the centre now rides a full radius high, and radial-from-axis
  // gets the sign wrong over the entire upper surface.
  if (out.x * p.x + out.y * (p.y - shape.at(z).yc) < 0) out.negate();
  return out;
}

/**
 * Wind an open sheet outward.
 *
 * Signed volume says nothing about a sheet, so this votes on face normals
 * instead, weighted by area and measured against the local section centre. It
 * measures rather than reasoning about parameter directions, which is the only
 * approach that has survived contact with mirrored and reversed patches.
 */
function faceOutward(geo, shape) {
  const pos = geo.getAttribute('position'), idx = geo.getIndex().array;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = new THREE.Vector3(), cen = new THREE.Vector3();
  let vote = 0;
  for (let t = 0; t + 2 < idx.length; t += 3) {
    a.fromBufferAttribute(pos, idx[t]);
    b.fromBufferAttribute(pos, idx[t + 1]);
    c.fromBufferAttribute(pos, idx[t + 2]);
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
    n.crossVectors(b.clone().sub(a), c.clone().sub(a));   // 2 x area x normal
    vote += n.x * cen.x + n.y * (cen.y - shape.at(cen.z).yc);
  }
  if (vote < 0) {
    for (let t = 0; t + 2 < idx.length; t += 3) {
      const s = idx[t + 1]; idx[t + 1] = idx[t + 2]; idx[t + 2] = s;
    }
    geo.getIndex().needsUpdate = true;
    geo.computeVertexNormals();
  }
  return geo;
}

/**
 * Skin the body: a grid of quads over (station, angle), closed at both ends.
 *
 * Stations are placed on a cosine spacing rather than evenly. The body is
 * nearly straight down the middle and turns hard at both tips, so uniform
 * spacing spends most of its triangles where nothing is happening and then
 * facets the nose. Cosine clustering puts them where the curvature is.
 */
function skinBody(shape, sec, { nStation = 140, nSeg = 64 } = {}) {
  const pos = [], idx = [];
  const v = new THREE.Vector3();
  const L = shape.length;

  const zs = [];
  for (let i = 0; i < nStation; i++) {
    zs.push(-L * (1 - Math.cos((i / (nStation - 1)) * Math.PI)) / 2);
  }

  for (const z of zs) {
    for (let j = 0; j < nSeg; j++) {
      surfacePoint(shape, sec, z, (j / nSeg) * Math.PI * 2, v);
      pos.push(v.x, v.y, v.z);
    }
  }
  for (let i = 0; i < nStation - 1; i++) {
    for (let j = 0; j < nSeg; j++) {
      const a = i * nSeg + j, b = i * nSeg + ((j + 1) % nSeg);
      idx.push(a, b, a + nSeg, b, b + nSeg, a + nSeg);
    }
  }

  // Close both ends onto a centre vertex. At the nose the ring has collapsed to
  // a point already and these triangles are near-degenerate, which is harmless;
  // at the tail the ring is the APU exhaust and the cap is real.
  for (const [ring, z] of [[0, zs[0]], [nStation - 1, zs[nStation - 1]]]) {
    const c = pos.length / 3;
    pos.push(0, shape.at(z).yc, z);
    for (let j = 0; j < nSeg; j++) {
      idx.push(c, ring * nSeg + j, ring * nSeg + ((j + 1) % nSeg));
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return orientOutward(g);
}

/**
 * A rectangular patch of the skin, lifted clear of it -- one window, one door,
 * one pane of the flight deck.
 *
 * Everything applied to a fuselage is applied to a curved, varying surface, so
 * building these as flat quads placed near the body means choosing between
 * floating above it and sinking into it, differently at every station. Sampling
 * the same `shape` the skin was built from and pushing out along the local
 * normal instead means a patch is by construction the right shape and sits the
 * same distance proud everywhere, on any body, without a single special case.
 *
 * The lift alone is not enough to keep it out of the skin -- millimetres on a
 * body tens of metres long are far below the depth buffer's resolution at that
 * range -- so the material is a `decal`, biased toward the camera. The lift
 * fixes the geometry; the bias fixes the raster.
 */
function patch(shape, sec, { z0, z1, th0, th1, lift, material, nz = 3, nt = 5 }) {
  const pos = [], idx = [];
  const v = new THREE.Vector3(), n = new THREE.Vector3();

  for (let i = 0; i < nz; i++) {
    const z = z0 + (z1 - z0) * (i / (nz - 1));
    for (let j = 0; j < nt; j++) {
      const th = th0 + (th1 - th0) * (j / (nt - 1));
      surfacePoint(shape, sec, z, th, v);
      surfaceNormal(shape, sec, z, th, n);
      pos.push(v.x + n.x * lift, v.y + n.y * lift, v.z + n.z * lift);
    }
  }
  for (let i = 0; i < nz - 1; i++) {
    for (let j = 0; j < nt - 1; j++) {
      const p = i * nt + j;
      idx.push(p, p + 1, p + nt, p + 1, p + nt + 1, p + nt);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return new THREE.Mesh(faceOutward(g, shape), decal(material));
}

/**
 * The nose cap -- the radome -- as a closed-at-the-front offset of the skin.
 *
 * Deliberately not a `patch`. A patch is a grid in (z, th), and at the nose the
 * whole first row collapses onto a single point: every quad along it becomes a
 * zero-area sliver, and a fan of degenerate triangles around the tip is exactly
 * the sort of thing that renders as flickering shards. Here the tip is one
 * vertex and a proper triangle fan, so there are no degenerate faces at all.
 */
function noseCap(shape, sec, { zEnd, lift, material, nStation = 12, nSeg = 64 }) {
  const pos = [], idx = [];
  const v = new THREE.Vector3(), n = new THREE.Vector3();

  // The apex. A surface normal is undefined at a point, but the axis is its
  // limit from every direction, so lift the tip straight forward.
  pos.push(0, shape.at(0).yc, lift);

  for (let i = 1; i < nStation; i++) {
    const z = zEnd * (1 - Math.cos((i / (nStation - 1)) * Math.PI / 2));
    for (let j = 0; j < nSeg; j++) {
      const th = (j / nSeg) * Math.PI * 2;
      surfacePoint(shape, sec, z, th, v);
      surfaceNormal(shape, sec, z, th, n);
      pos.push(v.x + n.x * lift, v.y + n.y * lift, v.z + n.z * lift);
    }
  }
  for (let j = 0; j < nSeg; j++) idx.push(0, 1 + j, 1 + ((j + 1) % nSeg));
  for (let i = 1; i < nStation - 1; i++) {
    const base = 1 + (i - 1) * nSeg, next = base + nSeg;
    for (let j = 0; j < nSeg; j++) {
      const jj = (j + 1) % nSeg;
      idx.push(base + j, base + jj, next + j, base + jj, next + jj, next + j);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return new THREE.Mesh(faceOutward(g, shape), decal(material));
}

/**
 * Which bits of skin are already spoken for, as (z, theta) boxes.
 *
 * Two decals on one patch of skin are two surfaces a few millimetres apart
 * competing for the depth buffer, and the result reads as flicker rather than
 * as either of them. Placing everything through one occupancy list makes that
 * impossible by construction, rather than by hand-written exclusion zones that
 * hold at one size and quietly stop holding at another -- which is exactly what
 * the overwing exits did, colliding with the aft door only on a short body.
 *
 * Only the +X side is tracked, because everything here is placed in mirrored
 * pairs, so the two sides can never disagree about what fits.
 */
function occupied(occ, box, dz = 0.06, dth = 0.02) {
  return occ.some((o) => box.z1 - dz < o.z0 && o.z1 < box.z0 + dz &&
                         box.t0 - dth < o.t1 && o.t0 < box.t1 + dth);
}

/**
 * A patch centred on a station, sized in metres, placed at a WORLD HEIGHT --
 * and mirrored to the other side.
 *
 * Placing by height rather than by angle is the whole trick to making a window
 * line look right. Windows sit a fixed distance above the cabin floor and the
 * floor is level, so the row has to be level too. Place by angle instead and
 * the row climbs with the section centre through the tailcone, arcing up the
 * side of the aeroplane in a way no real one does.
 *
 * Returns false, having placed nothing, if the station has run out of body or
 * the skin there is taken. The first is what stops the window row where the
 * rising belly reaches the window line -- rather than at a station picked by
 * hand and wrong at every other size.
 */
function decalRow(shape, sec, out, occ, {
  z, halfZ, y, halfArc, lift, material, nz = 2, nt = 3,
}) {
  const { r, yc } = shape.at(z);
  // Find the angle at which the surface passes through the wanted height. On a
  // circle that is one asin; on any other section the height is r*sec(th)*sin,
  // so it takes a solve. Bisect on the upper right quadrant, where the height
  // rises monotonically with angle for every section here.
  let lo = -Math.PI / 2, hi = Math.PI / 2;
  const yAt = (t) => yc + r * sec(t, z) * Math.sin(t);
  if (y <= yAt(lo) || y >= yAt(hi)) return false;
  for (let i = 0; i < 40; i++) {
    const m = (lo + hi) / 2;
    if (yAt(m) < y) lo = m; else hi = m;
  }
  const th = (lo + hi) / 2;
  if (Math.abs(Math.sin(th)) > 0.92) return false;
  const dth = halfArc / (r * sec(th, z));
  const box = { z0: z + halfZ, z1: z - halfZ, t0: th - dth, t1: th + dth };
  if (occupied(occ, box)) return false;
  occ.push(box);
  for (const s of [1, -1]) {
    out.push(patch(shape, sec, {
      z0: box.z0, z1: box.z1,
      th0: s > 0 ? box.t0 : Math.PI - box.t0,
      th1: s > 0 ? box.t1 : Math.PI - box.t1,
      lift, material, nz, nt,
    }));
  }
  return true;
}

/** Highest (dir 1) or lowest (dir -1) point of the section at a station. */
function extremeY(shape, sec, z, dir, n = 240) {
  const { r, yc } = shape.at(z);
  let best = -Infinity;
  for (let i = 0; i <= n; i++) {
    const th = (i / n) * Math.PI * 2;
    best = Math.max(best, dir * (r * sec(th, z) * Math.sin(th)));
  }
  return yc + dir * best;
}

/** Half-width of the section at a station -- the plan-view silhouette. */
function halfWidth(shape, sec, z, n = 240) {
  const { r } = shape.at(z);
  let best = 0;
  for (let i = 0; i <= n; i++) {
    const th = (i / n) * Math.PI * 2;
    best = Math.max(best, Math.abs(r * sec(th, z) * Math.cos(th)));
  }
  return best;
}

/* ---- the aeroplane ----------------------------------------------------- */

const DEG = Math.PI / 180;

/**
 * A conventional jetliner fuselage.
 *
 * **What you get by default is the bare outer mould line and nothing else.**
 * Windows, doors, exits, the flight deck, the radome and the APU exhaust are
 * all detail applied to a shape, and none of them can tell you whether the
 * shape underneath is right -- they mostly get in the way of seeing it. So they
 * are off unless asked for. The machinery for all of them is intact and
 * verified; `detail: true` turns the lot back on at once.
 *
 * Every term of the shape law is overridable through `shape`. Getting an OML
 * right is a matter of pushing on those numbers and looking, and a parameter
 * you have to edit a source file to change is a parameter you will not try.
 *
 * @param {number} radius     body radius. Default 1.88 -- a 737's 3.76 m tube.
 * @param {number} length     nose to tailcone tip. Default fineness 10.1.
 * @param {object} shape      overrides on the shape law; see JET above.
 * @param {Function} section  section shape, th -> radius multiplier.
 * @param {boolean} detail    shorthand for every feature flag at once.
 */
export function jetlinerFuselage({
  radius = 1.88, fineness = null, noseD = null, length = null,
  shape: shapeOverrides = {}, section = circularSection, ...rest
} = {}) {
  // The three deck inputs are named parameters rather than buried in `shape`,
  // because that is how they arrive and how they should read at a call site.
  const p = {
    ...JET, ...shapeOverrides,
    ...(fineness != null ? { fineness } : {}),
    ...(noseD != null ? { noseD } : {}),
  };
  return buildFuselage({ radius, length, p, section, ...rest });
}

/**
 * Loft a body from a size distribution and a section, and hang the detail on it.
 *
 * Everything below the shape law is shared: a D8 and a tube are skinned by the
 * same code, get their windows placed by the same code, and are checked by the
 * same code. What differs between two aeroplanes is `p` and `section`, and
 * nothing else should have to.
 */
function buildFuselage({
  radius, length = null, p, section = circularSection,
  detail = false,
  radome = detail,
  flightDeck = detail,
  cabinWindows = detail,
  doors = detail,
  exits = detail,
  apu = detail,
  nSeg = 64,
}) {
  const L = length ?? p.fineness * 2 * radius;
  const shape = jetShape({ length: L, radius, p });
  const g = new THREE.Group();
  const lift = radius * 0.005;
  const parts = [], occ = [];

  const body = new THREE.Mesh(skinBody(shape, section, { nSeg }), skin);
  g.add(body);

  // The radome is a different material to the skin on a real aeroplane -- it
  // has to be transparent to the radar behind it -- and is usually left
  // unpainted or in a contrasting grey.
  if (radome) {
    parts.push(noseCap(shape, section, {
      zEnd: -shape.lNose * 0.22, lift: lift * 0.5, material: painted, nSeg,
    }));
  }

  if (flightDeck) {
    // Two panes a side: a windscreen over the front quarter and a side window
    // aft of it, offset down and back the way a real one is. These are the one
    // thing still placed by angle rather than by height -- the flight deck
    // wraps over the crown, so there is no single height to place it at.
    const n = shape.lNose;
    for (const [z0, z1, t0, t1, nz, nt] of [
      [-0.30 * n, -0.60 * n, 78 * DEG, 40 * DEG, 4, 5],
      [-0.47 * n, -0.74 * n, 34 * DEG, 11 * DEG, 3, 4],
    ]) {
      occ.push({ z0, z1, t0: Math.min(t0, t1), t1: Math.max(t0, t1) });
      for (const s of [1, -1]) {
        parts.push(patch(shape, section, {
          z0, z1, lift, material: glass, nz, nt,
          th0: s > 0 ? t0 : Math.PI - t0,
          th1: s > 0 ? t1 : Math.PI - t1,
        }));
      }
    }
  }

  // Order matters: whatever is placed first owns the skin. Doors before exits
  // before windows, which is the order of how badly each one wants its nominal
  // station -- a door has to line up with a galley, a window is one of fifty.
  if (doors) {
    for (const z of [-shape.lNose * 1.14, shape.zTail - shape.lTail * 0.10]) {
      decalRow(shape, section, parts, occ, {
        z, halfZ: 1.83 / 2, y: radius * Math.sin(6 * DEG), halfArc: 0.86 / 2,
        lift: lift * 0.7, material: trim, nz: 3, nt: 4,
      });
    }
  }
  if (exits) {
    for (const z of [-L * 0.480, -L * 0.545]) {
      decalRow(shape, section, parts, occ, {
        z, halfZ: 1.10 / 2, y: radius * Math.sin(14 * DEG), halfArc: 0.51 / 2,
        lift: lift * 0.7, material: trim, nz: 2, nt: 3,
      });
    }
  }

  if (cabinWindows) {
    // 0.51 m (20 in) pitch -- the frame spacing, which is what actually sets
    // it -- at a fixed height, running aft until the body stops offering
    // anywhere to put one. A station already taken by a door is skipped, not
    // stopped at, which is why this cannot use decalRow's return value to end
    // the row.
    const pitch = 0.51, y = radius * Math.sin(20 * DEG);
    for (let z = -shape.lNose * 1.02; z > -L; z -= pitch) {
      const { r, yc } = shape.at(z);
      if (Math.abs((y - yc) / r) > 0.92) break;          // belly has reached it
      decalRow(shape, section, parts, occ, {
        z, halfZ: 0.115, y, halfArc: 0.165 / 2, lift, material: glass,
      });
    }
  }

  for (const p of parts) g.add(p);

  // The APU exhaust: the tailcone does not close to a point, it closes onto a
  // hole, and a hole has to read as one.
  if (apu) {
    const hole = new THREE.Mesh(
      new THREE.CircleGeometry(shape.rTip * 0.86, 24), cavity);
    hole.position.set(0, shape.at(-L).yc, -L + lift);
    hole.rotation.y = Math.PI;
    g.add(hole);
  }

  Object.assign(g.userData, {
    length: L, radius, section, shapeParams: p,
    /** Where the section is widest, and how wide. Plan view in two numbers. */
    maxHalfWidth: (() => {
      let w = 0, at = 0;
      for (let i = 0; i <= 200; i++) {
        const z = -L * i / 200, h = halfWidth(shape, section, z);
        if (h > w) { w = h; at = z; }
      }
      return { halfWidth: w, z: at };
    })(),
    keelHold: p.keelHold, crownHold: p.crownHold,
    /**
     * Curvature at the point, three ways. Meaningful ONLY because noseB is 1/2
     * -- at any other exponent the tip curvature is 0 or unbounded and there is
     * no radius to report, so these are null rather than numbers that lie.
     *
     * The section curve closes on rho = R^2 * noseA / (2 * lNose). What you
     * SEE, though, is the two meridians, and they are not that: the centreline
     * is still moving at the point, so crown and keel pick up (1 +/- keelHold)^2
     * of it. At keelHold 0.45 that is 2.10 and 0.30 -- the underside of the tip
     * is seven times tighter than the top. The tip is round in section and a
     * long way from round in profile, and only the meridians are visible.
     */
    noseTipRadius: Math.abs(p.noseB - 0.5) < 1e-9
      ? radius * radius * p.noseA / (2 * shape.lNose) : null,
    noseTipRadiusCrown: Math.abs(p.noseB - 0.5) < 1e-9
      ? radius * radius * p.noseA / (2 * shape.lNose) * (1 + p.keelHold) ** 2 : null,
    noseTipRadiusKeel: Math.abs(p.noseB - 0.5) < 1e-9
      ? radius * radius * p.noseA / (2 * shape.lNose) * (1 - p.keelHold) ** 2 : null,
    noseLength: shape.lNose, tailLength: shape.lTail,
    cabinZ: [shape.zNose, shape.zTail],
    fineness: L / (2 * radius),
    skinMesh: body,
    decals: parts,
    /** Surface geometry at a station -- what anything mounting to this asks. */
    shapeAt: (z) => shape.at(z),
    surfaceAt: (z, th) => surfacePoint(shape, section, z, th),
    normalAt: (z, th) => surfaceNormal(shape, section, z, th),
    /**
     * Crown, keel and half-width: the silhouettes, which is what side and plan
     * views are really about.
     *
     * Scanned over the section rather than taken as yc +/- r. For a circle the
     * two agree exactly, so nothing changes for a tube; for a double bubble
     * they do not, because the highest point of the section is over a LOBE and
     * not on the centreline. Assuming the crown is straight up would report the
     * valley between the lobes as the top of the aeroplane.
     */
    crownAt: (z) => extremeY(shape, section, z, 1),
    keelAt: (z) => extremeY(shape, section, z, -1),
    halfWidthAt: (z) => halfWidth(shape, section, z),
  });
  return g;
}

/* ---- D8 ---------------------------------------------------------------- */

/**
 * Proportions of a D8 double-bubble body, in half-heights so they hold at any
 * size. Three deck inputs as before -- half-height, fineness, nose length --
 * and four more that are what make it a D8 rather than a tube.
 *
 * The numbers are a first cut and should be treated as such: unlike the
 * jetliner, which was settled against photographs, there are very few D8s to
 * look at.
 */
const D8 = {
  fineness:   9.6,   // DECK: length / (2 * half-height)
  noseD:      1.45,  // DECK: nose length, in full heights
  tailD:      2.30,  // aft body length, in full heights
  noseA: 2.4, noseB: 0.50,
  tailA: 1.5, tailB: 0.70,
  tipR:       0.20,  // aft body does not close to a point the way a tube does
  keelHold:   0.80,  // a D8 is flat-bottomed: the keel holds almost all the way
  crownHold:  1.00,

  noseWidth:  1.20,  // ellipse aspect at the point
  bubble:     0.45,  // lobe offset, in lobe radii -- how far apart the bubbles
  trough:     0.30,  // depth of the aft valley, as a fraction of half-height
  troughWidth: 0.60, // angular width of that valley, radians
  // Where along the body each transition happens, as a fraction of length.
  uBubble:    0.24,  // elliptical nose has become the double bubble by here
  uOpen:      0.66,  // aft opening starts
  uTrough:    0.84,  // and is fully open by here
};

/**
 * A D8 double-bubble fuselage.
 *
 * Three shapes in one body, which is the whole difficulty. It starts as a wide
 * ELLIPSE at the point, becomes a DOUBLE BUBBLE over the cabin, and opens at the
 * back into a TROUGH between the two lobes for the engines to sit in. None of
 * that is a size distribution -- the half-height and centreline behave much like
 * a tube's -- it is the SECTION changing along the length, which is why the
 * section here is a function of station and not a constant.
 *
 * The double bubble is two overlapping circles, taken exactly rather than by
 * clipping meshes together, so the crease down the top and bottom centreline
 * comes out on its own. See doubleBubbleSection.
 *
 * The aft trough is that same section with its upper valley deepened. It is
 * worth being clear about what this does and does not claim: it produces the
 * fuselage-side shape the nacelles nest into. The nacelles themselves are
 * engines and belong to the engine library; what integrates them is that the
 * body offers them a seat, and `nacelleSeat` reports where that seat is so an
 * engine can be placed on it rather than guessed into position.
 */
export function d8Fuselage({
  radius = 1.90,            // HALF-HEIGHT, not a radius -- see the section note
  fineness = null,
  noseD = null,
  length = null,
  shape: shapeOverrides = {},
  ...rest
} = {}) {
  const p = {
    ...D8, ...shapeOverrides,
    ...(fineness != null ? { fineness } : {}),
    ...(noseD != null ? { noseD } : {}),
  };
  const L = length ?? p.fineness * 2 * radius;

  const bubble = doubleBubbleSection({ offset: p.bubble });
  const open = doubleBubbleSection({
    offset: p.bubble, trough: p.trough, troughWidth: p.troughWidth });

  // Ellipse to bubble to trough. The first and last stops are repeated at the
  // ends so the morph holds its shape there instead of drifting.
  const section = morphSection([
    [0.00, ellipticalSection(p.noseWidth)],
    [p.uBubble, bubble],
    [p.uOpen, bubble],
    [p.uTrough, open],
    [1.00, open],
  ], L);

  const g = buildFuselage({ radius, length: L, p, section, ...rest });

  const u = g.userData;
  const lobe = Math.atan2(Math.sqrt(1 - p.bubble * p.bubble), p.bubble);
  Object.assign(u, {
    isDoubleBubble: true,
    bubbleOffset: p.bubble,
    /** Section width over height at the cabin -- what makes it look like a D8. */
    cabinWidthOverHeight: 2 * u.halfWidthAt(-L * 0.45)
      / (u.crownAt(-L * 0.45) - u.keelAt(-L * 0.45)),
    /**
     * Where an engine can sit: the valley between the lobes, aft.
     *
     * Returned as a point and the local surface normal, so a nacelle is placed
     * ON the body rather than at coordinates that happen to look right and stop
     * being right the moment the trough or the bubble offset changes.
     */
    nacelleSeat: (side = 1, uz = 0.90) => {
      const z = -L * uz;
      // Out along the valley from dead centre, but not as far as the lobe crest.
      const th = Math.PI / 2 - side * p.troughWidth * 0.55;
      return {
        z,
        point: u.surfaceAt(z, th),
        normal: u.normalAt(z, th),
        valleyY: u.shapeAt(z).yc + u.shapeAt(z).r * section(Math.PI / 2, z),
        lobeAngle: lobe,
      };
    },
  });
  return g;
}

export const fuselages = { jetliner: jetlinerFuselage, d8: d8Fuselage };
