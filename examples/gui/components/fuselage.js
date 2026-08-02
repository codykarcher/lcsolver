/**
 * Fuselages.
 *
 * A fuselage here is a **lofted body**: a stack of closed sections swept along
 * a curved centreline. Three functions define one completely --
 *
 *   r(z)     the size of the section at station z
 *   yc(z)    where the section's centre sits, so that a chosen profile line
 *            -- keel forward, crown aft -- stays straight through the taper
 *   sec(th)  the SHAPE of the section, as a radius multiplier by angle
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
 * A circle. What almost every pressurised metal tube actually is -- a cylinder
 * is the only shape that carries pressure in pure tension.
 */
export const circularSection = () => 1;

/* ---- shape law --------------------------------------------------------- */

/**
 * Proportions of a conventional single-aisle jetliner, in diameters and radii
 * so they hold at any size. Numbers are off a 737-800 / A320: 37-38 m long,
 * 3.8-4.0 m across, so a fineness ratio right around 10.
 *
 * Both taper laws come from `(1 - x^A)^B`. B < 1 makes the end round rather
 * than pointed, A > 1 makes the slope vanish where the taper meets the barrel,
 * so neither end blends in with a crease. The tailcone still takes both
 * exponents directly; the nose does not, because B there is pinned at 1/2 and A
 * is solved for from the tip radius -- see noseRadiusExponent below.
 */
const JET = {
  fineness:   10.1,  // overall length / diameter
  noseD:       1.70, // nose length, in diameters
  tailD:       2.90, // tailcone length, in diameters
  noseRadius:  0.50, // RADIUS OF CURVATURE at the nose point, in radii
  tipFlat:     0.18, // see noseCentre(): how the tip cap is kept spherical
  tailA: 1.6, tailB: 0.75,
  tipR:        0.10, // tailcone tip radius, in radii -- the APU exhaust
  keelHold:    0.85, // fraction of the NOSE taper taken off the crown
  crownHold:   1.00, // fraction of the TAILCONE taper taken off the belly
};

/**
 * The nose profile: a sphere of radius `rho` at the point, faired out to the
 * barrel.
 *
 * Asking for the nose by its tip RADIUS rather than by an exponent is worth a
 * little algebra, because a radius is a thing you can picture and an exponent
 * is not. The family r = R*(1 - (1-t)^A)^B turns out to give it up cheaply:
 * near the point, (1-t)^A ~ 1 - At, so
 *
 *     r ~ R * (A t)^B      against a sphere's      r = sqrt(2 rho |z|)
 *
 * The sphere's square-root is B = 1/2 exactly. So B is not a free parameter at
 * all -- it is 1/2, or the tip is not spherical -- and with it fixed, matching
 * the coefficients gives the tip radius outright:
 *
 *     rho = R^2 A / (2 * lNose)      i.e.   A = 2 rho lNose / R^2
 *
 * One knob, and it is the one that means something. A also sets how the taper
 * meets the barrel, and that comes out right on its own: a blunt nose needs a
 * large A, which is a soft join, and A stays above 1 -- the condition for no
 * crease -- for any tip radius above about 0.15 of the body radius.
 */
function noseRadiusExponent(rho, lNose, radius) {
  return Math.max(1.02, 2 * rho * lNose / (radius * radius));
}

/**
 * How far the section centre has climbed by the time the body has grown to a
 * fraction `u` of full radius, as a fraction of the total climb.
 *
 * This exists because a spherical tip and a dead-straight keel are not quite
 * compatible, and the obvious ways of reconciling them both fail.
 *
 * Tie the centre to the radius -- yc proportional to (R - r), the tailcone's
 * own rule -- and the keel is straight and monotonic, but dyc/dz inherits the
 * sphere's vertical tangent at the point. The tip then comes out as a beak:
 * radius of curvature 3.4 rho along the crown and 0.02 rho along the keel,
 * which is a crease under the nose rather than a nose.
 *
 * Tie it to the station instead -- a smoothstep in z -- and the tip is properly
 * spherical, but the centre keeps climbing after the section has stopped
 * growing, and the keel develops a wobble: it dips, rises and falls again.
 *
 * So: tie it to the radius, but ramp the rate in from zero over the first
 * `flat` of the growth. Near the point the centre is stationary, so the cap is
 * a true sphere; past the ramp the rate is constant, so the keel is straight.
 * The ramp integrates a smoothstep, making the result C2 at the junction.
 *
 * The rate outside the ramp is hold/(1 - flat/2), which must stay below 1 or
 * the keel turns back on itself -- that, and not taste, is what caps `hold`.
 */
function noseCentre(u, hold, flat) {
  const m = hold / (1 - flat / 2);
  return u >= flat
    ? m * (u - flat / 2)
    : m * (u ** 3 / flat ** 2 - u ** 4 / (2 * flat ** 3));
}

/**
 * Build the r(z) and yc(z) pair for a three-part body: taper, barrel, taper.
 *
 * Both ends obey the same law, mirrored. A taper has to be spent somewhere --
 * as the section shrinks, one of the two profile lines has to come to meet the
 * other -- and the only question is which line moves:
 *
 *     tailcone   yc = +crownHold * (radius - r)     crown level, belly rises
 *     nose       yc = -keelHold  * (radius - r)     keel level, crown falls
 *
 * At hold = 1 the named line is held exactly straight and the whole taper goes
 * into the other one; at 0 the section stays centred and both lines close in
 * symmetrically. Nothing else changes between the two ends.
 *
 * The nose runs that rate through noseCentre() rather than applying it flat,
 * which holds the centre still over the first sliver of the growth so the point
 * can be a true sphere. Everywhere past the tip it is the plain rule above.
 *
 * Aft that is structural. The cabin ceiling and the fin root both run along the
 * top of the tube, so nothing up there is free to move, while the space under
 * the aft floor is exactly what gets given up for rotation clearance.
 *
 * Forward it is what a nose actually looks like. The keel runs on essentially
 * straight from under the wing and the crown comes down over the flight deck to
 * meet it, putting the point of the nose low -- which is also what gives the
 * crew their over-the-nose sight line. It is emphatically NOT the whole
 * forebody bending downward, which is what a centreline offset applied
 * independently of the taper produces.
 *
 * Keeping the rate at or below 1 makes it impossible for either end to bulge
 * past the barrel: yc -/+ r is monotonic in r over [0, radius], running from
 * -/+ hold*radius at the point to -/+ radius at the join, so the nose cannot
 * hang below the belly nor the tailcone rise above the roof. Forward the rate
 * is hold/(1 - tipFlat/2), which is why keelHold is clamped a little under 1.
 *
 * Both are C1 at their join for free, whatever the blend exponents are set to,
 * because dr/dz already vanishes there -- so no setting of the sliders can put
 * a kink into the parallel part of the body.
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

  const rho = p.noseRadius * radius;
  const noseA = noseRadiusExponent(rho, lNose, radius);
  // Above hold = 1 - tipFlat/2 the keel turns back on itself. Clamp rather than
  // let a slider produce a shape the geometry cannot support, and publish what
  // was actually used so the read-out cannot claim otherwise.
  const keelHold = Math.min(p.keelHold, 0.995 * (1 - p.tipFlat / 2));

  function at(z) {
    if (z > zNose) {                                     // nose
      const t = Math.min(1, Math.max(0, -z / lNose));
      // Exponent 1/2 is not a choice: it is what makes the point spherical.
      const r = radius * Math.sqrt(1 - Math.pow(1 - t, noseA));
      return {
        r,
        yc: radius * (noseCentre(r / radius, keelHold, p.tipFlat) - keelHold),
      };
    }
    if (z > zTail) return { r: radius, yc: 0 };          // barrel
    const s = Math.min(1, Math.max(0, (zTail - z) / lTail));   // tailcone
    const r = rTip + (radius - rTip) * Math.pow(1 - Math.pow(s, p.tailA), p.tailB);
    return { r, yc: p.crownHold * (radius - r) };
  }

  return { at, length, radius, lNose, lTail, zNose, zTail, rTip,
           rho, noseA, keelHold };
}

/* ---- lofting ----------------------------------------------------------- */

/** The surface point at station z, angle th. th = 0 is +X, th = pi/2 is up. */
function surfacePoint(shape, sec, z, th, out = new THREE.Vector3()) {
  const { r, yc } = shape.at(z);
  const rr = r * sec(th);
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
  const sinT = (y - yc) / r;
  if (Math.abs(sinT) > 0.92) return false;
  const th = Math.asin(sinT), dth = halfArc / r;
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
  radius = 1.88,
  length = null,
  shape: shapeOverrides = {},
  section = circularSection,
  detail = false,
  radome = detail,
  flightDeck = detail,
  cabinWindows = detail,
  doors = detail,
  exits = detail,
  apu = detail,
  nSeg = 64,
} = {}) {
  const p = { ...JET, ...shapeOverrides };
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
    keelHold: shape.keelHold, crownHold: p.crownHold,
    /** What the tip radius and derived blend exponent actually came out as. */
    noseRadius: shape.rho, noseExponent: shape.noseA,
    noseLength: shape.lNose, tailLength: shape.lTail,
    cabinZ: [shape.zNose, shape.zTail],
    fineness: L / (2 * radius),
    skinMesh: body,
    decals: parts,
    /** Surface geometry at a station -- what anything mounting to this asks. */
    shapeAt: (z) => shape.at(z),
    surfaceAt: (z, th) => surfacePoint(shape, section, z, th),
    normalAt: (z, th) => surfaceNormal(shape, section, z, th),
    /** Crown and keel lines, which is what a side view is really about. */
    crownAt: (z) => { const { r, yc } = shape.at(z); return yc + r; },
    keelAt: (z) => { const { r, yc } = shape.at(z); return yc - r; },
  });
  return g;
}

export const fuselages = { jetliner: jetlinerFuselage };
