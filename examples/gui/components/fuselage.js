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
 * The tailcone is one curve, `(1 - x^A)^B`: B < 1 rounds the end rather than
 * pointing it, A > 1 makes the slope vanish where it meets the barrel so it
 * blends in without a crease. The nose is not one curve at all -- it is a
 * radome, a break and a fairing. See noseProfile below.
 */
const JET = {
  fineness:   10.1,  // overall length / diameter
  noseD:       1.25, // nose length, in diameters: radome plus fairing
  tailD:       2.90, // tailcone length, in diameters
  // The nose is two bodies -- see noseProfile below.
  radomeFrac:  0.36, // radome's share of the nose length
  radomeAxis:  0.33, // radome axis height, 0 at the keel, 1 at the crown
  radomeBase:  0.58, // radome radius where it ends, in radii
  radomeTip:   0.20, // spherical tip radius, in radii
  radomeFull:  0.85, // 0 a cone, 1 a tangent ogive
  foreR:       2.00, // fairing: how fast the section opens to full
  foreK:       2.20, // fairing: how fast the keel settles onto the barrel line
  kinkBlend:   0.05, // window the break is smoothed over, of nose length
  tailA: 1.6, tailB: 0.75,
  tipR:        0.10, // tailcone tip radius, in radii -- the APU exhaust
  crownHold:   1.00, // fraction of the TAILCONE taper taken off the belly
};

/**
 * A spherically blunted ogive -- the radome.
 *
 * `fullness` runs from a plain cone at 0 to a TANGENT ogive at 1. It is worth
 * having as a knob because it does double duty: it sets how full the radome is
 * along its length, and it sets the slope the radome arrives at its own base
 * with, (1 - fullness) * rBase / L, which is one half of the break at the
 * windscreen. A tangent ogive arrives parallel to its base circle and gives the
 * sharpest break; a cone arrives at an angle and gives the softest.
 *
 * The point is then cut off by a sphere of radius `rho` tangent to that curve.
 * Tangency fixes both unknowns at once, from the sphere's centre lying rho
 * along the inward normal with its radial coordinate vanishing:
 *
 *     r(xt) * sqrt(1 + r'(xt)^2) = rho          where to cut
 *     xc = xt + r(xt) * r'(xt)                  where the centre goes
 *
 * so slope continuity across the graft is not checked afterwards, it is the
 * condition that was solved. Blunting shortens the ogive, so the virtual length
 * is solved for too and the finished radome is exactly the length asked for.
 */
function bluntedOgive({ rBase, length, rho, fullness }) {
  const raw = (x, L) => {
    const t = Math.min(1, Math.max(0, x / L));
    const Ro = (rBase * rBase + L * L) / (2 * rBase);   // tangent-ogive radius
    const og = Math.sqrt(Math.max(0, Ro * Ro - (L - x) * (L - x))) - (Ro - rBase);
    return (1 - fullness) * rBase * t + fullness * og;
  };
  const slope = (x, L) => {
    const h = L * 1e-6;
    const a = Math.max(0, x - h), b = Math.min(L, x + h);
    return (raw(b, L) - raw(a, L)) / (b - a);
  };

  const solve = (L) => {
    const g = (x) => raw(x, L) * Math.hypot(1, slope(x, L)) - rho;
    let lo = 0, hi = L;
    if (g(hi) <= 0) return { xt: L, rt: rBase, xc: L, x0: 0, blunt: false };
    for (let i = 0; i < 80; i++) {
      const m = (lo + hi) / 2;
      if (g(m) < 0) lo = m; else hi = m;
    }
    const xt = (lo + hi) / 2, rt = raw(xt, L), m = slope(xt, L);
    const xc = xt + rt * m;
    return { xt, rt, xc, x0: xc - rho, blunt: true };
  };

  let L = length, s = solve(L);
  for (let i = 0; i < 60; i++) {
    const err = (L - s.x0) - length;
    if (Math.abs(err) < 1e-12 * length) break;
    L -= err * 0.9;
    s = solve(L);
  }

  return {
    ...s, L,
    /** Radius a distance `d` aft of the finished point. */
    at: (d) => {
      const x = d + s.x0;
      if (s.blunt && x <= s.xt) {
        return Math.sqrt(Math.max(0, rho * rho - (x - s.xc) * (x - s.xc)));
      }
      return raw(Math.min(x, L), L);
    },
    /** dr/d(aft) where the radome meets its base ring. */
    baseSlope: (1 - fullness) * rBase / L,
  };
}

/**
 * The nose, as **two bodies with a break between them**.
 *
 * Every previous attempt here fitted one continuous curve from the point to the
 * barrel, and none of them looked like an aeroplane. They could not: a nose in
 * this class is not one shape. It is
 *
 *   1. a **radome** -- a blunted ogive of revolution about its OWN axis, which
 *      sits low, about a third of the way up the section from the keel. That is
 *      why the point of a jetliner is down near the belly line while the body
 *      behind it is centred;
 *   2. a **distinct break at the windscreen**, where the radome ends and the
 *      pressurised structure begins. It is a real joint on a real aeroplane and
 *      it reads as one;
 *   3. a **fairing** from there back into the barrel, which lifts the section
 *      centre from the radome axis up to zero and opens the radius out to full,
 *      arriving parallel so the parallel part starts without a crease.
 *
 * Splitting it in two is what makes the tip trivially right, incidentally. The
 * radome's axis is constant, so the blunting sphere is a genuine sphere on both
 * meridians rather than something that has to be argued about -- which is the
 * problem that ate the last three attempts.
 *
 * The break is smoothed over a short window rather than left as a corner. Real
 * ones are a joint, not a knife edge, and a slope discontinuity in a lofted
 * surface catches highlights in a way nothing else does.
 */
function noseProfile({ radius, lNose, p }) {
  // "A third of the way up from the bottom": measured across the full section,
  // so 0 is the keel, 0.5 the axis, 1 the crown.
  const yAxis = -radius + 2 * radius * p.radomeAxis;
  const lRadome = lNose * p.radomeFrac;
  const lFore = lNose - lRadome;
  // The radome cannot be so fat that its base circle already reaches below the
  // barrel's keel -- there would be nothing for the fairing to settle onto, and
  // the nose would hang under the tube however the blends were set. Clamp it.
  const rBase = Math.min(p.radomeBase * radius, 0.94 * (radius + yAxis));
  const rho = Math.min(p.radomeTip * radius, 0.98 * rBase);

  const dome = bluntedOgive({
    rBase, length: lRadome, rho, fullness: Math.min(1, Math.max(0, p.radomeFull)),
  });

  // The fairing, defined by its RADIUS and its KEEL, with the section centre
  // derived as the sum. Blending the centre directly instead is the obvious
  // thing and it does not work: the centre then lifts faster than the section
  // opens over the middle of the fairing, and the keel dips below the barrel's
  // own keel line -- the nose bulges under the tube. Measured, 50 stations of
  // it. Driving the keel makes that unrepresentable rather than merely
  // discouraged, which is the same reason the tailcone drives its crown.
  //
  // Both terms are flat at the barrel -- exponents above 1 -- so the join there
  // is free of a crease whatever the knobs say, and both hit the radome's base
  // values exactly at the break, so the two bodies meet.
  const keelRise = radius + yAxis - rBase;    // how high the keel sits at the break
  const fore = (d) => {
    const s = (d - lRadome) / lFore;
    const r = radius - (radius - rBase) * Math.pow(1 - s, p.foreR);
    const keel = -radius + keelRise * Math.pow(1 - s, p.foreK);
    return { r, yc: keel + r };
  };
  // Smoothing the break. The window sits ENTIRELY AFT of it, blending the
  // radome carried on along its own tangent into the fairing.
  //
  // Straddling the break instead means extrapolating the fairing forward, and
  // the two extrapolations then disagree about where the surface is at the
  // start of the window: the smoothstep's own derivative term, k' * (b - a),
  // picks that disagreement up and can turn the keel back on itself. Measured,
  // a handful of stations of it. Anchored at the break the two agree exactly --
  // both are (rBase, yAxis) there -- so b - a starts at zero, and the result is
  // C1 at both ends of the window with the turn spread smoothly across it.
  const w = Math.max(1e-9, Math.max(0, p.kinkBlend) * lNose);

  function at(d) {
    if (d <= lRadome) return { r: dome.at(d), yc: yAxis };
    if (d >= lRadome + w) return fore(d);
    const t = (d - lRadome) / w;
    const k = t * t * (3 - 2 * t);                       // smoothstep
    const a = { r: rBase + dome.baseSlope * (d - lRadome), yc: yAxis };
    const b = fore(d);
    return { r: a.r + (b.r - a.r) * k, yc: a.yc + (b.yc - a.yc) * k };
  }

  return { at, yAxis, lRadome, lFore, rBase, rho, dome,
           breakZ: -lRadome, blendWidth: w };
}

/**
 * Build the r(z) and yc(z) pair for the whole body: nose, barrel, tailcone.
 *
 * The two ends are not the same shape and should not be forced to share a law.
 *
 * **Aft** is one body. A taper has to be spent somewhere -- as the section
 * shrinks, one of the two profile lines has to come and meet the other -- and
 * yc = crownHold * (radius - r) decides which. At 1 the crown is held exactly
 * straight and the whole taper goes into the belly; at 0 the section stays
 * centred and both lines close in symmetrically. Holding the crown is
 * structural: the cabin ceiling and the fin root both run along the top of the
 * tube, so nothing up there is free to move, while the space under the aft
 * floor is exactly what gets given up for rotation clearance.
 *
 * **Forward** is two bodies with a joint between them -- a radome on its own
 * low axis, a break at the windscreen, then a fairing into the barrel. See
 * noseProfile. Trying to make one continuous curve do that was the mistake
 * behind several attempts at this file.
 *
 * Both ends still meet the barrel without a crease, but for different reasons:
 * aft because dr/dz vanishes there, forward because the fairing's exponents are
 * above 1 so both its terms arrive flat.
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

  const nose = noseProfile({ radius, lNose, p });

  function at(z) {
    if (z > zNose) return nose.at(-z);                   // nose
    if (z > zTail) return { r: radius, yc: 0 };          // barrel
    const s = Math.min(1, Math.max(0, (zTail - z) / lTail));   // tailcone
    const r = rTip + (radius - rTip) * Math.pow(1 - Math.pow(s, p.tailA), p.tailB);
    return { r, yc: p.crownHold * (radius - r) };
  }

  return { at, length, radius, lNose, lTail, zNose, zTail, rTip, nose };
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
    crownHold: p.crownHold,
    /** The radome, as built: where its axis sits and where it ends. */
    radomeLength: shape.nose.lRadome,
    radomeAxisY: shape.nose.yAxis,
    radomeBaseRadius: shape.nose.rBase,
    noseRadius: shape.nose.rho,
    noseCapRadius: shape.nose.dome.rt,
    noseBreakZ: shape.nose.breakZ,
    noseBlendWidth: shape.nose.blendWidth,
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
