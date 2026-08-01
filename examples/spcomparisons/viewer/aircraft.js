// aircraft.js -- build a viewable aircraft from a solved spcomparisons case.
//
// Pure geometry: (values, arch) -> THREE.Group. Nothing here reads the DOM or
// the network, so the same function serves the single-case viewer, a grid of
// twenty, or an A/B overlay.
//
// TWO KINDS OF GEOMETRY, and the difference is the point of the picture:
//
//   Sized      Drawn from the solve. Fuselage length and section, floor width
//              and length, seat rows and count, wing planform and the spar
//              cap/web thicknesses inside it, tail planform, fan diameter and
//              the engine's own flow areas, gear stations, leg lengths, track
//              and tyre sizes. If the optimiser moves one, the picture moves.
//
//   Furnishing Not in any sizing model, invented so it reads as an aeroplane:
//              overhead bins, sidewall lining, windows, doors, cockpit
//              glazing, the shape of a seat, blade count, cowl profile.
//              Tagged `placed: false` with a note, so the viewer can say so
//              rather than let the detail imply a claim.
//
// The furnishing is not decoration for its own sake. A cabin drawn as
// featureless blocks reads as "roughly this big". A cabin with an aisle you
// can see down and bins at shoulder height reads as "six abreast, and the
// aisle is where it should be" -- which is a thing you can be WRONG about,
// and therefore a thing worth drawing.
//
// Divergence from the ESP model is deliberate. This answers "does that look
// sane" and "how do these twenty differ" in a second; it is not the CSM.

import * as THREE from './lib/three.module.js';

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
const V = (v, k, d = 0) => (k in v && isFinite(v[k]) ? v[k] : d);
const IN = 0.0254;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

/** A tyre diameter that might be inches or metres. Decide by magnitude: the
 *  solves carry d_t in inches while w_t is in metres, and nothing has a five
 *  metre tyre. */
const tyreDia = (x, fallback) =>
  (!isFinite(x) || x <= 0) ? fallback : (x > 5 ? x * IN : x);

/** Merge parts into one geometry, so a repeated object is one draw call.
 *  `mergeGeometries` lives in an addon that is not vendored, and this is the
 *  four lines of it that matter here. */
function merge(parts) {
  const P = [], N = [];
  const o = new THREE.Object3D();
  for (const { geom, pos = [0, 0, 0], rot = [0, 0, 0] } of parts) {
    const g = geom.index ? geom.toNonIndexed() : geom.clone();
    o.position.set(...pos);
    o.rotation.set(...rot);
    o.updateMatrix();
    g.applyMatrix4(o.matrix);
    P.push(...g.attributes.position.array);
    N.push(...g.attributes.normal.array);
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(P, 3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
  return out;
}

/** A rounded rectangle, for anything punched out of the skin. */
function roundedRect(w, h, r) {
  r = Math.min(r, w / 2 - 1e-3, h / 2 - 1e-3);
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2);
  s.lineTo(w / 2 - r, -h / 2);
  s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  s.lineTo(w / 2, h / 2 - r);
  s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  s.lineTo(-w / 2 + r, h / 2);
  s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  s.lineTo(-w / 2, -h / 2 + r);
  s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  return s;
}

/**
 * A rounded opening that is literally part of the skin.
 *
 * Every point of the outline is mapped onto the fuselage surface -- station x
 * gives the section scale, height y gives the half-width -- so the patch is
 * curved exactly as the body is. A flat plate laid against a barrel is only
 * tangent at its centre and stands proud at the corners, which is what the
 * windows and doors were before.
 */
function skinPatch(x0, y0, w, h, r, side, wAtS, scaleAt, inset = 0.015,
                   droopAt = () => 0) {
  const pts = roundedRect(w, h, r).getPoints(56);
  const map = (u, v) => {
    const x = x0 + u, y = y0 + v;
    // the nose sections are dropped, so the section centre is not at y = 0
    return [x, y, side * (wAtS(y - droopAt(x), scaleAt(x)) - inset)];
  };
  const pos = [];
  const c = map(0, 0);
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    const A = map(a.x, a.y), B = map(b.x, b.y);
    if (side > 0) pos.push(...c, ...A, ...B);
    else pos.push(...c, ...B, ...A);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}

/** NACA 4-digit half-thickness distribution, used for every lifting surface. */
function naca(t, n = 24) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const x = 0.5 * (1 - Math.cos((Math.PI * i) / n));      // cosine spacing
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x * x
                        + 0.2843 * x ** 3 - 0.1015 * x ** 4);
    pts.push([x, yt]);
  }
  return pts;
}

/** Loft a closed section list into a triangle mesh. */
function loft(sections, material) {
  const pos = [];
  const push = (a, b, c) => pos.push(...a, ...b, ...c);
  for (let s = 0; s < sections.length - 1; s++) {
    const A = sections[s], B = sections[s + 1];
    const n = Math.min(A.length, B.length);
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      push(A[i], B[i], B[j]);
      push(A[i], B[j], A[j]);
    }
  }
  for (const S of [sections[0], sections[sections.length - 1]]) {    // caps
    const c = S.reduce((a, p) => [a[0] + p[0] / S.length, a[1] + p[1] / S.length,
                                  a[2] + p[2] / S.length], [0, 0, 0]);
    for (let i = 0; i < S.length; i++) push(c, S[i], S[(i + 1) % S.length]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return new THREE.Mesh(g, material);
}

const mat = (color, opacity = 1, side = THREE.DoubleSide, extra = {}) =>
  new THREE.MeshStandardMaterial({
    color, opacity, transparent: opacity < 1, side,
    roughness: 0.55, metalness: 0.05,
    depthWrite: opacity > 0.95, ...extra,
  });

// ---------------------------------------------------------------------------
// lifting surfaces
// ---------------------------------------------------------------------------
/**
 * A trapezoidal lifting surface as a lofted solid.
 * x aft, y up, z starboard. `vertical` rotates it into the x-y plane.
 */
function surface({ span, cRoot, cTip, tau, sweepDeg = 0, dihedralDeg = 0,
                   vertical = false, halves = true }, material) {
  const semi = vertical ? span : span / 2;
  const nSpan = 12, prof = naca(tau);
  const sideSections = (sign) => {
    const out = [];
    for (let k = 0; k <= nSpan; k++) {
      const fr = k / nSpan;
      const y = sign * fr * semi;
      const c = cRoot + (cTip - cRoot) * fr;
      const xLE = Math.abs(y) * Math.tan((sweepDeg * Math.PI) / 180);
      const zUp = Math.abs(y) * Math.tan((dihedralDeg * Math.PI) / 180);
      const ring = [];
      for (let i = 0; i < prof.length; i++)          // upper, LE -> TE
        ring.push(vertical ? [xLE + prof[i][0] * c, Math.abs(y), prof[i][1] * c]
                           : [xLE + prof[i][0] * c, zUp + prof[i][1] * c, y]);
      for (let i = prof.length - 2; i > 0; i--)      // lower, TE -> LE
        ring.push(vertical ? [xLE + prof[i][0] * c, Math.abs(y), -prof[i][1] * c]
                           : [xLE + prof[i][0] * c, zUp - prof[i][1] * c, y]);
      out.push(ring);
    }
    return out;
  };
  const g = new THREE.Group();
  g.add(loft(sideSections(1), material));
  if (halves && !vertical) g.add(loft(sideSections(-1), material));
  return g;
}

/**
 * The structural box inside a lifting surface: two caps and two webs, drawn at
 * the thickness the optimiser actually chose.
 *
 * `t_cap` and `t_web` are millimetres on a metre-scale aeroplane, so they are
 * invisible at true scale. `exaggerate` multiplies them for legibility and the
 * viewer reports the true value alongside -- drawing them to scale would be
 * honest and useless.
 */
function box({ span, cRoot, cTip, tau, sweepDeg = 0, dihedralDeg = 0,
               tCap, tWeb, front = 0.15, rear = 0.60, vertical = false,
               exaggerate = 1 }, capMat, webMat) {
  const semi = vertical ? span : span / 2;
  const g = new THREE.Group();
  const tc = tCap * exaggerate, tw = tWeb * exaggerate;
  const sides = vertical ? [1] : [1, -1];

  // One plate at a time. Each is a lofted ribbon following the same planform
  // AND THE SAME DIHEDRAL as the skin -- without the dihedral term the box
  // stays flat while the wing rises, and pokes out through the lower surface
  // somewhere around mid-span.
  const plate = (sign, kind) => {
    const secs = [];
    for (let k = 0; k <= 8; k++) {
      const f = k / 8, y = sign * f * semi;
      const c = cRoot + (cTip - cRoot) * f;
      const xLE = Math.abs(y) * Math.tan((sweepDeg * Math.PI) / 180);
      const zUp = vertical ? 0
                : Math.abs(y) * Math.tan((dihedralDeg * Math.PI) / 180);
      const x0 = xLE + front * c, x1 = xLE + rear * c;
      const h = 0.80 * tau * c;                  // clear of the skin
      const ring = [];
      const put = (x, v) => ring.push(
        vertical ? [x, Math.abs(y), v] : [x, zUp + v, y]);
      if (kind === 'capUpper') {
        put(x0,  h / 2);        put(x1,  h / 2);
        put(x1,  h / 2 - tc);   put(x0,  h / 2 - tc);
      } else if (kind === 'capLower') {
        put(x0, -h / 2 + tc);   put(x1, -h / 2 + tc);
        put(x1, -h / 2);        put(x0, -h / 2);
      } else if (kind === 'webFore') {
        put(x0,  h / 2);        put(x0 + tw,  h / 2);
        put(x0 + tw, -h / 2);   put(x0, -h / 2);
      } else {                                   // webAft
        put(x1 - tw,  h / 2);   put(x1,  h / 2);
        put(x1, -h / 2);        put(x1 - tw, -h / 2);
      }
      secs.push(ring);
    }
    return loft(secs, kind.startsWith('cap') ? capMat : webMat);
  };

  for (const sign of sides)
    for (const kind of ['capUpper', 'capLower', 'webFore', 'webAft'])
      g.add(plate(sign, kind));
  return g;
}

// ---------------------------------------------------------------------------
// fuselage
// ---------------------------------------------------------------------------
function fuseSection(x, R, dR, wdb, n = 40) {
  // components/fuselage.py: hfuse == Rfuse + 0.5*dRfuse and wfuse <= Rfuse +
  // wdb, so the section is not an ellipse. It is a capsule -- circular caps of
  // radius R pulled apart vertically by dR_fuse -- and on a double bubble the
  // caps are also pulled apart sideways by w_db, giving two lobes. Height is
  // 2R + dR, width 2(R + w_db).
  const ring = [];
  for (let i = 0; i < n; i++) {
    const th = (2 * Math.PI * i) / n;
    const c = Math.cos(th), sn = Math.sin(th);
    const zc = wdb * (c >= 0 ? 1 : -1);      // which lobe
    const yc = (dR / 2) * (sn >= 0 ? 1 : -1);  // which cap
    ring.push([x, yc + R * sn, zc + R * c]);
  }
  return ring;
}

function fuselage(v, arch, material) {
  const L = V(v, 'Fuse_l_fuse', 40);
  // The section profile is defined ONCE here and published on the returned
  // object. It used to be copy-pasted into cabin() and the flight deck, and
  // when the nose was reshaped those copies kept the old curve -- so the
  // windscreen was mapped onto a surface that no longer existed and stood
  // 12 cm off the skin.
  const R = V(v, 'Fuse_R_fuse', 1.9);
  const dR = V(v, 'Fuse_dR_fuse', 0);
  const wdb = arch.double_bubble ? V(v, 'Fuse_w_db', 0) : 0;
  const w = 2 * (R + wdb), h = 2 * R + dR;    // full width, full height
  const lNose = V(v, 'Fuse_l_nose', 0.12 * L);
  const lCone = V(v, 'Fuse_l_cone', 0.3 * L);
  const lShell = Math.max(0.1, L - lNose - lCone);

  const secs = [];
  const N = 22;
  // Nose: an ogive rather than a half-ellipse, dropped toward the tip. A
  // radome droops -- the cockpit roof stays high while the underside falls
  // away -- which is most of what makes a nose read as one.
  for (let i = 0; i <= N; i++) {
    const fr = i / N, x = fr * lNose;
    const sc = Math.pow(Math.max(1e-4, 1 - (1 - fr) ** 2), 0.62);
    const droop = -0.30 * R * (1 - fr) ** 2.2;
    secs.push(fuseSection(x, R * sc, dR * sc, wdb * sc)
              .map(([X, Y, Z]) => [X, Y + droop, Z]));
  }
  secs.push(fuseSection(lNose + lShell, R, dR, wdb));
  // Tailcone: upswept, tapering to a slim APU exhaust rather than a stub.
  for (let i = 1; i <= N; i++) {
    const fr = i / N, x = lNose + lShell + fr * lCone;
    const sc = Math.max(0.055, 1 - 0.945 * fr ** 1.45);
    const lift = 0.42 * (h / 2) * fr ** 2;
    secs.push(fuseSection(x, R * sc, dR * sc, wdb * sc)
              .map(([X, Y, Z]) => [X, Y + lift, Z]));
  }
  const scaleAt = (x) => {
    if (x <= lNose) {
      const fr = clamp(x / Math.max(lNose, 1e-6), 0, 1);
      return Math.pow(Math.max(1e-4, 1 - (1 - fr) ** 2), 0.62);
    }
    if (x <= lNose + lShell) return 1;
    const fr = clamp((x - lNose - lShell) / Math.max(lCone, 1e-6), 0, 1);
    return Math.max(0.055, 1 - 0.945 * fr ** 1.45);
  };
  const droopAt = (x) => (x <= lNose
    ? -0.30 * R * (1 - clamp(x / Math.max(lNose, 1e-6), 0, 1)) ** 2.2 : 0);
  const wAtS = (y, sc) => {
    const Rs = R * sc, dRs = dR * sc, ws = wdb * sc;
    const a = Math.max(0, Math.abs(y) - dRs / 2);
    return ws + (a >= Rs ? 0 : Math.sqrt(Math.max(0, Rs * Rs - a * a)));
  };
  return { mesh: loft(secs, material), L, R, dR, wdb, w, h,
           lNose, lShell, lCone, hdb: wdb, scaleAt, droopAt, wAtS };
}

// ---------------------------------------------------------------------------
// cabin -- dimensions from the solve, everything about how it LOOKS invented
// ---------------------------------------------------------------------------
/** One economy seat: S-profile shell on a pedestal, facing FORWARD (-x).
 *
 *  x is aft, so a passenger looks toward -x and the backrest is at the +x end.
 *  It was the other way round, which put 180 people facing the tail.
 *
 *  The back is four slabs whose lean walks from upright at the base through a
 *  recline and back toward vertical at the shoulder -- the S an airline seat
 *  has in side view. Cheaper than lofting a real shell and reads the same at
 *  this scale.
 */
function seatGeometry(pitch, sw) {
  const pan = clamp(sw * 0.90, 0.32, 0.56);
  const D = clamp(pitch * 0.50, 0.32, 0.56);
  const hSeat = 0.40;                            // cushion top above the floor
  const parts = [];

  // cushion, dished slightly nose-down at the front lip
  parts.push({ geom: new THREE.BoxGeometry(D * 0.92, 0.09, pan),
               pos: [D * 0.02, hSeat, 0], rot: [0, 0, 0.04] });

  // backrest: four slabs, leaning aft then easing upright -> an S
  const tilts = [-0.06, -0.20, -0.26, -0.15];
  const segH = 0.165, tBack = 0.055;
  let bx = D * 0.40, by = hSeat + 0.03;
  for (let i = 0; i < tilts.length; i++) {
    const a = tilts[i];
    parts.push({
      geom: new THREE.BoxGeometry(tBack, segH, pan * (1 - 0.05 * i)),
      pos: [bx - Math.sin(a) * segH / 2, by + Math.cos(a) * segH / 2, 0],
      rot: [0, 0, a],
    });
    bx -= Math.sin(a) * segH;
    by += Math.cos(a) * segH;
  }
  // headrest, tipped forward off the top of the back
  parts.push({ geom: new THREE.BoxGeometry(tBack * 1.4, 0.15, pan * 0.62),
               pos: [bx - 0.02, by + 0.07, 0], rot: [0, 0, 0.12] });

  // armrests
  for (const z of [pan / 2 + 0.035, -pan / 2 - 0.035])
    parts.push({ geom: new THREE.BoxGeometry(D * 0.72, 0.045, 0.05),
                 pos: [D * 0.02, hSeat + 0.17, z] });

  // pedestal and foot rail, not four table legs
  parts.push({ geom: new THREE.BoxGeometry(D * 0.20, hSeat - 0.06, pan * 0.34),
               pos: [D * 0.06, (hSeat - 0.06) / 2, 0] });
  parts.push({ geom: new THREE.BoxGeometry(D * 0.72, 0.05, 0.05),
               pos: [D * 0.02, 0.04, 0] });
  return merge(parts);
}

/**
 * Floor, seats in banks with an aisle between them, overhead bins, sidewall
 * lining and windows.
 *
 * Sized: floor length and width, row count, passenger count -- so the abreast
 * count, the seat pitch and the seat width all follow from the solve rather
 * than from taste. How a seat looks, and the bins and lining entirely, do not.
 */
function cabin(v, f, M) {
  const g = new THREE.Group();
  const notes = [];

  /** Half-width of the section at height y, for a section scaled by `sc`. */
  const wAtS = f.wAtS;
  const wAt = (y) => wAtS(y, 1);
  const normalAt = (y) => {
    const a = Math.abs(y) - f.dR / 2;
    if (a <= 0) return { ny: 0, nz: 1 };
    const nz = Math.sqrt(Math.max(1e-6, f.R * f.R - a * a));
    const m = Math.hypot(a, nz);
    return { ny: Math.sign(y) * a / m, nz: nz / m };
  };
  const scaleAt = f.scaleAt;

  // ---- the floor ----------------------------------------------------------
  // One deck at the vertical mid point of the fuselage, running the whole
  // length and filling whatever the section gives it. Height and width are no
  // longer computed from h_hold and w_floor: the first was inconsistent with
  // the section and the second is only true at the centreline, and a deck that
  // simply fills the body is both what the aeroplane has and what reads.
  // 40% of the section height, not the mid point.
  const yFloor = -f.h / 2 + 0.40 * f.h;
  const head = f.h / 2 - yFloor;
  const hBeam = V(v, 'Fuse_h_floor', 0.2);

  const rows = Math.round(V(v, 'Fuse_n_rows', 0));
  const nPass = Math.round(V(v, 'Fuse_n_pass', 0));

  // Cabin stations: entry area behind the flight deck, then the rows, then an
  // aft vestibule, then the pressure bulkhead.
  const xEntry = f.lNose + 0.6;
  const xSeat0 = xEntry + 1.9;
  const xShellEnd = f.lNose + f.lShell;
  const pitchSolve = rows > 0 ? V(v, 'Fuse_l_floor', f.lShell) / rows : 0.8;
  const room = Math.max(1, xShellEnd - 1.6 - xSeat0);
  const pitch = rows > 0 ? Math.min(pitchSolve, room / rows) : 0.8;
  const xSeatEnd = xSeat0 + rows * pitch;
  const xBulk = Math.min(xSeatEnd + 1.5, xShellEnd + 0.4);

  // Rear pressure bulkhead: a dome bulging AFT, spanning the FULL section --
  // radius R at that station, on the section centreline, not the half-width at
  // floor height. It closes the whole barrel, so the floor runs under it.
  // A double bubble gets one dome per lobe, which is what the section is.
  const Rb = f.R * scaleAt(xBulk);
  {
    const wdbB = f.wdb * scaleAt(xBulk);
    for (const zc of (wdbB > 1e-6 ? [wdbB, -wdbB] : [0])) {
      const dome = new THREE.Mesh(
        new THREE.SphereGeometry(Rb, 28, 16, 0, Math.PI * 2, 0, Math.PI / 2),
        M.lining);
      dome.rotation.z = -Math.PI / 2;         // convex toward the tail
      // A sphere of radius R fills the width but not the height: the section
      // is a capsule, 2R + dR tall. Stretch the dome vertically by that much
      // so it closes the whole barrel. After the rotation the dome's local x
      // is world y, so that is the axis to scale.
      dome.scale.x = (f.R + f.dR / 2) / f.R;
      dome.position.set(xBulk, 0, zc);
      dome.userData.layer = 'structure';
      g.add(dome);
    }
  }

  // The deck runs the whole length -- right up to the nose, and on under the
  // dome to its apex, since the dome bulges aft over it.
  {
    const secs = [];
    const xa = 0.02 * f.L, xb = xBulk + Rb * 0.92, n = 40;
    for (let i = 0; i <= n; i++) {
      const x = xa + (xb - xa) * (i / n);
      const w = Math.max(0.02, wAtS(yFloor, scaleAt(x)) * 0.97);
      secs.push([[x, yFloor + hBeam / 2, -w], [x, yFloor + hBeam / 2, w],
                 [x, yFloor - hBeam / 2, w], [x, yFloor - hBeam / 2, -w]]);
    }
    const slab = loft(secs, M.floor);
    slab.userData.layer = 'structure';
    g.add(slab);
  }

  if (rows <= 0 || nPass <= 0)
    return { group: g, notes, rows: 0, x0: xSeat0, lFloor: 0,
             wFloor: 2 * wAt(yFloor), yFloor, zSkin: f.w / 2, head,
             wCab: 2 * wAt(yFloor), normalAt, wAt, wAtS, scaleAt,
             xEntry, xSeat0, xSeatEnd, xBulk };

  const abreast = Math.max(1, Math.round(nPass / rows));
  const W_SEAT = 0.50, W_AISLE = 0.51, W_SYS = 0.10;
  const nAisle = abreast >= 7 ? 2 : 1;
  const banks = nAisle === 1
    ? [Math.ceil(abreast / 2), Math.floor(abreast / 2)]
    : [Math.floor(abreast / 4), abreast - 2 * Math.floor(abreast / 4),
       Math.floor(abreast / 4)];

  const need = abreast * W_SEAT + nAisle * W_AISLE + 2 * W_SYS;
  const have = 2 * wAt(yFloor);
  const fit = Math.min(1, have / need);
  const aisleW = W_AISLE * fit, sw = W_SEAT * fit;
  const cabW = abreast * sw + nAisle * aisleW;

  const zs = [];
  let z = -cabW / 2 + sw / 2;
  banks.forEach((n, bi) => {
    for (let i = 0; i < n; i++) { zs.push(z); z += sw; }
    if (bi < banks.length - 1) z += aisleW;
  });

  const SEAT_H = 1.28;
  const vFit = clamp((head - 0.12) / SEAT_H, 0.25, 1);
  const seats = new THREE.InstancedMesh(seatGeometry(pitch, sw), M.seat,
                                        rows * zs.length);
  const o = new THREE.Object3D();
  let k = 0;
  for (let r = 0; r < rows; r++)
    for (const zz of zs) {
      o.position.set(xSeat0 + (r + 0.5) * pitch, yFloor + hBeam / 2, zz);
      o.rotation.set(0, 0, 0);
      o.scale.set(1, vFit, 1);
      o.updateMatrix();
      seats.setMatrixAt(k++, o.matrix);
    }
  seats.instanceMatrix.needsUpdate = true;
  seats.userData.layer = 'payload';
  g.add(seats);

  // ---- overhead bins: an omega in section ---------------------------------
  const binH = clamp(0.42, 0.18, head * 0.26);
  const binW = clamp(sw * 1.55, 0.44, 0.82);
  const yBin = yFloor + Math.min(1.62, head - binH * 0.62);
  const binProfile = (w, h) => {
    const p2 = new THREE.Shape();
    p2.moveTo(w * 0.46, h * 0.50);
    p2.quadraticCurveTo(w * 0.56, h * 0.10, w * 0.44, -h * 0.30);
    p2.quadraticCurveTo(w * 0.30, -h * 0.56, 0, -h * 0.54);
    p2.quadraticCurveTo(-w * 0.34, -h * 0.52, -w * 0.46, -h * 0.20);
    p2.quadraticCurveTo(-w * 0.54, h * 0.06, -w * 0.40, h * 0.22);
    p2.lineTo(-w * 0.30, h * 0.16);
    p2.quadraticCurveTo(-w * 0.36, h * 0.02, -w * 0.30, -h * 0.20);
    p2.quadraticCurveTo(-w * 0.14, -h * 0.40, 0, -h * 0.38);
    p2.quadraticCurveTo(w * 0.24, -h * 0.38, w * 0.32, -h * 0.16);
    p2.quadraticCurveTo(w * 0.42, h * 0.14, w * 0.34, h * 0.46);
    p2.closePath();
    return p2;
  };
  const lBin = xSeatEnd - xSeat0;
  const binGeom = new THREE.ExtrudeGeometry(binProfile(binW, binH), {
    depth: lBin, bevelEnabled: false, curveSegments: 8 });
  binGeom.rotateY(Math.PI / 2);
  const binOut = Math.min(cabW / 2 + binW * 0.5, wAt(yBin)) - binW * 0.50;
  for (const s2 of (banks.length === 3 ? [1, -1, 0] : [1, -1])) {
    const bin = new THREE.Mesh(binGeom, M.bin);
    bin.position.set(xSeat0, yBin, s2 * binOut);
    if (s2 < 0) bin.scale.z = -1;
    bin.userData.layer = 'cabin';
    g.add(bin);
  }

  // ---- sidewall lining ----------------------------------------------------
  for (const s2 of [1, -1]) {
    const panelH = Math.min(1.7, head * 0.92);
    const panel = new THREE.Mesh(
      new THREE.BoxGeometry(xSeatEnd - xSeat0, panelH, 0.04), M.lining);
    panel.position.set((xSeat0 + xSeatEnd) / 2, yFloor + panelH / 2,
                       s2 * (wAt(yFloor + panelH / 2) - 0.05));
    panel.userData.layer = 'cabin';
    g.add(panel);
  }

  // ---- cabin windows: punched, rounded, one per row ------------------------
  // Cabin windows: patches OF the skin, one per row per side. Merged into one
  // geometry -- they are no longer identical prisms, so they cannot be
  // instanced, but 60 patches of ~56 triangles is nothing.
  const yWin = yFloor + Math.min(1.05, head * 0.62);
  const zSkin = wAt(yWin);
  const winPos = [];
  for (let r = 0; r < rows; r++)
    for (const s2 of [1, -1]) {
      const patch = skinPatch(xSeat0 + (r + 0.5) * pitch, yWin,
                              0.26, 0.36, 0.10, s2, wAtS, scaleAt);
      winPos.push(...patch.attributes.position.array);
    }
  const winGeom = new THREE.BufferGeometry();
  winGeom.setAttribute('position',
                       new THREE.Float32BufferAttribute(winPos, 3));
  winGeom.computeVertexNormals();
  const wins = new THREE.Mesh(winGeom, M.window);
  wins.userData.layer = 'cabin';
  wins.userData.kind = 'opening';
  g.add(wins);

  notes.push(
    `cabin: ${nPass} seats, ${rows} rows x ${abreast} abreast at ` +
    `${(100 * pitch).toFixed(0)} cm pitch, ${(100 * sw).toFixed(0)} cm seats, ` +
    `${(100 * aisleW).toFixed(0)} cm aisle${nAisle > 1 ? 's' : ''} ` +
    `(w_seat/w_aisle are model constants).`);
  notes.push(
    `floor is one deck at the vertical mid point, filling the section from ` +
    `x = ${(f.lNose * 0.3).toFixed(1)} m to the rear pressure bulkhead at ` +
    `x = ${xBulk.toFixed(1)} m. Height and width are geometric, not h_hold ` +
    `and w_floor -- those two disagree with the section.`);
  if (pitch < pitchSolve - 1e-3)
    notes.push(
      `NOTE: l_floor/rows gives ${(100 * pitchSolve).toFixed(0)} cm pitch, but ` +
      `l_floor (${V(v, 'Fuse_l_floor', 0).toFixed(1)} m) is longer than the ` +
      `constant section (${f.lShell.toFixed(1)} m), so the rows are drawn at ` +
      `${(100 * pitch).toFixed(0)} cm to fit between the entry area and the ` +
      `bulkhead.`);
  if (fit < 0.995)
    notes.push(
      `NOTE: the row needs ${need.toFixed(2)} m, the deck is ` +
      `${have.toFixed(2)} m, so seats and aisle are drawn ` +
      `${(100 * fit).toFixed(0)}% of nominal.`);
  if (vFit < 0.995)
    notes.push(`NOTE: ${head.toFixed(2)} m of headroom, seats at ` +
               `${(100 * vFit).toFixed(0)}% height.`);
  notes.push(
    'seat shape, bins, lining, windows and doors are furnishing -- nothing in ' +
    'the sizing model describes them.');
  return { group: g, notes, rows, abreast, pitch, x0: xSeat0,
           lFloor: xSeatEnd - xSeat0, wFloor: 2 * wAt(yFloor), zSkin, yFloor,
           head, wCab: cabW, normalAt, wAt, wAtS, scaleAt,
           xEntry, xSeat0, xSeatEnd, xBulk };
}

/**
 * The wing-body fairing: the blister under the belly that covers the centre
 * section and the gear bays. It is one of the largest things on the outside of
 * an airliner and the model says nothing about it, but without it the wing
 * appears to stab straight into a bare tube.
 */
function bellyFairing(f, xWing, cRoot, yWing, material) {
  const x0 = xWing - 0.55 * cRoot, x1 = xWing + 1.75 * cRoot;
  const secs = [];
  const n = 20;
  for (let i = 0; i <= n; i++) {
    const t = i / n, x = x0 + (x1 - x0) * t;
    // fullness peaks just aft of mid-chord, tapering to nothing at each end
    const bulge = Math.sin(Math.PI * Math.pow(t, 0.85)) ** 1.25;
    const wz = (f.R + f.wdb) * (1 + 0.20 * bulge);
    const yTop = yWing + 0.35 * f.R;
    const yBot = -f.h / 2 - 0.30 * f.R * bulge;
    const ring = [];
    const m = 22;
    for (let k = 0; k < m; k++) {
      const th = Math.PI * (k / (m - 1)) - Math.PI / 2;   // -90..+90, one side
      ring.push([x, yTop + (yBot - yTop) * Math.cos(th) ** 2,
                 wz * Math.sin(th)]);
    }
    secs.push(ring);
  }
  return loft(secs, material);
}

/** A canted winglet at the tip, swept back like the real thing. */
function winglet(f, { span, cRoot, cTip, tau, sweepDeg, dihedralDeg }, material,
                 up = true) {
  const semi = span / 2;
  const hW = clamp(0.085 * span, 0.6, 2.6);
  const g = new THREE.Group();
  for (const sgn of [1, -1]) {
    const w = surface({ span: hW, cRoot: cTip, cTip: cTip * 0.38, tau,
                        sweepDeg: 42, vertical: true }, material);
    w.position.set(semi * Math.tan((sweepDeg * Math.PI) / 180),
                   semi * Math.tan((dihedralDeg * Math.PI) / 180),
                   sgn * semi);
    w.rotation.x = sgn * (up ? -0.28 : 2.86);      // cant outboard
    g.add(w);
  }
  return g;
}

/** Flap-track fairings: the pods under the trailing edge. */
function flapTracks(f, { span, cRoot, cTip, sweepDeg, dihedralDeg }, material) {
  const g = new THREE.Group();
  const semi = span / 2;
  for (const fr of [0.22, 0.40, 0.58, 0.76])
    for (const sgn of [1, -1]) {
      const y = fr * semi;
      const c = cRoot + (cTip - cRoot) * fr;
      const xLE = y * Math.tan((sweepDeg * Math.PI) / 180);
      const zUp = y * Math.tan((dihedralDeg * Math.PI) / 180);
      const len = c * 0.55, rad = c * 0.055;
      const pod = new THREE.Mesh(
        new THREE.CapsuleGeometry(rad, len, 4, 10), material);
      pod.rotation.z = Math.PI / 2;
      pod.position.set(xLE + 0.78 * c, zUp - 0.055 * c, sgn * y);
      g.add(pod);
    }
  return g;
}

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------
/**
 * A turbofan with a fan, a core and a nozzle rather than a smooth tube.
 *
 * Sized: fan diameter, and the engine's own station areas -- A_25 at the LPC
 * face, A_5 at the core nozzle, A_7 at the bypass nozzle -- which set the hub
 * and exit diameters from the cycle rather than by eye. Cowl profile, blade
 * count and pylon shape are furnishing.
 */
function engine(v, dFan, len, M, { pylon = 0, blades = 20 } = {}) {
  const g = new THREE.Group();
  const rFan = dFan / 2;
  const dFromA = (A, d) => (A > 0 ? 2 * Math.sqrt(A / Math.PI) : d);
  const rHub = clamp(0.5 * dFromA(V(v, 'Eng_A_25', 0), 0.32 * dFan),
                     0.12 * rFan, 0.5 * rFan);
  const rCore = clamp(0.5 * dFromA(V(v, 'Eng_A_5', 0), 0.45 * dFan),
                      0.18 * rFan, 0.66 * rFan);
  const rByp = clamp(0.5 * dFromA(V(v, 'Eng_A_7', 0), 0.85 * dFan),
                     0.55 * rFan, 1.0 * rFan);

  const lipL = 0.12 * len;
  const lip = new THREE.Mesh(
    new THREE.CylinderGeometry(rFan * 1.06, rFan * 1.15, lipL, 28, 1, true),
    M.nacelle);
  lip.rotation.z = Math.PI / 2;
  lip.position.x = -len / 2 + lipL / 2;
  g.add(lip);

  const cowl = new THREE.Mesh(
    new THREE.CylinderGeometry(rFan * 1.15, rByp * 1.05, len * 0.74, 28, 1, true),
    M.nacelle);
  cowl.rotation.z = Math.PI / 2;
  cowl.position.x = -len / 2 + lipL + len * 0.37;
  g.add(cowl);

  const spinner = new THREE.Mesh(
    new THREE.ConeGeometry(rHub * 1.05, rHub * 2.0, 16), M.hub);
  spinner.rotation.z = Math.PI / 2;
  spinner.position.x = -len / 2 + lipL + rHub * 0.5;
  g.add(spinner);

  const bl = Math.max(0.05, rFan - rHub);         // fan blades on a disc
  const bladeParts = [];
  for (let i = 0; i < blades; i++) {
    const th = (2 * Math.PI * i) / blades;
    bladeParts.push({
      geom: new THREE.BoxGeometry(0.030 * len, bl * 0.96, bl * 0.30),
      pos: [0, Math.cos(th) * (rHub + bl / 2), Math.sin(th) * (rHub + bl / 2)],
      rot: [th, 0.55, 0],                         // stagger
    });
  }
  const fan = new THREE.Mesh(merge(bladeParts), M.blade);
  fan.position.x = -len / 2 + lipL + rHub * 1.2;
  g.add(fan);

  const core = new THREE.Mesh(
    new THREE.CylinderGeometry(rCore * 1.28, rCore, len * 0.42, 20, 1, true),
    M.core);
  core.rotation.z = Math.PI / 2;
  core.position.x = len * 0.18;
  g.add(core);

  const plug = new THREE.Mesh(
    new THREE.ConeGeometry(rCore * 0.9, len * 0.26, 18), M.hub);
  plug.rotation.z = -Math.PI / 2;
  plug.position.x = len * 0.50;
  g.add(plug);

  if (pylon > 0) {
    const p = new THREE.Mesh(
      new THREE.BoxGeometry(len * 0.62, pylon, rFan * 0.22), M.nacelle);
    p.position.set(len * 0.08, pylon / 2 + rFan * 0.95, 0);
    g.add(p);
  }
  return g;
}

// ---------------------------------------------------------------------------
// landing gear
// ---------------------------------------------------------------------------
/** A strut with a wheel pair, at the tyre size the model chose. */
function gearLeg(len, dTyre, wTyre, dOleo, M) {
  const g = new THREE.Group();
  const rT = dTyre / 2;
  const strut = new THREE.Mesh(
    new THREE.CylinderGeometry(dOleo / 2, dOleo / 2 * 0.8, len, 12), M.gear);
  strut.position.y = -len / 2;
  g.add(strut);
  const axle = new THREE.Mesh(
    new THREE.CylinderGeometry(dOleo * 0.22, dOleo * 0.22, wTyre * 1.6, 10),
    M.gear);
  axle.rotation.x = Math.PI / 2;
  axle.position.y = -len;
  g.add(axle);
  for (const z of [wTyre * 0.62, -wTyre * 0.62]) {
    const t = new THREE.Mesh(new THREE.CylinderGeometry(rT, rT, wTyre, 18), M.tyre);
    t.rotation.x = Math.PI / 2;
    t.position.set(0, -len, z);
    g.add(t);
    const hub = new THREE.Mesh(
      new THREE.CylinderGeometry(rT * 0.45, rT * 0.45, wTyre * 1.04, 12), M.hub);
    hub.rotation.x = Math.PI / 2;
    hub.position.copy(t.position);
    g.add(hub);
  }
  return g;
}

// ---------------------------------------------------------------------------
// assembly
// ---------------------------------------------------------------------------
/** Three ways to look at it.
 *  `outer` -- the mould line as an aeroplane, internals hidden behind it.
 *  `ghost` -- the default: skin faint enough to read the cabin through.
 *  `xray`  -- everything faint, for reading structure against structure.
 */
const SKIN_OPACITY = { outer: 0.97, ghost: 0.13, xray: 0.06 };

const MATS = (mode = 'ghost') => ({
  skin:    mat(0xbcc6d4, SKIN_OPACITY[mode] ?? 0.13),
  cap:     mat(0xd55e00, 1),
  web:     mat(0xeea800, 1),
  floor:   mat(0x8899aa, 0.85),
  seat:    mat(0x37506b, 0.98),
  bin:     mat(0x9aa7b6, 0.55),
  lining:  mat(0xa8b4c2, 0.20),
  window:  mat(0x7fd4ff, 0.55, THREE.DoubleSide, { emissive: 0x123c52 }),
  door:    mat(0x6a747f, 0.5),
  nacelle: mat(0x9aa3ad, 0.42),
  core:    mat(0x6f7783, 0.85),
  blade:   mat(0xc9d2dc, 0.95, THREE.DoubleSide,
               { metalness: 0.5, roughness: 0.3 }),
  hub:     mat(0x4a5058, 0.95),
  tyre:    mat(0x24262a, 1),
  tank:    mat(0x009e73, 0.6),
  stack:   mat(0x0072b2, 0.72),
  pack:    mat(0xcc79a7, 0.78),
  gear:    mat(0x555a61, 0.95),
});

/**
 * Build the aircraft. Returns a THREE.Group whose children carry
 * `userData.layer` so the viewer can toggle them, and `userData.placed=false`
 * where the geometry is furnishing rather than a model output.
 */
export function build(values, arch, opts = {}) {
  const v = values;
  const mode = opts.skin ?? 'ghost';
  const M = MATS(mode);
  const ex = opts.exaggerate ?? 40;
  if (mode === 'xray')
    for (const k of Object.keys(M))
      if (k !== 'skin') {
        M[k].opacity = Math.min(M[k].opacity, 0.45);
        M[k].transparent = true;
        M[k].depthWrite = false;
      }

  const root = new THREE.Group();
  const add = (obj, layer, placed = true, note = '') => {
    obj.traverse(o => { if (!o.userData.layer) o.userData.layer = layer; });
    if (!obj.userData.layer) obj.userData.layer = layer;
    obj.userData.placed = placed;
    obj.userData.note = note;
    root.add(obj);
    return obj;
  };

  // ---- fuselage -----------------------------------------------------------
  const f = fuselage(v, arch, M.skin);
  add(f.mesh, 'oml');

  // ---- flight deck --------------------------------------------------------
  // On the same deck as the cabin, since there is only one now. Pilots sit
  // just aft of the windscreen with the glareshield in front of them, which
  // puts them where the nose is still wide enough to hold two seats abreast.
  // All furnishing: the sizing model stops at the pressure vessel.
  const deck = new THREE.Group();
  const deckScaleAt = f.scaleAt, deckWAtS = f.wAtS;
  const yDeck = -f.h / 2 + 0.40 * f.h;
  const halfAt = (x) => deckWAtS(yDeck, deckScaleAt(x));

  // put the pilots where the nose is ~55% of full width
  let xPilot = f.lNose * 0.55;
  for (let i = 0; i < 12 && halfAt(xPilot) < deckWAtS(yDeck, 1) * 0.55; i++)
    xPilot += f.lNose * 0.04;
  const seatZ = Math.min(0.42, halfAt(xPilot) * 0.45);

  const pilotG = seatGeometry(0.95, 0.52);
  for (const dz of [seatZ, -seatZ]) {
    const ps = new THREE.Mesh(pilotG, M.seat);
    ps.position.set(xPilot, yDeck + 0.10, dz);
    deck.add(ps);
  }
  // glareshield ahead of them
  const glare = new THREE.Mesh(
    new THREE.BoxGeometry(0.30, 0.18, seatZ * 2.6), M.bin);
  glare.position.set(xPilot - 0.75, yDeck + 0.72, 0);
  deck.add(glare);

  // Windscreen and side windows: openings IN the nose skin, like the cabin
  // windows, so they appear on the outside of the aeroplane. Flat panes
  // floating inside the radome were invisible from anywhere useful.
  const droopAt = f.droopAt;
  const yGlass = yDeck + 0.78;
  const glassPos = [];
  for (const s2 of [1, -1]) {
    // windscreen pair, wrapped round the nose
    glassPos.push(...skinPatch(xPilot - 0.92, yGlass, 0.62, 0.42, 0.09, s2,
                               deckWAtS, deckScaleAt, 0.012, droopAt)
                  .attributes.position.array);
    // eyebrow / side window just aft of it
    glassPos.push(...skinPatch(xPilot - 0.10, yGlass - 0.06, 0.50, 0.34, 0.10,
                               s2, deckWAtS, deckScaleAt, 0.012, droopAt)
                  .attributes.position.array);
  }
  const fwg = new THREE.BufferGeometry();
  fwg.setAttribute('position', new THREE.Float32BufferAttribute(glassPos, 3));
  fwg.computeVertexNormals();
  const fwMesh = new THREE.Mesh(fwg, M.window);
  fwMesh.userData.kind = 'opening';
  deck.add(fwMesh);
  add(deck, 'cabin', false,
      'flight deck -- crew seats, glareshield, windscreen and forward side ' +
      'windows are furnishing: the sizing model stops at the pressure vessel');

  // ---- cabin --------------------------------------------------------------
  const cab = cabin(v, f, M);
  const yFloor = cab.yFloor;
  add(cab.group, 'payload', true, cab.notes.join('  '));

  if (cab.rows > 0) {
    // ---- cabin bulkheads ---------------------------------------------------
    // A bulkhead closing the cabin off from the entry area, and another at the
    // aft end. Furnishing; the model sizes no cabin layout. (Galley and
    // lavatory monuments removed at your request.)
    const bulks = new THREE.Group();
    const hCab = Math.min(1.95, cab.head * 0.95);
    for (const bx of [cab.xSeat0 - 0.35, cab.xSeatEnd + 0.35]) {
      const bulk = new THREE.Mesh(
        new THREE.BoxGeometry(0.06, hCab, cab.wCab * 0.98), M.lining);
      bulk.position.set(bx, cab.yFloor + hCab / 2, 0);
      bulks.add(bulk);
    }
    add(bulks, 'cabin', false,
        'cabin bulkheads are furnishing: the model sizes no cabin layout');
  }

  if (cab.rows > 0) {
    // Doors, like the windows, are patches OF the skin -- curved with the
    // body rather than flat plates laid against it.
    const dh = Math.min(1.85, cab.head * 0.92), dw = 0.85;
    const yDoor = cab.yFloor + dh / 2;
    const xDoors = [(cab.xEntry + cab.xSeat0) / 2,
                    (cab.xSeatEnd + cab.xBulk) / 2];
    const dPos = [];
    for (const s2 of [1, -1])
      for (const xd of xDoors)
        dPos.push(...skinPatch(xd, yDoor, dw, dh, 0.28, s2,
                               cab.wAtS, cab.scaleAt, 0.012)
                  .attributes.position.array);
    const dg = new THREE.BufferGeometry();
    dg.setAttribute('position', new THREE.Float32BufferAttribute(dPos, 3));
    dg.computeVertexNormals();
    const doorMesh = new THREE.Mesh(dg, M.door);
    doorMesh.userData.kind = 'opening';
    add(doorMesh, 'cabin', false,
        'doors are furnishing: the model does not place them');
  }

  // ---- wing ---------------------------------------------------------------
  const b = V(v, 'Wing_b', 30), cr = V(v, 'Wing_c_root', 5);
  const ct = V(v, 'Wing_c_tip', 1.2), tau = V(v, 'Wing_tau', 0.12);
  const xWing = V(v, 'Fuse_x_wing', V(v, 'Wing_x_w', 0.45 * f.L));
  const sweep = opts.sweepDeg ?? 25;
  const dihedral = 5;
  const wingGeom = { span: b, cRoot: cr, cTip: ct, tau, sweepDeg: sweep,
                     dihedralDeg: dihedral };
  // The wing box passes UNDER the cabin floor -- that is what a low-wing
  // airliner's centre section does, and drawing it on the waterline put the
  // spar caps straight through the seated passengers. Drop it so the box top
  // clears the floor, which pushes the root fairing a little below the belly,
  // as it is on the real thing.
  const yWing = Math.min(-0.12 * f.h,
                         yFloor - 0.5 * tau * cr - V(v, 'Fuse_h_floor', 0.2));
  const wing = surface(wingGeom, M.skin);
  wing.position.set(xWing, yWing, 0);
  add(wing, 'oml');

  add(bellyFairing(f, xWing, cr, yWing, M.skin), 'oml', false,
      'wing-body fairing: every airliner has one and no sizing model here ' +
      'describes it, so its shape is drawn to cover the centre section and ' +
      'the gear bays');
  // Both of these are built in WING coordinates -- station measured from the
  // root leading edge -- so they have to be offset onto the wing. Added
  // straight to the root they hung off the nose.
  const wl = winglet(f, wingGeom, M.skin);
  wl.position.set(xWing, yWing, 0);
  add(wl, 'oml', false,
      'winglets are furnishing: the model sizes a planform, not a tip device');
  const ft = flapTracks(f, wingGeom, M.nacelle);
  ft.position.set(xWing, yWing, 0);
  add(ft, 'oml', false, 'flap-track fairings are furnishing');

  const wbox = box({ ...wingGeom, tCap: V(v, 'Wing_box_t_cap', 0.003),
                     tWeb: V(v, 'Wing_box_t_web', 0.001), exaggerate: ex },
                   M.cap, M.web);
  wbox.position.copy(wing.position);
  add(wbox, 'structure', true,
      `wing box: cap ${(1000 * V(v, 'Wing_box_t_cap', 0)).toFixed(2)} mm, ` +
      `web ${(1000 * V(v, 'Wing_box_t_web', 0)).toFixed(2)} mm (x${ex} for legibility)`);

  // ---- tails --------------------------------------------------------------
  const xTail = f.lNose + f.lShell + 0.55 * f.lCone;
  const piTail = !!arch.double_bubble;

  const vtSpan = V(v, 'VT_b_vt', 5), vtRoot = V(v, 'VT_c_root_vt', 3);
  const vtGeom = { span: vtSpan, cRoot: vtRoot,
                   cTip: V(v, 'VT_c_tip_vt', vtRoot * 0.4), tau: 0.10,
                   sweepDeg: 35, vertical: true };
  const fins = new THREE.Group(), finBoxes = new THREE.Group();
  for (const z of piTail ? [f.w * 0.42, -f.w * 0.42] : [0]) {
    const fin = surface(vtGeom, M.skin);
    fin.position.set(xTail - 1, 0.30 * f.h, z);
    fins.add(fin);
    const fb = box({ ...vtGeom, tCap: V(v, 'VT_box_t_cap', 0.001),
                     tWeb: V(v, 'VT_box_t_web', 0.001), exaggerate: ex },
                   M.cap, M.web);
    fb.position.copy(fin.position);
    finBoxes.add(fb);
  }
  add(fins, 'oml');
  add(finBoxes, 'structure');

  const htGeom = { span: V(v, 'HT_b_ht', 10), cRoot: V(v, 'HT_c_root_ht', 3),
                   cTip: V(v, 'HT_c_tip_ht', V(v, 'HT_c_root_ht', 3) * 0.35),
                   tau: 0.10, sweepDeg: piTail ? 8 : 30,
                   dihedralDeg: piTail ? 0 : 6 };
  const ht = surface(htGeom, M.skin);
  ht.position.set(piTail ? xTail - 0.6 : xTail,
                  piTail ? 0.30 * f.h + vtSpan : 0.22 * f.h, 0);
  add(ht, 'oml');
  const htb = box({ ...htGeom, tCap: V(v, 'HT_box_t_cap', 0.001),
                    tWeb: V(v, 'HT_box_t_web', 0.001), exaggerate: ex },
                  M.cap, M.web);
  htb.position.copy(ht.position);
  add(htb, 'structure', true, piTail
      ? 'pi tail: fins on the fuselage shoulders, tailplane across their tips'
      : '');

  // ---- propulsion ---------------------------------------------------------
  const dFan = V(v, 'Eng_d_f', 1.5) || 1.5;
  // `l_nacelle` is the cowl length the drag model wants, which is shorter than
  // the whole installation: 1.23 m against a 1.36 m fan on the b737 case. A
  // fan that size needs about two diameters of nacelle to read as one.
  const lNac = Math.max(V(v, 'l_nacelle', 0), 1.85 * dFan);
  const nEng = Math.max(1, Math.round(V(v, 'n_eng', 2)));
  const props = new THREE.Group();

  if (arch.rear_engines) {
    for (let i = 0; i < nEng; i++) {
      const sgn = i % 2 === 0 ? 1 : -1;
      const e = engine(v, dFan, lNac, M);
      e.position.set(f.lNose + f.lShell + 0.26 * f.lCone, 0.34 * f.h,
                     sgn * (f.w * 0.22 + f.hdb * 0.4));
      props.add(e);
    }
    add(props, 'propulsion', true,
        'D8: engines on the aft fuselage between the pi-tail fins, ingesting ' +
        'the boundary layer. Station from the architecture, not from x_eng.');
  } else {
    // `y_eng` is the spanwise station the solve carries, but on the b737 case
    // it reads 1.03 m -- inboard of the fuselage side. It is inherited from
    // SPaircraft's rear-engine datum, so it is used only when it lands
    // somewhere an underwing pod could actually be.
    const yEng = V(v, 'y_eng', 0);
    const usable = yEng > f.w * 0.6 && yEng < b / 2;
    const frac = [0.34, 0.60];
    for (let i = 0; i < nEng; i++) {
      const sgn = i % 2 === 0 ? 1 : -1;
      const zEng = sgn * (usable ? yEng
                                 : frac[Math.floor(i / 2) % frac.length] * (b / 2));
      const cLocal = cr + (ct - cr) * (Math.abs(zEng) / (b / 2));
      const xLE = xWing + Math.abs(zEng) * Math.tan((sweep * Math.PI) / 180);
      const yLocal = yWing + Math.abs(zEng) * Math.tan((dihedral * Math.PI) / 180);
      // The nacelle hangs AHEAD of the leading edge, not under it: on a 737
      // the fan face is roughly a nacelle length forward of the LE and tucked
      // close underneath. Sitting it under the wing reads as an A400 pod.
      const drop = 0.52 * dFan + 0.06 * cLocal;
      const e = engine(v, dFan, lNac, M, { pylon: drop * 0.62 });
      e.position.set(xLE - 0.45 * lNac, yLocal - drop, zEng);
      props.add(e);
    }
    add(props, 'propulsion', usable, usable
      ? `${nEng} engines under the wing at y_eng = ${yEng.toFixed(2)} m, ` +
        `fan ${dFan.toFixed(2)} m.`
      : `${nEng} engines podded under the wing at ` +
        `${frac.slice(0, Math.ceil(nEng / 2)).map(x => (100 * x).toFixed(0) + '%').join(' and ')} ` +
        'semi-span. The solve carries y_eng from SPaircraft\'s rear-engine ' +
        'datum, which puts it inside the fuselage here, so the spanwise ' +
        'station is geometric. Fan diameter and the nozzle areas are the solve\'s.');
  }

  // ---- energy: sized by the model, placed by assumption -------------------
  const bayY = yFloor - 0.34 * f.h, bayW = f.w * 0.66, bayH = 0.40 * f.h;
  const bay = (len, x0, material, note) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(len, bayH, bayW), material);
    m.position.set(x0 + len / 2, bayY, 0);
    return add(m, 'energy', false, note);
  };
  if (arch.cryo_tank)
    bay(V(v, 'Tank_l_tank', 0) || 0.22 * f.L, f.lNose + f.lShell * 0.55, M.tank,
        'LH2 tank: length from the solve where available, position assumed ' +
        'under-floor aft');
  if (arch.fuel_cell)
    bay(0.10 * f.L, f.lNose + 0.10 * f.lShell, M.stack,
        'fuel-cell stack: volume not sized geometrically; block is indicative, ' +
        'position assumed under-floor forward');
  if (arch.battery)
    bay(0.35 * f.L, f.lNose + 0.12 * f.lShell, M.pack,
        'battery pack: mass is sized, geometry is not; position assumed ' +
        'under-floor');

  // ---- landing gear -------------------------------------------------------
  // The main gear hangs off the WING, at its rear spar, not off the fuselage
  // belly. Stations, leg lengths, track and tyre sizes are the solve's; the
  // tyre diameters are carried in inches while the widths are in metres.
  //
  // LG_y_m comes back at 1.03 m on the b737 case -- inboard of the fuselage
  // side, and less than a fifth of a real 737's 5.7 m track. It is used
  // anyway when it clears the fuselage, and pushed outboard to the wing root
  // when it does not, which is flagged.
  const gear = new THREE.Group();
  const lm = V(v, 'LG_l_m', 2) || 2, ln = V(v, 'LG_l_n', lm * 0.9) || lm * 0.9;
  const xm = V(v, 'LG_x_m', xWing + 1.5), xn = V(v, 'LG_x_n', f.lNose + 1);
  const dOleo = V(v, 'LG_d_oleo', 0.25) || 0.25;
  const dtm = tyreDia(V(v, 'LG_d_t_m', 0), 1.0);
  const dtn = tyreDia(V(v, 'LG_d_t_n', 0), 0.7);
  const wtm = V(v, 'LG_w_t_m', 0.4) || 0.4, wtn = V(v, 'LG_w_t_n', 0.3) || 0.3;

  const trackSolve = V(v, 'LG_y_m', 0) || V(v, 'LG_T', f.w) / 2;
  const minTrack = (f.R + f.wdb) * 0.85;         // clear of the fuselage side
  const track = Math.max(trackSolve, minTrack);
  const trackMoved = track > trackSolve + 1e-6;

  // the wing's lower surface at that spanwise station -- where the leg attaches
  const cAt = cr + (ct - cr) * clamp(track / (b / 2), 0, 1);
  const yAttach = yWing + track * Math.tan((dihedral * Math.PI) / 180)
                  - 0.45 * tau * cAt;
  const legLen = Math.max(lm, (yAttach - (yWing - 0.5 * f.h)) * 0.0 + lm);

  for (const s2 of [1, -1]) {
    const leg = gearLeg(legLen, dtm, wtm, dOleo, M);
    leg.position.set(xm, yAttach, s2 * track);
    gear.add(leg);
  }
  const nose = gearLeg(ln, dtn, wtn, dOleo * 0.7, M);
  nose.position.set(xn, -f.h / 2 + 0.05, 0);
  gear.add(nose);
  add(gear, 'gear', !trackMoved,
      `gear from the solve: main at x = ${xm.toFixed(1)} m, leg ` +
      `${lm.toFixed(2)} m, tyre ${dtm.toFixed(2)} m; nose at x = ` +
      `${xn.toFixed(1)} m. Legs hang from the wing lower surface at ` +
      `+/-${track.toFixed(2)} m.` +
      (trackMoved
        ? ` NOTE: LG_y_m is ${trackSolve.toFixed(2)} m, inboard of the ` +
          `fuselage side at ${(f.R + f.wdb).toFixed(2)} m, so the legs are ` +
          `moved out to the wing root. A real 737 tracks 5.7 m.`
        : ''));

  root.userData.extent = f.L;
  return root;
}

export const LAYERS = ['oml', 'structure', 'cabin', 'payload', 'propulsion',
                       'energy', 'gear'];
