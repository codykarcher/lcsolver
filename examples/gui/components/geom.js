/**
 * Geometry helpers shared across components.
 *
 * Nothing here knows what an aeroplane is. These are the two or three shapes
 * that keep reappearing -- a rod between two points, a body of revolution, a
 * ring of twisted blades -- factored out so each component file is about the
 * thing it models rather than about vertex buffers.
 *
 * Axis convention for anything with an axis: **+Z**. three.js lathes and
 * extrudes about Y, so the helpers rotate for you; callers work in Z.
 */
import * as THREE from 'three';

const UP = new THREE.Vector3(0, 1, 0);

/** A cylinder from point `a` to point `b`. Returns null for zero length. */
export function rod(a, b, radius, material, segments = 16) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-9) return null;
  const m = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, len, segments), material);
  m.position.copy(a).addScaledVector(dir, 0.5);
  m.quaternion.setFromUnitVectors(UP, dir.clone().normalize());
  return m;
}

/**
 * Force a surface of revolution to face outward.
 *
 * `LatheGeometry` winds its faces according to the direction the profile runs.
 * A profile given front-to-back -- the natural way to describe an engine, and
 * the way every profile in this library is written -- runs in *decreasing* z,
 * which is bottom-to-top reversed, so every face ends up wound inward. With
 * the default `FrontSide` material those faces are then culled and the solid
 * looks like an empty shell you can see straight through.
 *
 * Rather than make each caller order its profile to suit three.js, fix it
 * here, by signed volume -- see the comment in the body for why that and not
 * something simpler.
 */
export function orientOutward(geo) {
  const idx = geo.getIndex();
  const pos = geo.getAttribute('position');
  if (!idx || !pos) return geo;

  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = new THREE.Vector3(), cen = new THREE.Vector3();
  const arr = idx.array;

  // Signed volume, sum of a . (b x c) / 6 over the triangles. For a closed
  // mesh this is the enclosed volume, positive exactly when the winding faces
  // outward, and it does not care where the origin is. Every triangle votes in
  // proportion to its size, so no single face can decide the answer.
  //
  // The earlier version tested one triangle -- the outermost -- which is fine
  // on a plain tube but fails on a cone with a flat base: the base rim is as
  // far from the axis as anything, its normal is parallel to the axis, and the
  // radial component it is judged by is then zero plus rounding. That made the
  // decision flip with the size of the part.
  let vol = 0, radial = 0, scale = 0;
  for (let t = 0; t + 2 < arr.length; t += 3) {
    a.fromBufferAttribute(pos, arr[t]);
    b.fromBufferAttribute(pos, arr[t + 1]);
    c.fromBufferAttribute(pos, arr[t + 2]);
    vol += a.dot(n.crossVectors(b, c)) / 6;

    // Area-weighted radial vote, kept as a fallback for open shells where a
    // signed volume means nothing. n is twice the face area normal, so this
    // weights by area without normalising.
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
    n.crossVectors(b.clone().sub(a), c.clone().sub(a));
    const r = Math.hypot(cen.x, cen.z);           // axis is still Y here
    if (r > 1e-9) radial += (n.x * cen.x + n.z * cen.z) / r;
    scale = Math.max(scale, Math.abs(cen.x), Math.abs(cen.y), Math.abs(cen.z));
  }

  const closed = Math.abs(vol) > 1e-6 * Math.pow(scale || 1, 3);
  const inward = closed ? vol < 0 : radial < 0;

  if (inward) {
    for (let t = 0; t + 2 < arr.length; t += 3) {
      const tmp = arr[t + 1]; arr[t + 1] = arr[t + 2]; arr[t + 2] = tmp;
    }
    idx.needsUpdate = true;
    geo.computeVertexNormals();
  }
  return geo;
}

/**
 * A body of revolution about +Z.
 *
 * @param {Array<[number,number]>} profile  [[z, r], ...], in order along the
 *   axis, either direction. Start and end at r = 0 (or return to the first
 *   point) for a closed solid; leave an end at r > 0 and it is an open shell
 *   with a visible hole, which is almost never what you want.
 */
export function latheZ(profile, material, segments = 48) {
  const pts = profile.map(([z, r]) => new THREE.Vector2(Math.max(r, 1e-5), z));
  const geo = orientOutward(new THREE.LatheGeometry(pts, segments));
  const m = new THREE.Mesh(geo, material);
  m.rotation.x = Math.PI / 2;            // lathe axis Y -> Z
  return m;
}

/** A hollow tube (a case or a duct wall), axis +Z, from z0 to z1. */
export function tubeZ(rInner, rOuter, z0, z1, material, segments = 48) {
  return latheZ([[z0, rInner], [z0, rOuter], [z1, rOuter], [z1, rInner],
                 [z0, rInner]], material, segments);
}

/**
 * A box with rounded edges, centred on the origin, depth along +Z.
 *
 * Plain BoxGeometry is the wrong primitive for cast parts. A crankcase or a
 * cylinder head has no sharp arrises anywhere -- it was poured into sand --
 * and a hard 90-degree edge reads as a cardboard carton at any size. Rounding
 * costs a bevel and fixes it.
 *
 * Built as a rounded rectangle extruded with a bevel, so the corners are
 * rounded in plan AND the ends are eased.
 */
export function roundedBox(w, h, d, r, material) {
  r = Math.min(r, w / 2 - 1e-4, h / 2 - 1e-4);
  const b = Math.min(r * 0.6, d * 0.24, w / 4, h / 4);
  // bevelSize pushes the profile OUTWARD, so a shape built at full size comes
  // out 2b too wide and too tall. Shrink it by the bevel and the finished
  // solid is exactly w x h x d, which is what every caller assumes.
  const iw = w - 2 * b, ih = h - 2 * b;
  const ir = Math.max(1e-4, Math.min(r - b, iw / 2 - 1e-4, ih / 2 - 1e-4));
  const x = iw / 2 - ir, y = ih / 2 - ir;
  r = ir;

  const sh = new THREE.Shape();
  sh.moveTo(-x - r, -y);
  sh.lineTo(-x - r, y);
  sh.quadraticCurveTo(-x - r, y + r, -x, y + r);
  sh.lineTo(x, y + r);
  sh.quadraticCurveTo(x + r, y + r, x + r, y);
  sh.lineTo(x + r, -y);
  sh.quadraticCurveTo(x + r, -y - r, x, -y - r);
  sh.lineTo(-x, -y - r);
  sh.quadraticCurveTo(-x - r, -y - r, -x - r, -y);
  sh.closePath();

  const geo = new THREE.ExtrudeGeometry(sh, {
    depth: Math.max(1e-4, d - 2 * b), bevelEnabled: true,
    bevelThickness: b, bevelSize: b, bevelSegments: 3, curveSegments: 6,
  });
  geo.translate(0, 0, -(d - 2 * b) / 2);
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, material);
}

/**
 * A pipe following a path, capped so it is a closed solid.
 *
 * TubeGeometry leaves its ends open; a bead at each end closes them and
 * rounds the pipe off at the same time. Slightly over-sized, because a sphere
 * of exactly the tube radius meets the rim coincidentally and z-fights.
 */
export function pipe(points, radius, material, segments = 48) {
  const g = new THREE.Group();
  const curve = new THREE.CatmullRomCurve3(points, false, 'centripetal');
  g.add(new THREE.Mesh(
    new THREE.TubeGeometry(curve, segments, radius, 12, false), material));
  for (const p of [points[0], points[points.length - 1]]) {
    const cap = new THREE.Mesh(
      new THREE.SphereGeometry(radius * 1.04, 12, 8), material);
    cap.position.copy(p);
    g.add(cap);
  }
  return g;
}

/**
 * One twisted, tapered blade, lofted from a stack of elliptical sections.
 *
 * Built with the span along +Y, chord along Z and thickness along X, so a row
 * of them is made by rotating copies about Z. An ellipse is a crude aerofoil,
 * but at the size a fan blade appears on screen it is indistinguishable from a
 * real section, and it costs eight points a station.
 *
 * `twist` is in radians, positive meaning leading edge into the +Z side, and
 * is interpolated linearly root to tip -- which is what makes a fan read as a
 * fan rather than as a paddle.
 */
export function bladeGeometry({
  rHub, rTip, chordRoot, chordTip, twistRoot = 0.9, twistTip = 0.25,
  thickness = 0.10, nSpan = 8, nSec = 10,
}) {
  const pos = [];
  const idx = [];
  const lerp = (a, b, t) => a + (b - a) * t;

  for (let i = 0; i < nSpan; i++) {
    const t = i / (nSpan - 1);
    const y = lerp(rHub, rTip, t);
    const c = lerp(chordRoot, chordTip, t);
    const th = thickness * c;
    const b = lerp(twistRoot, twistTip, t);
    const cb = Math.cos(b), sb = Math.sin(b);
    for (let j = 0; j < nSec; j++) {
      const phi = (j / nSec) * Math.PI * 2;
      const zc = (c / 2) * Math.cos(phi);
      const xc = (th / 2) * Math.sin(phi);
      pos.push(xc * cb + zc * sb, y, -xc * sb + zc * cb);
    }
  }
  for (let i = 0; i < nSpan - 1; i++) {
    for (let j = 0; j < nSec; j++) {
      const a = i * nSec + j;
      const b = i * nSec + ((j + 1) % nSec);
      idx.push(a, b, a + nSec, b, b + nSec, a + nSec);
    }
  }
  // Cap root and tip with triangle fans so the blade is a closed solid.
  for (const [base, flip] of [[0, true], [(nSpan - 1) * nSec, false]]) {
    for (let j = 1; j < nSec - 1; j++) {
      if (flip) idx.push(base, base + j + 1, base + j);
      else idx.push(base, base + j, base + j + 1);
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/**
 * A ring of `count` blades about +Z, plus the hub they sit in.
 *
 * One geometry, `count` meshes -- blades are identical by construction and
 * there is no reason to build the buffer more than once.
 */
export function bladeRow({ count, material, hubMaterial, hubLength, ...blade }) {
  const g = new THREE.Group();
  const geo = bladeGeometry(blade);
  for (let i = 0; i < count; i++) {
    const m = new THREE.Mesh(geo, material);
    m.rotation.z = (i / count) * Math.PI * 2;
    g.add(m);
  }
  if (hubMaterial) {
    const hub = new THREE.Mesh(
      new THREE.CylinderGeometry(blade.rHub * 1.02, blade.rHub * 1.02,
                                 hubLength ?? blade.rHub * 1.6, 32),
      hubMaterial);
    hub.rotation.x = Math.PI / 2;
    g.add(hub);
  }
  g.userData.sharedGeometry = geo;
  return g;
}

/** Dispose every geometry under `obj`, including shared blade buffers. */
export function disposeTree(obj) {
  const seen = new Set();
  obj.traverse((o) => {
    if (o.isMesh && !seen.has(o.geometry)) { seen.add(o.geometry); o.geometry.dispose(); }
    const s = o.userData?.sharedGeometry;
    if (s && !seen.has(s)) { seen.add(s); s.dispose(); }
  });
}
