/**
 * Parametric landing gear.
 *
 * Four variants -- 1, 2, 4 and 6 wheels -- from one set of four parameters:
 *
 *     rWheel   rim radius, the wheel without its tire
 *     rTire    tire outer radius
 *     lStrut   strut length, measured from the attachment point to the axle
 *     rStrut   strut radius
 *
 * Everything else (tire width, axle diameter, bogie proportions, wheel
 * spacing) is derived from those four in PROPORTIONS below. Those derivations
 * are chosen to look right, not to mean anything -- there is no physics in
 * this file. When a real model has a number for one of them it should be
 * promoted to a parameter and removed from that table.
 *
 * Frame and origin
 * ----------------
 * The group's origin is the ATTACHMENT POINT, and the gear hangs down in -Y,
 * so a gear can be dropped straight onto a wing or fuselage station without
 * the caller working out where its top is. Axles run along X; a bogie beam
 * runs fore-and-aft along Z.
 *
 * Because the whole thing hangs below the origin, `group.userData.contactY`
 * gives the ground-contact height (a negative number) for anyone who needs to
 * stand the gear on a surface:
 *
 *     const g = landingGear({ wheels: 4 });
 *     g.position.y = -g.userData.contactY;   // now sitting on y = 0
 *
 * Variants
 * --------
 * 1 wheel   an inverted-U fork straddling a single wheel
 * 2 wheels  one axle, the strut meeting it between the wheels
 * 4 wheels  a fore-aft bogie beam carrying two axles, strut on a centre pivot
 * 6 wheels  the same with three axles
 */
import * as THREE from 'three';
import * as M from './materials.js';

/**
 * Visual proportions, all as fractions of a parameter. Not physical.
 */
const P = {
  tireWidth:    0.62,   // x rTire
  axleRadius:   0.42,   // x rStrut
  forkRadius:   0.40,   // x rStrut
  clearance:    0.12,   // x rTire -- gaps between parts
  pistonRadius: 0.78,   // x rStrut -- the bright inner cylinder
  pistonTravel: 0.45,   // fraction of the drawn strut that is piston
  beamWidth:    1.50,   // x rStrut -- bogie beam, across
  beamHeight:   1.15,   // x rStrut -- bogie beam, deep
  axleSpacing:  2.30,   // x rTire -- between bogie axles, fore-and-aft
  trunnion:     1.45,   // x rStrut -- attachment boss radius
  hubProud:     0.10,   // x rTire -- how far the hub stands out from the rim
};

const SEG = 32;         // radial segments on bodies of revolution

/* ------------------------------------------------------------------ *
 * Pieces
 * ------------------------------------------------------------------ */

/**
 * One wheel: tire, rim, hub. Axis along X, centred on the group origin.
 *
 * The tire is a lathe rather than a cylinder so it has rounded shoulders. That
 * single detail is most of what separates a wheel from a hockey puck.
 */
function wheel(rWheel, rTire, wTire) {
  const g = new THREE.Group();
  const half = wTire / 2;
  const sh = Math.min(0.30 * wTire, 0.45 * (rTire - rWheel));  // shoulder

  const prof = [];
  prof.push(new THREE.Vector2(rWheel, -half));
  prof.push(new THREE.Vector2(rTire - sh, -half));
  for (let i = 0; i <= 6; i++) {                    // shoulder, inboard
    const a = -Math.PI / 2 + (i / 6) * (Math.PI / 2);
    prof.push(new THREE.Vector2(rTire - sh + sh * Math.cos(a),
                                -half + sh + sh * Math.sin(a)));
  }
  for (let i = 0; i <= 6; i++) {                    // shoulder, outboard
    const a = (i / 6) * (Math.PI / 2);
    prof.push(new THREE.Vector2(rTire - sh + sh * Math.cos(a),
                                half - sh + sh * Math.sin(a)));
  }
  prof.push(new THREE.Vector2(rWheel, half));

  const tire = new THREE.Mesh(new THREE.LatheGeometry(prof, SEG), M.tire);
  tire.rotation.z = Math.PI / 2;                    // lathe axis Y -> X
  g.add(tire);

  const rim = new THREE.Mesh(
    new THREE.CylinderGeometry(rWheel, rWheel, wTire * 0.94, SEG), M.rim);
  rim.rotation.z = Math.PI / 2;
  g.add(rim);

  // A hub standing slightly proud on each face, so the wheel reads as having
  // a near and a far side rather than being a flat disc.
  const rHub = rWheel * 0.42;
  for (const s of [-1, 1]) {
    const hub = new THREE.Mesh(
      new THREE.CylinderGeometry(rHub, rHub * 0.86, rTire * P.hubProud, SEG),
      M.hardware);
    hub.rotation.z = Math.PI / 2;
    hub.position.x = s * (wTire * 0.47 + rTire * P.hubProud * 0.5);
    g.add(hub);
  }
  return g;
}

/** A cylinder from point a to point b, of the given radius and material. */
function rod(a, b, radius, material) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-9) return null;
  const m = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, len, Math.max(8, SEG / 2)),
    material);
  m.position.copy(a).addScaledVector(dir, 0.5);
  // Default cylinder axis is +Y; rotate it onto `dir`.
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0),
                                  dir.clone().normalize());
  return m;
}

/**
 * The strut, drawn from `yTop` down to `yBot`.
 *
 * Two concentric cylinders: a painted outer, and a polished piston emerging
 * from it. The bright band is the single cue that says "oleo" rather than
 * "pipe", which is why it is worth the extra mesh.
 */
function strut(yTop, yBot, rStrut) {
  const g = new THREE.Group();
  const len = yTop - yBot;
  if (len <= 1e-6) return g;

  const lPiston = len * P.pistonTravel;
  const lOuter = len - lPiston * 0.75;              // they overlap

  const outer = new THREE.Mesh(
    new THREE.CylinderGeometry(rStrut, rStrut, lOuter, SEG), M.structure);
  outer.position.y = yTop - lOuter / 2;
  g.add(outer);

  const rP = rStrut * P.pistonRadius;
  const piston = new THREE.Mesh(
    new THREE.CylinderGeometry(rP, rP, lPiston, SEG), M.piston);
  piston.position.y = yBot + lPiston / 2;
  g.add(piston);

  // Gland nut where the piston leaves the outer cylinder.
  const gland = new THREE.Mesh(
    new THREE.CylinderGeometry(rStrut * 1.12, rStrut * 1.12, rStrut * 0.45, SEG),
    M.hardware);
  gland.position.y = yTop - lOuter;
  g.add(gland);

  // Attachment boss at the top.
  const tr = new THREE.Mesh(
    new THREE.CylinderGeometry(rStrut * P.trunnion, rStrut * P.trunnion,
                               rStrut * 1.6, SEG),
    M.structure);
  tr.rotation.z = Math.PI / 2;                      // pin axis across
  tr.position.y = yTop;
  g.add(tr);

  return g;
}

/**
 * A torque link: the scissor pair that stops the piston rotating in the
 * cylinder. Two short arms meeting at an elbow, standing off to -Z.
 */
function torqueLink(yTop, yBot, rStrut) {
  const g = new THREE.Group();
  const rArm = rStrut * 0.20;
  const z = -(rStrut * 1.25);
  const yMid = (yTop + yBot) / 2;
  const elbow = new THREE.Vector3(0, yMid, z - rStrut * 0.55);
  const upper = new THREE.Vector3(0, yTop, z * 0.55);
  const lower = new THREE.Vector3(0, yBot, z * 0.55);

  for (const [a, b] of [[upper, elbow], [elbow, lower]]) {
    const m = rod(a, b, rArm, M.structure);
    if (m) g.add(m);
  }
  const pin = new THREE.Mesh(
    new THREE.CylinderGeometry(rArm * 1.5, rArm * 1.5, rStrut * 0.5, 12),
    M.hardware);
  pin.rotation.z = Math.PI / 2;
  pin.position.copy(elbow);
  g.add(pin);
  return g;
}

/** An axle along X, spanning +/- `halfSpan`, centred at (0, y, z). */
function axle(halfSpan, rAxle, y, z) {
  const m = new THREE.Mesh(
    new THREE.CylinderGeometry(rAxle, rAxle, halfSpan * 2, Math.max(8, SEG / 2)),
    M.hardware);
  m.rotation.z = Math.PI / 2;
  m.position.set(0, y, z);
  return m;
}

/**
 * The inverted-U fork of the single-wheel gear, as one swept tube.
 *
 * Drawn as a path with real corner arcs rather than a mitred join: a fork is
 * a bent tube in life and a sharp corner reads as two separate parts.
 */
function forkU(xf, yAxle, yTop, rFork) {
  const rc = Math.min(xf * 0.85, (yTop - yAxle) * 0.55);
  const pts = [];
  const seg = 10;

  // Left leg, densified so the spline does not bow it.
  for (let i = 0; i <= 3; i++) {
    pts.push(new THREE.Vector3(-xf, yAxle + (i / 3) * (yTop - rc - yAxle), 0));
  }
  for (let i = 1; i <= seg; i++) {                  // corner, left
    const a = Math.PI - (i / seg) * (Math.PI / 2);
    pts.push(new THREE.Vector3(-xf + rc + rc * Math.cos(a),
                               yTop - rc + rc * Math.sin(a), 0));
  }
  for (let i = 1; i < seg; i++) {                   // across the top
    pts.push(new THREE.Vector3(-xf + rc + (i / seg) * (2 * xf - 2 * rc),
                               yTop, 0));
  }
  for (let i = 0; i <= seg; i++) {                  // corner, right
    const a = (Math.PI / 2) - (i / seg) * (Math.PI / 2);
    pts.push(new THREE.Vector3(xf - rc + rc * Math.cos(a),
                               yTop - rc + rc * Math.sin(a), 0));
  }
  for (let i = 1; i <= 3; i++) {                    // right leg
    pts.push(new THREE.Vector3(xf, yTop - rc - (i / 3) * (yTop - rc - yAxle), 0));
  }

  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  return new THREE.Mesh(
    new THREE.TubeGeometry(curve, pts.length * 3, rFork, 16, false),
    M.structure);
}

/** The fore-and-aft beam of a bogie, with its centre pivot boss. */
function bogieBeam(length, rStrut, y) {
  const g = new THREE.Group();
  const w = rStrut * P.beamWidth;
  const h = rStrut * P.beamHeight;

  const beam = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, length), M.structure);
  beam.position.set(0, y, 0);
  g.add(beam);

  // Taper the ends by capping them with a smaller box -- cheap, but it stops
  // the beam looking like a length of 2x4.
  for (const s of [-1, 1]) {
    const cap = new THREE.Mesh(
      new THREE.BoxGeometry(w * 0.98, h * 0.55, h * 0.9), M.structure);
    cap.position.set(0, y, s * (length / 2 + h * 0.44));
    g.add(cap);
  }

  const pivot = new THREE.Mesh(
    new THREE.CylinderGeometry(rStrut * 1.05, rStrut * 1.05, w * 1.5, SEG),
    M.hardware);
  pivot.rotation.z = Math.PI / 2;
  pivot.position.set(0, y, 0);
  g.add(pivot);

  return g;
}

/* ------------------------------------------------------------------ *
 * Assembly
 * ------------------------------------------------------------------ */

/**
 * Build a landing gear.
 *
 * @param {object}  opts
 * @param {number}  opts.wheels  1, 2, 4 or 6
 * @param {number}  opts.rWheel  rim radius
 * @param {number}  opts.rTire   tire outer radius (must exceed rWheel)
 * @param {number}  opts.lStrut  attachment point to axle
 * @param {number}  opts.rStrut  strut radius
 * @param {boolean} opts.torqueLinks  draw the scissor links (default true)
 * @returns {THREE.Group}
 */
export function landingGear({
  wheels = 2,
  rWheel = 0.28,
  rTire = 0.45,
  lStrut = 1.30,
  rStrut = 0.09,
  torqueLinks = true,
} = {}) {
  if (![1, 2, 4, 6].includes(wheels)) {
    throw new Error(`landingGear: wheels must be 1, 2, 4 or 6; got ${wheels}`);
  }
  if (rTire <= rWheel) {
    throw new Error(`landingGear: rTire (${rTire}) must exceed rWheel (${rWheel})`);
  }

  const g = new THREE.Group();
  g.name = `landingGear${wheels}`;

  const wTire = rTire * P.tireWidth;
  const rAxle = rStrut * P.axleRadius;
  const rFork = rStrut * P.forkRadius;
  const gap = rTire * P.clearance;
  const yAxle = -lStrut;

  const addWheel = (x, z) => {
    const w = wheel(rWheel, rTire, wTire);
    w.position.set(x, yAxle, z);
    g.add(w);
  };

  if (wheels === 1) {
    // The fork straddles the wheel, so the strut can only come down as far as
    // the top of the U. Clamp so a short lStrut degrades to a stub rather
    // than to an inside-out strut.
    const xf = wTire / 2 + gap + rFork;
    const yTop = yAxle + rTire + gap + rFork;
    const yStrutBot = Math.min(yTop, -rStrut * 2);

    addWheel(0, 0);
    g.add(forkU(xf, yAxle, yTop, rFork));
    g.add(axle(xf + rFork * 0.5, rAxle, yAxle, 0));
    g.add(strut(0, yStrutBot, rStrut));
    if (torqueLinks && yStrutBot < -rStrut * 3) {
      g.add(torqueLink(-rStrut * 1.2, yStrutBot + rStrut * 0.8, rStrut));
    }
  } else if (wheels === 2) {
    // Strut down between the wheels, onto the axle.
    const xw = rStrut + wTire / 2 + gap;

    addWheel(-xw, 0);
    addWheel(xw, 0);
    g.add(axle(xw + wTire * 0.6, rAxle, yAxle, 0));
    g.add(strut(0, yAxle, rStrut));
    if (torqueLinks) {
      g.add(torqueLink(-rStrut * 1.2, yAxle + rStrut * 1.2, rStrut));
    }

    // Collar where the strut meets the axle, so the two do not merely
    // intersect.
    const collar = new THREE.Mesh(
      new THREE.CylinderGeometry(rStrut * 1.15, rStrut * 1.15, rStrut * 1.9, SEG),
      M.hardware);
    collar.position.y = yAxle;
    g.add(collar);
  } else {
    // 4 and 6: a bogie beam carrying nAxles axles, strut on the centre pivot.
    const nAxles = wheels / 2;
    const dz = rTire * P.axleSpacing;
    const xw = rStrut * P.beamWidth * 0.5 + wTire / 2 + gap;
    const zs = [];
    for (let i = 0; i < nAxles; i++) zs.push((i - (nAxles - 1) / 2) * dz);

    for (const z of zs) {
      addWheel(-xw, z);
      addWheel(xw, z);
      g.add(axle(xw + wTire * 0.6, rAxle, yAxle, z));
    }
    g.add(bogieBeam((nAxles - 1) * dz, rStrut, yAxle));
    g.add(strut(0, yAxle, rStrut));
    if (torqueLinks) {
      g.add(torqueLink(-rStrut * 1.2, yAxle + rStrut * 1.4, rStrut));
    }
  }

  // Where the tires touch the ground, relative to the origin.
  g.userData.contactY = yAxle - rTire;
  g.userData.params = { wheels, rWheel, rTire, lStrut, rStrut };
  return g;
}

/** Free the geometry of a gear built here. Materials are shared; leave them. */
export function disposeGear(group) {
  group.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });
}

export default landingGear;
