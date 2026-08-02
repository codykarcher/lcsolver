/**
 * Engine cores, without ducting.
 *
 * An installed engine is two things: the machine, and whatever fairing wraps
 * it. This file is only the machine. Nacelles, bypass ducts, cowlings and
 * inlet lips belong in a separate component, because the same core turns up
 * inside very different fairings and because the fairing is usually the part
 * the optimizer has an opinion about.
 *
 * Six prototypes:
 *
 *     turbofan       fan in a case, core body, converging nozzle
 *     turbojet       shell with a compressor face and a nozzle
 *     turboprop      propeller, gearbox snout, core, exhaust stack
 *     electricMotor  finned outrunner can and shaft
 *     pistonEngine   horizontally opposed, air cooled
 *     rotaryEngine   Wankel, one or two rotors
 *
 * Detail is deliberately low. No internal machinery is modelled: a turbofan is
 * a fan, a body and a nozzle, and the compressor a turbojet appears to have is
 * one row of blades at the inlet with a blanking disc behind it. What matters
 * at this stage is that each type is unmistakable in silhouette.
 *
 * Frame
 * -----
 * Axis along **Z**, with **+Z forward** -- air enters at +Z and leaves at -Z,
 * matching the landing gear's fore-and-aft Z. The origin is the centre of the
 * **front face**: the fan plane, the propeller plane, the compressor face, the
 * shaft end. So an engine placed at a station on an aeroplane hangs aft of
 * that station, which is how engine positions are usually quoted.
 *
 * Every builder sets `userData.length` (total extent aft of the origin, a
 * positive number) and `userData.rMax` (largest radius), so callers can lay
 * engines out or size a fairing without measuring the mesh.
 *
 * Each takes a single leading size parameter with the rest derived. Those
 * derivations are visual, not physical -- same rule as the landing gear.
 */
import * as THREE from 'three';
import * as M from './materials.js';
import { latheZ, tubeZ, bladeRow, epitrochoid, rod } from './geom.js';

const SEG = 48;

/** Tag a finished engine with its extent so callers need not measure it. */
function finish(g, length, rMax, name) {
  g.name = name;
  g.userData.length = length;
  g.userData.rMax = rMax;
  return g;
}

/**
 * An ogive spinner/nose cone, tip at +Z.
 *
 * Closed with a disc at the base (the leading `r = 0`) so it is a solid even
 * when nothing sits behind it. An open base is a hole you can see through the
 * moment the hub is smaller than the spinner or the view gets low.
 */
/** The ogive radius law, shared by the spinner and the swirl painted on it. */
const ogiveR = (t, rBase) => rBase * Math.cos(t * Math.PI / 2) ** 0.72;

function spinner(rBase, len, material, zBase = 0) {
  const prof = [[zBase, 0]];
  const n = 12;
  for (let i = 0; i <= n; i++) {
    const t = i / n;                       // 0 at base, 1 at tip
    // Ogive rather than a straight cone: a straight cone reads as a party hat.
    prof.push([zBase + t * len, ogiveR(t, rBase)]);
  }
  return latheZ(prof, material, SEG);
}

/**
 * The painted spiral on a spinner nose.
 *
 * On a real engine it is there so ground crew can see at a glance that the
 * fan is turning. It follows the cone's own radius law so it lies on the
 * surface, standing proud by rather less than its own thickness, and winds in
 * to nothing at the tip because that is where the radius goes.
 */
function spinnerSwirl(rBase, len, zBase, turns = 1.15) {
  const rTube = 0.055 * rBase;
  const pts = [];
  const n = 96;
  for (let i = 0; i <= n; i++) {
    const t = 0.04 + (0.965 - 0.04) * (i / n);
    const r = ogiveR(t, rBase) + rTube * 0.45;
    const th = turns * Math.PI * 2 * t;
    pts.push(new THREE.Vector3(r * Math.cos(th), r * Math.sin(th),
                               zBase + t * len));
  }
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  const g = new THREE.Group();
  g.add(new THREE.Mesh(
    new THREE.TubeGeometry(curve, n * 2, rTube, 8, false), M.marking));
  // TubeGeometry leaves its ends open. They are small, but a hole is a hole:
  // cap each with a bead, which also rounds the ends of the stroke.
  for (const p of [pts[0], pts[pts.length - 1]]) {
    // 1.06 rather than 1.0: a sphere of exactly rTube meets the tube's rim
    // coincidentally and z-fights along the seam.
    const cap = new THREE.Mesh(
      new THREE.SphereGeometry(rTube * 1.06, 10, 8), M.marking);
    cap.position.copy(p);
    g.add(cap);
  }
  return g;
}

/* ==================================================================== *
 * Turbofan
 * ==================================================================== */

/**
 * High-bypass turbofan core: fan, fan case, outlet guide vanes, core body,
 * converging nozzle and exhaust plug.
 *
 * The guide vanes are not decoration. With the nacelle omitted there is an
 * open annulus between the fan case and the core, and without something
 * spanning it the two read as unrelated parts floating together.
 */
export function turbofan({
  rFan = 0.90, bypassRatio = 9, blades = 40, vanes = 40,
} = {}) {
  if (!(bypassRatio > 0)) throw new Error('turbofan: bypassRatio must be > 0');

  const g = new THREE.Group();
  const R = rFan;

  // Bypass ratio fixes the core diameter. Taking it as the ratio of bypassed
  // frontal area to core frontal area,
  //
  //     BPR = (pi rFan^2 - pi rCore^2) / (pi rCore^2)
  //
  // so rCore = rFan / sqrt(BPR + 1). A modern high-bypass fan lands near 9,
  // which puts the core at about a third of fan radius; an early low-bypass
  // engine near 1 puts it at 0.7, which is why those engines look like tubes.
  const rCore = R / Math.sqrt(bypassRatio + 1);
  const rHub = 0.62 * rCore;              // fan hub sits inside the core line

  // Core length scales with the core, not the fan: tied to fan radius instead,
  // a high-bypass engine would grow into a needle.
  const z0 = -0.30 * R;                   // core front, just behind the fan
  const Lc = 5.2 * rCore;
  const zAft = z0 - Lc;

  const rotor = new THREE.Group();
  rotor.userData.rotating = true;
  rotor.userData.spin = -1;
  const lSpin = 2.05 * rHub;
  rotor.add(spinner(rHub, lSpin, M.painted, 0.02 * R));
  rotor.add(spinnerSwirl(rHub, lSpin, 0.02 * R));

  // Blade angle is measured here from the ENGINE AXIS, so a large angle lays
  // the chord into the plane of the fan face and a small one stands it on
  // edge. A fan is coarse at the root and fine at the tip -- the tip is
  // travelling far faster, so it meets the flow at a shallower angle to the
  // disc -- which means the angle from the axis GROWS outboard. Having it the
  // other way round is what made the tips look like knives.
  const fan = bladeRow({
    count: blades, material: M.blade, hubMaterial: M.casing,
    hubLength: 1.75 * rHub,
    rHub, rTip: 0.98 * R,
    chordRoot: 0.308 * R, chordTip: 0.280 * R,
    twistRoot: 0.50, twistTip: 1.05, thickness: 0.10,
  });
  fan.position.z = -0.14 * R;
  rotor.add(fan);
  g.add(rotor);

  // Fan case: bare unfaired metal, so a thin band rather than a thick ring,
  // carried well down the engine.
  const zCaseAft = -0.94 * R;
  const rCaseOut = 1.045 * R;
  g.add(tubeZ(1.02 * R, rCaseOut, 0.18 * R, zCaseAft, M.casing, SEG));

  // Outlet guide vanes span the bypass annulus. Without them the fan case and
  // the core read as two unrelated parts floating together.
  if (vanes > 0) {
    const ogv = bladeRow({
      count: vanes, material: M.casing,
      rHub: rCore * 1.02, rTip: 1.02 * R,
      chordRoot: 0.26 * R, chordTip: 0.24 * R,
      twistRoot: 0.22, twistTip: 0.10, thickness: 0.09,
    });
    ogv.position.z = -0.46 * R;
    g.add(ogv);
  }

  // Core body. A spline through five stations rather than straight segments:
  // it swells through the turbine and then falls away to the outlet with the
  // curvature staying continuous, where a piecewise profile creases at every
  // joint and catches the light as a ring.
  const ctrl = [
    new THREE.Vector2(z0, rCore),
    new THREE.Vector2(z0 - 0.28 * Lc, 1.04 * rCore),
    new THREE.Vector2(z0 - 0.56 * Lc, 1.13 * rCore),   // turbine
    new THREE.Vector2(z0 - 0.80 * Lc, 1.03 * rCore),
    new THREE.Vector2(zAft, 0.78 * rCore),             // outlet
  ];
  const spline = new THREE.SplineCurve(ctrl).getPoints(48);
  g.add(latheZ([[z0, 0], ...spline.map((p) => [p.x, p.y]), [zAft, 0]],
               M.casing, SEG));

  // Nozzle: a thin hot lip hugging the outlet, not a separate cowl.
  g.add(tubeZ(0.785 * rCore, 0.845 * rCore,
              zAft + 0.10 * Lc, zAft - 0.015 * Lc, M.hot, SEG));

  // Exhaust plug, sized to leave only a narrow annulus against the outlet
  // wall -- that gap is the exhaust, and on a real engine it is a slot, not
  // an open ring.
  g.add(latheZ([
    [zAft - 0.015 * Lc, 0], [zAft - 0.015 * Lc, 0.66 * rCore],
    [zAft - 0.40 * Lc, 0],
  ], M.hot, SEG));

  g.userData.rCore = rCore;
  g.userData.bypassRatio = bypassRatio;
  return finish(g, -(zAft - 0.40 * Lc), rCaseOut, 'turbofan');
}

/* ==================================================================== *
 * Turbojet
 * ==================================================================== */

/**
 * Turbojet: a case with a compressor face at the inlet and a nozzle aft.
 *
 * The case is a shell so the inlet is genuinely open and you see the blades
 * down it; a blanking disc a little way behind stops you seeing out the back.
 */
export function turbojet({ rCase = 0.42, blades = 26 } = {}) {
  const g = new THREE.Group();
  const R = rCase;

  const rotor = new THREE.Group();
  rotor.userData.rotating = true;
  rotor.add(spinner(0.25 * R, 0.55 * R, M.casing, 0.05 * R));
  rotor.add(bladeRow({
    count: blades, material: M.blade, hubMaterial: M.casing,
    hubLength: 0.3 * R,
    rHub: 0.27 * R, rTip: 0.95 * R,
    chordRoot: 0.24 * R, chordTip: 0.20 * R,
    twistRoot: 1.15, twistTip: 0.45, thickness: 0.09,
  }));
  g.add(rotor);

  g.add(tubeZ(0.94 * R, R, 0.10 * R, -2.70 * R, M.casing, SEG));

  const blank = new THREE.Mesh(
    new THREE.CircleGeometry(0.95 * R, SEG), M.hot);
  blank.position.z = -0.75 * R;
  blank.rotation.y = Math.PI;                 // face the inlet
  g.add(blank);

  // A pair of banding rings, which is most of what stops a plain cylinder
  // looking like a plain cylinder.
  for (const z of [-0.95 * R, -1.85 * R]) {
    g.add(tubeZ(R, 1.06 * R, z + 0.05 * R, z - 0.05 * R, M.accessory, SEG));
  }

  g.add(latheZ([
    [-2.65 * R, R], [-3.05 * R, 0.78 * R], [-3.35 * R, 0.62 * R],
    [-3.35 * R, 0.55 * R], [-3.05 * R, 0.71 * R], [-2.65 * R, 0.93 * R],
    [-2.65 * R, R],
  ], M.hot, SEG));

  g.add(latheZ([
    [-2.60 * R, 0], [-2.60 * R, 0.34 * R], [-3.10 * R, 0.26 * R],
    [-3.50 * R, 0],
  ], M.hot, SEG));

  return finish(g, 3.50 * R, 1.06 * R, 'turbojet');
}

/* ==================================================================== *
 * Turboprop
 * ==================================================================== */

/**
 * Turboprop: propeller, reduction gearbox snout, core body, side exhaust.
 *
 * The gearbox is what distinguishes this from a turbojet with a propeller
 * bolted on -- the snout is much smaller in diameter than the core behind it,
 * and the offset between them is the recognisable feature.
 */
export function turboprop({ rProp = 1.25, blades = 4 } = {}) {
  const g = new THREE.Group();
  const R = rProp;
  const rCore = 0.24 * R;

  const rotor = new THREE.Group();
  rotor.userData.rotating = true;
  rotor.add(spinner(0.115 * R, 0.30 * R, M.painted, 0.03 * R));
  rotor.add(bladeRow({
    count: blades, material: M.painted, hubMaterial: M.accessory,
    hubLength: 0.13 * R,
    rHub: 0.10 * R, rTip: R,
    chordRoot: 0.15 * R, chordTip: 0.085 * R,
    twistRoot: 1.20, twistTip: 0.22, thickness: 0.11,
  }));
  g.add(rotor);

  // Gearbox snout, then a step out to the core.
  g.add(latheZ([
    [-0.02 * R, 0], [-0.02 * R, 0.135 * R],
    [-0.30 * R, 0.145 * R], [-0.36 * R, 0.20 * R],
    [-0.42 * R, rCore], [-1.25 * R, rCore], [-1.45 * R, 0.20 * R],
    [-1.45 * R, 0],
  ], M.casing, SEG));

  // Annular intake below the snout, which is where a turboprop breathes.
  const scoop = new THREE.Mesh(
    new THREE.BoxGeometry(0.26 * R, 0.11 * R, 0.22 * R), M.accessory);
  scoop.position.set(0, -0.20 * R, -0.44 * R);
  g.add(scoop);

  // Exhaust stack, swept aft and outboard.
  const st = rod(new THREE.Vector3(0.16 * R, 0.10 * R, -1.05 * R),
                 new THREE.Vector3(0.30 * R, 0.16 * R, -1.35 * R),
                 0.075 * R, M.hot, 20);
  if (st) g.add(st);

  g.add(latheZ([
    [-1.45 * R, 0], [-1.45 * R, 0.20 * R], [-1.58 * R, 0.16 * R], [-1.58 * R, 0],
  ], M.hot, SEG));

  return finish(g, 1.58 * R, R, 'turboprop');
}

/* ==================================================================== *
 * Electric motor
 * ==================================================================== */

/**
 * Outrunner electric motor: finned can, end bells, output shaft, mount flange.
 *
 * Drawn as an outrunner (the case itself is the rotor) because that is what
 * an electric aircraft propulsor almost always is, and because the fins
 * running the length of the can are the feature that says "motor" instantly.
 */
export function electricMotor({ rMotor = 0.20, lMotor = 0.34, fins = 24 } = {}) {
  const g = new THREE.Group();
  const R = rMotor, L = lMotor;

  g.add(latheZ([
    [0, 0], [0, 0.96 * R], [-0.06 * L, R], [-0.94 * L, R],
    [-L, 0.96 * R], [-L, 0],
  ], M.casing, SEG));

  // Axial cooling fins.
  const finGeo = new THREE.BoxGeometry(0.055 * R, 0.16 * R, 0.80 * L);
  for (let i = 0; i < fins; i++) {
    const a = (i / fins) * Math.PI * 2;
    const f = new THREE.Mesh(finGeo, M.casing);
    f.position.set(Math.cos(a) * R * 1.05, Math.sin(a) * R * 1.05, -0.5 * L);
    f.rotation.z = a;
    g.add(f);
  }

  // Windings glimpsed through the gap between can and rear bell.
  g.add(tubeZ(0.72 * R, 0.9 * R, -0.90 * L, -0.98 * L, M.winding, SEG));

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16 * R, 0.16 * R, 0.55 * R, 20), M.hardware);
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = 0.22 * R;
  g.add(shaft);

  // Rear mount flange with bolt bosses.
  g.add(tubeZ(0.30 * R, 1.16 * R, -L, -L - 0.07 * R, M.accessory, SEG));
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
    const b = new THREE.Mesh(
      new THREE.CylinderGeometry(0.10 * R, 0.10 * R, 0.10 * R, 12), M.hardware);
    b.rotation.x = Math.PI / 2;
    b.position.set(Math.cos(a) * 0.95 * R, Math.sin(a) * 0.95 * R,
                   -L - 0.08 * R);
    g.add(b);
  }

  return finish(g, L + 0.15 * R, 1.16 * R, 'electricMotor');
}

/* ==================================================================== *
 * Piston engine -- horizontally opposed
 * ==================================================================== */

/**
 * Air-cooled horizontally opposed piston engine, four or six cylinders.
 *
 * The cylinders stick straight out to port and starboard with their cooling
 * fins showing, which is the whole silhouette of a light-aircraft engine. Fins
 * are drawn as a stack of discs -- crude, but at this size a fin is two
 * triangles' worth of information and a stack of them reads correctly.
 */
export function pistonEngine({ cylinders = 4, size = 0.34, finsPer = 7 } = {}) {
  if (cylinders % 2) throw new Error('pistonEngine: cylinders must be even');
  const g = new THREE.Group();
  const S = size;
  const perSide = cylinders / 2;

  const caseLen = S * (1.15 + 0.95 * perSide);
  const crank = new THREE.Mesh(
    new THREE.BoxGeometry(1.05 * S, 0.95 * S, caseLen), M.accessory);
  crank.position.z = -caseLen / 2 - 0.30 * S;
  g.add(crank);

  // Propeller shaft and flange.
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.11 * S, 0.11 * S, 0.42 * S, 20), M.hardware);
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = 0.16 * S;
  g.add(shaft);
  g.add(tubeZ(0.11 * S, 0.30 * S, -0.02 * S, -0.10 * S, M.hardware, 24));

  const rBarrel = 0.30 * S;
  const xIn = 0.52 * S, xOut = 1.18 * S;
  const finGeo = new THREE.CylinderGeometry(0.40 * S, 0.40 * S, 0.035 * S, 20);
  const barGeo = new THREE.CylinderGeometry(rBarrel, rBarrel, xOut - xIn, 20);
  const headGeo = new THREE.BoxGeometry(0.34 * S, 0.72 * S, 0.66 * S);

  for (let i = 0; i < perSide; i++) {
    // Banks are staggered fore-and-aft, as they are on a real opposed engine.
    const z = -0.72 * S - i * 0.95 * S;
    for (const s of [-1, 1]) {
      const zs = z - (s < 0 ? 0.14 * S : 0);

      const bar = new THREE.Mesh(barGeo, M.finned);
      bar.rotation.z = Math.PI / 2;
      bar.position.set(s * (xIn + xOut) / 2, 0, zs);
      g.add(bar);

      for (let f = 0; f < finsPer; f++) {
        const t = (f + 0.5) / finsPer;
        const fin = new THREE.Mesh(finGeo, M.finned);
        fin.rotation.z = Math.PI / 2;
        fin.position.set(s * (xIn + t * (xOut - xIn)), 0, zs);
        g.add(fin);
      }

      const head = new THREE.Mesh(headGeo, M.accessory);
      head.position.set(s * (xOut + 0.15 * S), 0, zs);
      g.add(head);
    }
  }

  // Accessory case on the back.
  const acc = new THREE.Mesh(
    new THREE.BoxGeometry(0.80 * S, 0.70 * S, 0.34 * S), M.accessory);
  acc.position.z = -caseLen - 0.45 * S;
  g.add(acc);

  return finish(g, caseLen + 0.62 * S, xOut + 0.32 * S, 'pistonEngine');
}

/* ==================================================================== *
 * Rotary (Wankel)
 * ==================================================================== */

/**
 * Wankel rotary, one or two rotors.
 *
 * The housing outline is the real two-lobed epitrochoid rather than a rounded
 * rectangle, because that silhouette is the only thing that identifies the
 * type at a glance -- get it wrong and this is just a lump with a shaft.
 */
export function rotaryEngine({ rotors = 2, size = 0.20, e = null } = {}) {
  const g = new THREE.Group();
  const R = size;
  const ecc = e ?? 0.145 * R;

  const housing = epitrochoid(R, ecc, 128);
  const plate = epitrochoid(R * 1.13, ecc, 96);
  const wRotor = 0.72 * R;
  const wPlate = 0.16 * R;

  const extrude = (shape, depth, mat, z0) => {
    const m = new THREE.Mesh(
      new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false }), mat);
    m.position.z = z0;
    return m;
  };

  let z = -wPlate;
  g.add(extrude(plate, wPlate, M.accessory, z));       // front plate

  for (let i = 0; i < rotors; i++) {
    g.add(extrude(housing, wRotor, M.finned, z - wRotor));
    z -= wRotor;
    g.add(extrude(plate, wPlate, M.accessory, z - wPlate));
    z -= wPlate;
  }

  // Eccentric output shaft, right through and proud at both ends.
  const zFront = 0.30 * R, zBack = z - 0.25 * R;
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.17 * R, 0.17 * R, zFront - zBack, 20),
    M.hardware);
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = (zFront + zBack) / 2;
  g.add(shaft);

  // Intake and exhaust stubs on opposite lobes.
  for (const [x, y, mat] of [[1.02 * R, 0.42 * R, M.accessory],
                             [-1.02 * R, -0.42 * R, M.hot]]) {
    const s = new THREE.Mesh(
      new THREE.CylinderGeometry(0.17 * R, 0.17 * R, 0.42 * R, 16), mat);
    s.rotation.z = Math.PI / 2;
    s.position.set(x * 1.12, y, z / 2);
    g.add(s);
  }

  return finish(g, -z, R * 1.3, 'rotaryEngine');
}

export const ENGINES = {
  turbofan, turbojet, turboprop, electricMotor, pistonEngine, rotaryEngine,
};
export default ENGINES;
