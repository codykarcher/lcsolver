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

/** Default turbofan slenderness: overall length over fan diameter. */
const LENGTH_OVER_DIAMETER = 2.5;

/** Tag a finished engine with its extent so callers need not measure it. */
function finish(g, length, rMax, name) {
  g.name = name;
  g.userData.length = length;
  g.userData.rMax = rMax;
  return g;
}

/**
 * An ogive spinner/nose cone, tip at +Z, closed with a disc at the base.
 */
/** The ogive radius law used by the spinner. */
const ogiveR = (t, rBase) => rBase * Math.cos(t * Math.PI / 2) ** 0.72;

/**
 * The spinner's swirl, as paint rather than as an object.
 *
 * On a real engine the mark is there so ground crew can see at a glance that
 * the fan is turning, and it is exactly that -- paint. Modelling it as a tube
 * lying on the cone gave it a thickness it should not have and a silhouette
 * that broke the nose profile.
 *
 * It is drawn in the lathe's own UV space, which makes the spiral trivial: a
 * lathe maps u to angle and v along the profile, so a spiral -- angle
 * proportional to distance along the cone -- is a straight diagonal line.
 * Mapped back onto the cone, a constant width in u narrows toward the tip
 * because the circumference does, which is what a painted swirl actually
 * does.
 *
 * Built once and shared. Returns the plain painted material where there is no
 * canvas to draw on, so the module still loads outside a browser.
 */
let _spinnerMat = null;
function spinnerMaterial() {
  if (_spinnerMat) return _spinnerMat;
  if (typeof document === 'undefined') { _spinnerMat = M.painted; return _spinnerMat; }

  const S = 512;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#23262b';                  // matches M.painted
  ctx.fillRect(0, 0, S, S);

  // Front of the cone only, stopping just shy of the tip. Extended 20% down
  // the cone from where it started; TURNS scales with the span so the spiral
  // simply carries on at the same pitch rather than winding tighter.
  const V1 = 0.975, SPAN = 0.415 * 1.20, V0 = V1 - SPAN;
  const TURNS = 0.85 * 1.20;
  ctx.strokeStyle = '#f0f2f4';
  ctx.lineWidth = 0.055 * S;
  ctx.lineCap = 'round';
  // Three passes offset by a full turn so the stroke wraps cleanly in u.
  for (const off of [-1, 0, 1]) {
    ctx.beginPath();
    for (let i = 0; i <= 64; i++) {
      const t = i / 64;
      const v = V0 + (V1 - V0) * t;
      const u = 0.5 - TURNS * t + off;        // decreasing: winds the other way
      // CanvasTexture flips Y, so canvas row 0 is v = 1, the tip.
      const x = u * S, y = (1 - v) * S;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.anisotropy = 8;
  _spinnerMat = new THREE.MeshStandardMaterial({
    map: tex, roughness: 0.48, metalness: 0.15,
  });
  return _spinnerMat;
}

/**
 * The core's finish, as paint on one continuous surface.
 *
 * Splitting the core into a silver lathe and a burnt lathe meeting at a shared
 * sample kept the SHAPE continuous but not the shading: each mesh averages its
 * own vertex normals, so the two end rows disagree and the junction reads as a
 * crease. Nothing short of one mesh removes that.
 *
 * So the core is a single lathe and the transition is textured. Colour comes
 * from `map`; roughness and metalness are carried in the green and blue of a
 * second texture, so the burnt end is genuinely duller and less reflective
 * rather than merely darker. A lathe's v runs along the profile, front to
 * back, which is exactly the axis the finish changes along.
 */
let _coreMat = null;
function coreMaterial() {
  if (_coreMat) return _coreMat;
  if (typeof document === 'undefined') { _coreMat = M.casing; return _coreMat; }

  const H = 512;
  const strip = (stops) => {
    const cv = document.createElement('canvas');
    cv.width = 4; cv.height = H;
    const ctx = cv.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    // CanvasTexture flips Y, so row 0 is v = 1 -- the aft end.
    for (const [at, colour] of stops) grad.addColorStop(at, colour);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 4, H);
    const t = new THREE.CanvasTexture(cv);
    t.wrapS = THREE.RepeatWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    return t;
  };

  // v 1.00 .. 0.80 burnt, 0.80 .. 0.60 blending, 0.60 .. 0 bare metal.
  const colour = strip([[0, '#4a4038'], [0.20, '#4a4038'],
                        [0.40, '#c3c8ce'], [1, '#c3c8ce']]);
  colour.colorSpace = THREE.SRGBColorSpace;
  // green = roughness, blue = metalness; 0.74/0.75 burnt, 0.34/0.92 bare.
  const finish = strip([[0, 'rgb(0,189,191)'], [0.20, 'rgb(0,189,191)'],
                        [0.40, 'rgb(0,87,235)'], [1, 'rgb(0,87,235)']]);

  _coreMat = new THREE.MeshStandardMaterial({
    map: colour, roughnessMap: finish, metalnessMap: finish,
    roughness: 1, metalness: 1,
  });
  return _coreMat;
}

function spinner(rBase, len, material, zBase = 0) {
  const prof = [[zBase, 0]];
  const n = 20;                            // finer: the swirl is mapped on it
  for (let i = 0; i <= n; i++) {
    const t = i / n;                       // 0 at base, 1 at tip
    // Ogive rather than a straight cone: a straight cone reads as a party hat.
    prof.push([zBase + t * len, ogiveR(t, rBase)]);
  }
  return latheZ(prof, material, SEG);
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
  rFan = 0.90, bypassRatio = 9, length = null, blades = 40, vanes = 40,
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

  // `length` is NOSE TO TAIL -- spinner tip to plug tip -- not merely the
  // extent aft of the origin, which would leave the spinner uncounted. Left
  // null it is LENGTH_OVER_DIAMETER against the fan diameter.
  //
  // The fan case is NOT stretched by it. Case length follows from fan chord
  // and vane row, which are the fan's business; a longer engine is a longer
  // core.
  const lSpin = 2.05 * rHub;
  const zNose = 0.02 * R + lSpin;         // spinner tip, ahead of the origin
  const z0 = -0.30 * R;                   // core front, just behind the fan
  const NOSE = -z0, TAIL = 1.40;          // aft extent = NOSE + TAIL * Lc
  const L = length ?? LENGTH_OVER_DIAMETER * 2 * R;
  // A core shorter than about a diameter stops looking like a core, so a very
  // short request is clamped and the reported lengths then describe what was
  // actually built rather than what was asked for.
  const Lc = Math.max(1.2 * rCore, (L - zNose - NOSE) / TAIL);
  const zAft = z0 - Lc;

  const rotor = new THREE.Group();
  rotor.userData.rotating = true;
  rotor.userData.spin = -1;
  rotor.add(spinner(rHub, lSpin, spinnerMaterial(), 0.02 * R));

  // Blade angle is measured here from the ENGINE AXIS, so a large angle lays
  // the chord into the plane of the fan face and a small one stands it on
  // edge. A fan is coarse at the root and fine at the tip -- the tip is
  // travelling far faster, so it meets the flow at a shallower angle to the
  // disc -- which means the angle from the axis GROWS outboard. Having it the
  // other way round is what made the tips look like knives.
  const fan = bladeRow({
    count: blades, material: M.blade, hubMaterial: M.casing,
    hubLength: 1.75 * rHub,
    rHub, rTip: 1.00 * R,
    chordRoot: 0.308 * R, chordTip: 0.280 * R,
    twistRoot: 0.50, twistTip: 1.05, thickness: 0.10,
  });
  fan.position.z = -0.14 * R;
  rotor.add(fan);
  g.add(rotor);

  // Fan case: bare unfaired metal, so a thin band rather than a thick ring.
  // Runs 0.18 rFan forward of the fan plane to zCaseAft; the original 0.80
  // rFan span, carried 10% further aft.
  const zCaseAft = 0.18 * R - 0.88 * R;   // = -0.70 rFan
  const rCaseIn = 1.012 * R;              // 0.012 rFan of blade tip clearance
  const rCaseOut = rCaseIn + 0.025 * R;
  g.add(tubeZ(rCaseIn, rCaseOut, 0.18 * R, zCaseAft, M.casing, SEG));

  // Outlet guide vanes span the bypass annulus. Without them the fan case and
  // the core read as two unrelated parts floating together.
  //
  // Sat well forward, the vanes left a bare stretch of case trailing behind
  // them. Placed off the case's own aft end instead, so the trailing edge
  // stays just inside the fairing whatever the case length is set to.
  if (vanes > 0) {
    const cOgv = 0.26 * R;                            // vane chord, root
    const ogv = bladeRow({
      count: vanes, material: M.casing,
      rHub: rCore * 1.02, rTip: rCaseIn,
      chordRoot: cOgv, chordTip: 0.24 * R,
      twistRoot: 0.22, twistTip: 0.10, thickness: 0.09,
    });
    ogv.position.z = zCaseAft + 0.04 * R + 0.5 * cOgv;   // ~= -0.61 rFan
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

  // One mesh, finish painted on. See coreMaterial().
  g.add(latheZ([[z0, 0], ...spline.map((p) => [p.x, p.y]), [zAft, 0]],
               coreMaterial(), SEG));

  // Looking up the exhaust should be looking into a hole. Without this the
  // core's own aft cap is the first thing you meet, lit and metallic.
  const mouth = new THREE.Mesh(
    new THREE.RingGeometry(0.58 * rCore, 0.795 * rCore, SEG), M.cavity);
  mouth.rotation.y = Math.PI;                  // face aft
  mouth.position.z = zAft - 0.004 * Lc;
  g.add(mouth);

  // Exhaust plug, sized to leave only a narrow annulus against the outlet
  // wall -- that gap is the exhaust, and on a real engine it is a slot, not
  // an open ring.
  g.add(latheZ([
    [zAft - 0.015 * Lc, 0], [zAft - 0.015 * Lc, 0.66 * rCore],
    [zAft - 0.40 * Lc, 0],
  ], M.hot, SEG));

  g.userData.rCore = rCore;
  g.userData.bypassRatio = bypassRatio;
  g.userData.coreLength = Lc;
  g.userData.noseZ = zNose;
  g.userData.overallLength = zNose + NOSE + TAIL * Lc;
  g.userData.lengthOverDiameter = g.userData.overallLength / (2 * R);
  return finish(g, NOSE + TAIL * Lc, rCaseOut, 'turbofan');
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
