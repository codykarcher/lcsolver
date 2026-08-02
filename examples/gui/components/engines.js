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
import { latheZ, tubeZ, bladeRow, rod } from './geom.js';

const SEG = 48;

/**
 * Default turbofan slenderness: nose-to-tail length over FAN diameter (blade
 * tips, 2 x rFan) -- not over the case, which is 3.7% wider.
 */
const LENGTH_OVER_DIAMETER = 2.0;

/**
 * RADII BELOW ARE MULTIPLES OF rCore, the BPR-derived radius at the fan duct's
 * back face. So the whole core follows bypass ratio: drop BPR and the core
 * swells with it, in proportion, rather than only the duct station moving.
 *
 * They are then ceilinged at MAX_RADIUS x FAN RADIUS, which is the one place
 * a fan-relative number belongs -- the core cannot be allowed out through its
 * own fan case however low the bypass ratio goes.
 */

/** Core outlet wall. */
const R_OUTLET = 1.25;

/**
 * Ceiling on every radius in the core, as a fraction of FAN RADIUS.
 *
 * Applies to the BPR-derived duct-face radius too, so a very low bypass ratio
 * cannot drive the core out through its own fan case. When it binds on the
 * duct face the geometry stops honouring BPR exactly, and
 * `userData.bypassRatioEffective` reports what was actually built.
 */
const MAX_RADIUS = 0.95;

/**
 * Exhaust annulus, as a FRACTION of the outlet radius rather than an absolute
 * offset. That is what makes the plug scale with the outlet: widen the outlet
 * and the cone grows with it, keeping the slot in proportion, where a fixed
 * offset would have left the same slot around an ever-fatter cone.
 *
 * 0.308 is the previous fixed 0.24 rCore taken against the default 0.78
 * outlet, so the default engine is unchanged.
 */
const EXHAUST_GAP_FRAC = 0.308;

/**
 * Exhaust plug, as fractions of core length: how far the tip runs aft of the
 * core's exit plane, and how far the base is buried forward of it.
 *
 * The plug is defined by its TIP, not its base, for two reasons. The tip sets
 * overall length, and the base wants to sit inside the core: given a base
 * aft of the exit plane the cone simply hangs there, and widening the exhaust
 * annulus put daylight in the gap and detached it.
 *
 * Shortening the plug does NOT shorten the engine -- overall length is held to
 * LENGTH_OVER_DIAMETER -- it lengthens the core to compensate, which scales
 * the plug back up. So a spike 70% of its drawn length is not 70% of the
 * fraction: solving f/(1 + f) for the ratio gives 0.40 -> 0.25, and since the
 * size term cancels that holds at any fan radius and bypass ratio.
 */
const PLUG_TIP = 0.25;
const PLUG_EMBED = 0.05;

/**
 * The two movable stations that shape the core, between the fixed inlet and
 * the fixed outlet.
 *
 *     z   fraction of the EXPOSED core, 0 at the fan duct's back face and 1
 *         at the exit plane. Measured from the duct rather than from the core
 *         front because the part forward of the duct exit is inside the
 *         bypass annulus and is not shaped by these.
 *     r   radius as a multiple of rCore, ceilinged (see the note above)
 *
 * The inlet is not adjustable because bypass ratio sets it, and the outlet is
 * not adjustable because the exhaust annulus does. Everything between is
 * these two points and the spline through them.
 *
 * Defaults are the profile settled on by eye, and are no longer exposed as
 * controls -- fan radius and bypass ratio are the parameters.
 *
 * The list is not fixed at three -- any number of stations works, they are
 * simply spline control points between inlet and outlet.
 */
const CORE_SECTIONS = [{ z: 0.36, r: 1.25 }, { z: 0.67, r: 1.53 }];

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
  sections = CORE_SECTIONS, outletRadius = R_OUTLET,
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
  const rCoreIdeal = R / Math.sqrt(bypassRatio + 1);
  const rCore = Math.min(rCoreIdeal, MAX_RADIUS * R);
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
  const NOSE = -z0;                       // aft extent = NOSE + TAIL * Lc
  const TAIL = 1 + PLUG_TIP;              // core, then the spike beyond it
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

  // Core body. A spline through six stations rather than straight segments:
  // it necks in behind the fan, swells through the turbine and falls away to
  // the outlet with the curvature staying continuous, where a piecewise
  // profile creases at every joint and catches the light as a ring. Only the
  // inlet station is load-bearing -- it is what bypass ratio sets -- the rest
  // are shape.
  // Sorted and clamped rather than trusted: two stations given out of order
  // fold the profile back on itself, and the lathe of a folded profile is a
  // shape with its inside out.
  const secs = sections
    .map((sec) => ({ z: Math.min(0.98, Math.max(0.02, sec.z)),
                     r: Math.max(0.02, sec.r) }))
    .sort((a, b) => a.z - b.z);

  // Outlet wall, and the plug that scales with it. Measured IN THE EXIT
  // PLANE, which is the only place the annulus is visible.
  // Multiples of rCore, so the profile follows bypass ratio; ceilinged
  // against fan radius, so it can never leave the duct.
  const cap = (mult) => Math.min(Math.max(0.02, mult) * rCore, MAX_RADIUS * R);
  const rOutlet = cap(outletRadius);
  const rPlugExit = rOutlet * (1 - EXHAUST_GAP_FRAC);
  // Continuing the same taper forward to the buried base.
  const rPlugBase = rPlugExit * (1 + PLUG_EMBED / PLUG_TIP);

  // Bypass ratio fixes the core radius AT THE FAN DUCT'S BACK FACE -- that is
  // the station where the flow splits, so it is the one the area ratio is
  // about. The core runs at that radius from its front to the duct exit and
  // is only shaped aft of it, which is also where it is visible.
  const zSplit = zCaseAft;
  const Lv = Math.max(0.05 * Lc, zSplit - zAft);       // exposed core length

  const ctrl = [
    new THREE.Vector2(z0, rCore),                      // inside the duct
    new THREE.Vector2(zSplit, rCore),                  // duct back face: BPR
    ...secs.map((sec) => new THREE.Vector2(zSplit - sec.z * Lv, cap(sec.r))),
    new THREE.Vector2(zAft, rOutlet),                  // outlet
  ];
  // Clamp the SAMPLED curve, not just the control points. A Catmull-Rom
  // through points that all obey the cap can still bulge past it between
  // them -- at BPR 0.05 the surface reached 0.99 rFan from controls held at
  // 0.95 -- and it is the surface the cap is about. Where it binds the
  // overshoot flattens into a straight section, which is what a limit looks
  // like.
  const rCap = MAX_RADIUS * R;
  const spline = new THREE.SplineCurve(ctrl).getPoints(48)
    .map((p) => new THREE.Vector2(p.x, Math.min(p.y, rCap)));

  // One mesh, finish painted on. See coreMaterial().
  const core = latheZ([[z0, 0], ...spline.map((p) => [p.x, p.y]), [zAft, 0]],
                      coreMaterial(), SEG);
  core.name = 'core';
  g.add(core);

  // Looking up the exhaust should be looking into a hole. Without this the
  // core's own aft cap is the first thing you meet, lit and metallic. Run a
  // little past the annulus at both edges so it is hidden behind wall and
  // plug rather than ending flush with either.
  const mouth = new THREE.Mesh(
    new THREE.RingGeometry(rPlugExit * 0.93, rOutlet * 1.02, SEG), M.cavity);
  mouth.rotation.y = Math.PI;                  // face aft
  mouth.position.z = zAft - 0.004 * Lc;
  g.add(mouth);

  // Exhaust plug: base buried inside the core, tip out beyond the exit, so it
  // emerges from the body rather than hanging behind it.
  g.add(latheZ([
    [zAft + PLUG_EMBED * Lc, 0], [zAft + PLUG_EMBED * Lc, rPlugBase],
    [zAft - PLUG_TIP * Lc, 0],
  ], M.hot, SEG));

  g.userData.rCore = rCore;
  g.userData.bypassRatio = bypassRatio;
  g.userData.bypassRatioEffective = (R * R - rCore * rCore) / (rCore * rCore);
  g.userData.coreLength = Lc;
  g.userData.noseZ = zNose;
  g.userData.sections = secs;
  g.userData.splitZ = zSplit;
  g.userData.exposedCoreLength = Lv;
  g.userData.outletRadius = rOutlet;
  g.userData.plugExitRadius = rPlugExit;
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
  rotor.userData.spin = -1;
  rotor.add(spinner(0.25 * R, 0.55 * R, M.casing, 0.05 * R));
  // Angle from the axis grows outboard -- coarse root, fine tip. This row was
  // built the other way round, the same inversion the fan had.
  rotor.add(bladeRow({
    count: blades, material: M.blade, hubMaterial: M.casing,
    hubLength: 0.3 * R,
    rHub: 0.27 * R, rTip: 0.95 * R,
    chordRoot: 0.24 * R, chordTip: 0.20 * R,
    twistRoot: 0.45, twistTip: 0.95, thickness: 0.09,
  }));
  g.add(rotor);

  g.add(tubeZ(0.94 * R, R, 0.10 * R, -2.70 * R, M.casing, SEG));

  // Looking down the inlet should end in darkness, not in a lit disc.
  const blank = new THREE.Mesh(
    new THREE.CircleGeometry(0.95 * R, SEG), M.cavity);
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

  // ...and so should looking up the nozzle.
  const mouth = new THREE.Mesh(
    new THREE.RingGeometry(0.05 * R, 0.56 * R, SEG), M.cavity);
  mouth.rotation.y = Math.PI;
  mouth.position.z = -3.33 * R;
  g.add(mouth);

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
  rotor.userData.spin = -1;
  rotor.add(spinner(0.115 * R, 0.30 * R, M.painted, 0.03 * R));
  // A propeller is coarse at the root and fine at the tip, so the angle from
  // the axis grows outboard -- and a prop turns a long way, so the spread is
  // wider than the fan's.
  rotor.add(bladeRow({
    count: blades, material: M.painted, hubMaterial: M.accessory,
    hubLength: 0.13 * R,
    rHub: 0.10 * R, rTip: R,
    chordRoot: 0.15 * R, chordTip: 0.085 * R,
    twistRoot: 0.42, twistTip: 1.20, thickness: 0.11,
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

  const mouth = new THREE.Mesh(
    new THREE.CircleGeometry(0.145 * R, SEG), M.cavity);
  mouth.rotation.y = Math.PI;
  mouth.position.z = -1.578 * R;
  g.add(mouth);

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

  // On an outrunner the case turns, so the can, its fins and the shaft are
  // one rotor and the mount is what stays still.
  const rotor = new THREE.Group();
  rotor.userData.rotating = true;
  rotor.userData.spin = -1;
  g.add(rotor);

  rotor.add(latheZ([
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
    rotor.add(f);
  }

  // Windings glimpsed through the gap between can and rear bell.
  g.add(tubeZ(0.72 * R, 0.9 * R, -0.90 * L, -0.98 * L, M.winding, SEG));

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16 * R, 0.16 * R, 0.55 * R, 20), M.hardware);
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = 0.22 * R;
  rotor.add(shaft);

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

export const ENGINES = {
  turbofan, turbojet, turboprop, electricMotor, pistonEngine,
};
export default ENGINES;
