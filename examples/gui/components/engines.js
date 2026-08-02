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
  g.userData.plugBaseRadius = rPlugBase;
  g.userData.overallLength = zNose + NOSE + TAIL * Lc;
  g.userData.lengthOverDiameter = g.userData.overallLength / (2 * R);
  return finish(g, NOSE + TAIL * Lc, rCaseOut, 'turbofan');
}

/* ==================================================================== *
 * Turbojet
 * ==================================================================== */

/** Turbojet slenderness: nose-to-tail over CASE diameter. */
const JET_LENGTH_OVER_DIAMETER = 4.5;

/** Plug tip, as a fraction of body length aft of the nozzle exit. */
const JET_PLUG_TIP = 0.10;

/** Case wall thickness, x rCase. */
const JET_WALL = 0.035;

/**
 * Case profile stations between the inlet lip and the nozzle exit.
 *
 *     z   fraction of body length aft of the lip
 *     r   radius as a multiple of rCase
 *
 * Defaults hold the case parallel and let it converge only at the end, which
 * is what a plain turbojet looks like.
 */
const JET_SECTIONS = [{ z: 0.35, r: 0.95 }, { z: 0.70, r: 1.0 }];

/** Nozzle exit radius, x rCase. */
const JET_EXIT = 0.90;

/**
 * Turbojet: an annular casing with a compressor face at the inlet and a
 * converging nozzle aft.
 *
 * The casing is one CLOSED ANNULUS -- outer wall aft, inner wall forward,
 * joined round the lip -- rather than a single-sided shell. That makes the
 * bore genuinely open, so you see the compressor down it, while the part
 * itself is still a solid with no boundary edge anywhere.
 *
 * A dark liner just inside the bore does the rest: an inlet should read as a
 * hole, and bare metal all the way down reads as a pipe.
 */
export function turbojet({
  rCase = 0.42, length = null, nozzleExit = JET_EXIT,
  sections = JET_SECTIONS, blades = 26,
} = {}) {
  const g = new THREE.Group();
  const R = rCase;
  const t = JET_WALL * R;

  // Nose is the inlet lip at z = 0, tail is the plug tip, so overall length is
  // simply the extent aft -- the spinner sits inside the inlet and never
  // reaches forward of the lip.
  const L = length ?? JET_LENGTH_OVER_DIAMETER * 2 * R;
  const Lb = L / (1 + JET_PLUG_TIP);            // lip to nozzle exit
  const zExit = -Lb;

  const secs = sections
    .map((sec) => ({ z: Math.min(0.97, Math.max(0.03, sec.z)),
                     r: Math.max(0.05, sec.r) }))
    .sort((a, b) => a.z - b.z);

  // The case is held parallel until the compressor is behind it: a spline
  // straight from the lip to the first station starts curving inward at the
  // lip itself, narrowing the case over the very blades it houses.
  const lSpin = 0.62 * R;
  const zFanAft = Math.max(-(lSpin + 0.20 * R), -0.5 * secs[0].z * Lb);

  const rExit = Math.max(0.1, nozzleExit) * R;
  const ctrl = [
    new THREE.Vector2(0, R),
    new THREE.Vector2(zFanAft, R),                 // still parallel here
    ...secs.map((sec) => new THREE.Vector2(-sec.z * Lb, sec.r * R)),
    new THREE.Vector2(zExit, rExit),
  ];
  const wall = new THREE.SplineCurve(ctrl).getPoints(40);

  /** Case inner radius at an axial station, read off the wall curve. */
  const wallInnerAt = (z) => {
    for (let i = 0; i < wall.length - 1; i++) {
      const a = wall[i], b = wall[i + 1];
      if (z <= a.x && z >= b.x) {
        const f = (a.x - z) / ((a.x - b.x) || 1);
        return a.y + (b.y - a.y) * f - t;
      }
    }
    return wall[wall.length - 1].y - t;
  };

  // Closed annulus: down the outside, back up the inside, shut at the lip.
  const prof = [
    ...wall.map((p) => [p.x, p.y]),
    ...[...wall].reverse().map((p) => [p.x, Math.max(0.02 * R, p.y - t)]),
  ];
  prof.push(prof[0]);
  const casing = latheZ(prof, M.casing, SEG);
  casing.name = 'casing';
  g.add(casing);

  // Dark liner inside the bore.
  g.add(latheZ([
    ...wall.map((p) => [p.x, Math.max(0.02 * R, p.y - t)]),
    ...[...wall].reverse().map((p) => [p.x, Math.max(0.01 * R, p.y - 1.8 * t)]),
    [0, R - t],
  ], M.cavity, SEG));

  // Compressor face, set back inside the lip.
  const rotor = new THREE.Group();
  rotor.userData.rotating = true;
  rotor.userData.spin = -1;
  // Set back so the spinner tip stays inside the lip. Referenced to the
  // spinner's own length, not to body length: tied to the body it crept out
  // through the inlet as the engine got shorter, and nose-to-tail then
  // measured from the spinner rather than from the lip.
  rotor.position.z = -(lSpin + 0.06 * R);
  rotor.add(spinner(0.26 * R, lSpin, spinnerMaterial(), 0));
  rotor.add(bladeRow({
    count: blades, material: M.blade, hubMaterial: M.casing,
    hubLength: 0.30 * R,
    rHub: 0.27 * R, rTip: 0.93 * R,
    chordRoot: 0.24 * R, chordTip: 0.20 * R,
    twistRoot: 0.45, twistTip: 0.95, thickness: 0.09,
  }));
  g.add(rotor);

  // Blanking disc a little way behind the compressor, so the inlet ends in
  // darkness rather than a view straight through the engine.
  const blank = new THREE.Mesh(
    new THREE.CircleGeometry(0.94 * R, SEG), M.cavity);
  blank.position.z = -0.22 * Lb;
  blank.rotation.y = Math.PI;
  g.add(blank);

  // Exhaust plug, base buried inside the nozzle so it grows out of the body.
  const rPlugExit = rExit * (1 - EXHAUST_GAP_FRAC);
  const zPlugTip = zExit - JET_PLUG_TIP * Lb;
  const embed = 0.06 * Lb;
  // Continuing the cone's taper forward would put its base at 1.11 x the exit
  // radius -- wider than the hole it comes out of, so it burst through the
  // casing. Held inside the case wall at the station where it actually sits.
  const rPlugBase = Math.min(
    rPlugExit * (1 + embed / (JET_PLUG_TIP * Lb)),
    0.90 * wallInnerAt(zExit + embed));
  g.add(latheZ([
    [zExit + embed, 0], [zExit + embed, rPlugBase], [zPlugTip, 0],
  ], M.hot, SEG));

  const mouth = new THREE.Mesh(
    new THREE.RingGeometry(rPlugExit * 0.9, rExit - t * 0.5, SEG), M.cavity);
  mouth.rotation.y = Math.PI;
  mouth.position.z = zExit + 0.004 * Lb;
  g.add(mouth);

  g.userData.bodyLength = Lb;
  g.userData.exitRadius = rExit;
  g.userData.plugExitRadius = rPlugExit;
  g.userData.plugBaseRadius = rPlugBase;
  g.userData.sections = secs;
  g.userData.overallLength = L;
  g.userData.lengthOverDiameter = L / (2 * R);
  return finish(g, L, R, 'turbojet');
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

/** Visual proportions, all x bore. Not physical. */
const PE = {
  barrel:   0.58,   // barrel radius
  fin:      0.84,   // cooling fin radius
  finThick: 0.055,
  cylLen:   1.35,   // case side to head face
  caseHalf: 0.72,   // crankcase half-width
  caseHigh: 1.45,   // crankcase height
  spacing:  1.55,   // between cylinders on one bank
  stagger:  0.20,   // fore-aft offset between the two banks
  head:     0.95,   // head block, across
  shaft:    0.17,   // propeller shaft radius
  flange:   0.46,   // propeller flange radius
  sump:     0.55,   // sump depth below the case
};

/**
 * Crude Lycoming-like sizing: shaft power in, cylinder count and bore out.
 *
 * Fitted by eye to the published lineup (O-235, O-320, O-360, IO-390, IO-540,
 * IO-580, IO-720), and it wants only two observations:
 *
 * 1. Power tracks displacement at about half a horsepower per cubic inch --
 *    0.47 at the bottom of the range, 0.55 at the top, so 0.52 splits it.
 *
 * 2. Lycoming barely changes bore. Almost the whole family runs 5.125 in and
 *    grows by ADDING CYLINDERS: the O-360, IO-540 and IO-720 are four, six and
 *    eight cylinders of very nearly the same 90 cu in pot. So cylinder count
 *    steps with power and bore only trims the remainder.
 *
 * With bore/stroke 1.17 (also Lycoming's, 5.125/4.375), a cylinder is
 * (pi/4) b^2 (b/1.17) = 0.671 b^3, which inverts for the bore.
 *
 * Accurate to a few percent against the real engines, which is far better than
 * this needs to be:
 *
 *     O-235   115 hp -> 4 cyl, 4.35 in   (real 4.375)
 *     O-360   180 hp -> 4 cyl, 5.05 in   (real 5.125)
 *     IO-540  300 hp -> 6 cyl, 5.23 in   (real 5.125)
 *     IO-720  400 hp -> 8 cyl, 5.23 in   (real 5.125)
 *
 * @param {number} power  shaft horsepower
 * @returns {{cylinders:number, bore:number, boreInches:number,
 *            displacement:number}}  bore in metres, displacement in cu in
 */
export function lycomingSizing(power) {
  const hp = Math.max(20, power);
  // Steps taken from where the family actually changes: the four-cylinder
  // range tops out at the IO-390, six spans the O-540s, eight is the IO-720.
  const cylinders = hp < 225 ? 4 : hp < 360 ? 6 : 8;
  const displacement = hp / 0.52;                       // cubic inches
  const boreInches = Math.cbrt((displacement / cylinders) / 0.671);
  return { cylinders, displacement, boreInches, bore: boreInches * 0.0254 };
}

/**
 * Air-cooled horizontally opposed piston engine, without a propeller.
 *
 * The cylinders stick straight out to port and starboard with their cooling
 * fins showing, which is the whole silhouette of a light-aircraft engine. Fins
 * are a stack of discs -- crude, but a fin is two triangles' worth of
 * information and a stack of them reads correctly at any size you would draw
 * this at.
 *
 * Sized on POWER: cylinder count and bore come from `lycomingSizing`, and
 * everything else is proportioned to bore. Pass `cylinders` or `bore`
 * explicitly to override either.
 *
 * The banks are staggered fore-and-aft, as they are in life: opposed cylinders
 * share a crank throw, so they cannot sit in the same plane.
 */
export function pistonEngine({
  power = 180, cylinders = null, bore = null, finsPer = 7,
} = {}) {
  const sized = lycomingSizing(power);
  cylinders = cylinders ?? sized.cylinders;
  bore = bore ?? sized.bore;
  if (cylinders % 2) throw new Error('pistonEngine: cylinders must be even');
  const g = new THREE.Group();
  const B = bore;
  const perSide = cylinders / 2;

  const caseLen = (perSide - 1) * PE.spacing * B + 2.0 * B;
  const zCaseFront = -0.30 * B;
  const zCaseBack = zCaseFront - caseLen;

  // Propeller shaft and flange. Tagged as the rotor so a viewer has something
  // to turn even with no propeller fitted.
  const rotor = new THREE.Group();
  rotor.userData.rotating = true;
  rotor.userData.spin = -1;
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(PE.shaft * B, PE.shaft * B, 0.62 * B, 20),
    M.hardware);
  shaft.rotation.x = Math.PI / 2;
  shaft.position.z = zCaseFront + 0.31 * B;
  rotor.add(shaft);
  rotor.add(tubeZ(PE.shaft * B, PE.flange * B,
                  -0.02 * B, -0.13 * B, M.hardware, 28));
  g.add(rotor);

  // Crankcase, with a nose that tapers to the shaft.
  const crank = new THREE.Mesh(
    new THREE.BoxGeometry(2 * PE.caseHalf * B, PE.caseHigh * B, caseLen),
    M.accessory);
  crank.position.z = zCaseFront - caseLen / 2;
  g.add(crank);

  const nose = latheZ([
    [zCaseFront + 0.16 * B, 0], [zCaseFront + 0.16 * B, PE.flange * B],
    [zCaseFront - 0.10 * B, 0.78 * B], [zCaseFront - 0.10 * B, 0],
  ], M.accessory, 28);
  g.add(nose);

  // Sump slung under the case.
  const sump = new THREE.Mesh(
    new THREE.BoxGeometry(1.15 * PE.caseHalf * B, PE.sump * B, caseLen * 0.78),
    M.accessory);
  sump.position.set(0, -(PE.caseHigh * 0.5 + PE.sump * 0.45) * B,
                    zCaseFront - caseLen / 2);
  g.add(sump);

  const xIn = PE.caseHalf * B;
  const xOut = xIn + PE.cylLen * B;
  const finGeo = new THREE.CylinderGeometry(PE.fin * B, PE.fin * B,
                                            PE.finThick * B, 20);
  const barGeo = new THREE.CylinderGeometry(PE.barrel * B, PE.barrel * B,
                                            xOut - xIn, 20);
  const headGeo = new THREE.BoxGeometry(0.42 * B, PE.head * B, PE.head * B);

  for (let i = 0; i < perSide; i++) {
    const z = zCaseFront - 0.95 * B - i * PE.spacing * B;
    for (const side of [-1, 1]) {
      const zs = z - (side < 0 ? PE.stagger * B : 0);

      const bar = new THREE.Mesh(barGeo, M.finned);
      bar.rotation.z = Math.PI / 2;
      bar.position.set(side * (xIn + xOut) / 2, 0, zs);
      g.add(bar);

      for (let f = 0; f < finsPer; f++) {
        const t = (f + 0.5) / finsPer;
        const fin = new THREE.Mesh(finGeo, M.finned);
        fin.rotation.z = Math.PI / 2;
        fin.position.set(side * (xIn + t * (xOut - xIn)), 0, zs);
        g.add(fin);
      }

      const head = new THREE.Mesh(headGeo, M.accessory);
      head.position.set(side * (xOut + 0.19 * B), 0, zs);
      g.add(head);

      // Exhaust stub down and aft, and an intake riser up and forward. Two
      // small pipes, but they are most of what stops a finned cylinder
      // looking like a stack of washers.
      const ex = rod(new THREE.Vector3(side * xOut, -0.15 * B, zs),
                     new THREE.Vector3(side * (xOut + 0.10 * B),
                                       -0.85 * B, zs - 0.30 * B),
                     0.13 * B, M.hot, 14);
      if (ex) g.add(ex);
      const inl = rod(new THREE.Vector3(side * (xOut - 0.15 * B), 0.30 * B, zs),
                      new THREE.Vector3(side * (xIn + 0.15 * B),
                                        0.92 * B, zs + 0.10 * B),
                      0.11 * B, M.accessory, 14);
      if (inl) g.add(inl);
    }
  }

  // Accessory case on the back.
  const acc = new THREE.Mesh(
    new THREE.BoxGeometry(1.5 * PE.caseHalf * B, PE.caseHigh * 0.85 * B,
                          0.55 * B),
    M.accessory);
  acc.position.z = zCaseBack - 0.27 * B;
  g.add(acc);

  const length = -(zCaseBack - 0.55 * B);
  g.userData.power = power;
  g.userData.bore = B;
  g.userData.boreInches = B / 0.0254;
  g.userData.cylinders = cylinders;
  g.userData.displacement = sized.displacement;
  g.userData.width = 2 * (xOut + 0.40 * B);
  g.userData.height = (PE.caseHigh + PE.sump) * B + 0.92 * B;
  return finish(g, length, xOut + 0.40 * B, 'pistonEngine');
}

export const ENGINES = {
  turbofan, turbojet, turboprop, electricMotor, pistonEngine,
};
export default ENGINES;
