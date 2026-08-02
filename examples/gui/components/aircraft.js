/**
 * Whole aeroplanes, assembled from the component library.
 *
 * Nothing here draws anything. It converts a **sizing deck** -- the numbers a
 * solve produces -- into the parameters each component wants, and puts the
 * results where the deck says. That conversion is the entire content of this
 * file and it is worth having in one place, because most of it is the kind of
 * arithmetic that is silently wrong: quarter-chord sweep against leading-edge
 * sweep, tail arms measured from one reference and applied to another, tyre
 * diameters in inches sitting beside everything else in metres.
 *
 * Anything the deck does NOT determine is marked. A sizing solve is largely
 * two-dimensional -- it knows a wing's area and where its quarter chord is, and
 * has no opinion about how far down the fuselage the wing box sits -- so the
 * vertical placements here are choices, and are labelled as such rather than
 * being allowed to look like results.
 */
import * as THREE from 'three';
import { jetlinerFuselage } from './fuselage.js';
import { liftingSurface, verticalTail } from './wing.js';
import { turbofan } from './engines.js';
import { underMountPylon } from './pylon.js';
import { landingGear } from './landing_gear.js';

const DEG = Math.PI / 180;
const IN = 0.0254;

/**
 * Turn a solved SPaircraft solution into a deck.
 *
 * Written because typing the numbers out is the failure mode this whole file
 * exists to avoid, and doing it once by hand was already once too many. Hand a
 * solution dictionary in and the geometry comes from it; nothing between the
 * solve and the drawing is retyped.
 *
 * Two things are worked out rather than read, because the solution stores them
 * implicitly:
 *
 * **The wing's planform breaks.** TASOPT's wing is three panels -- a constant
 * carry-through to the side of body at eta_o, a taper to the crank at eta_s,
 * then a taper to the tip -- and the solution gives the chords and the two
 * chord SLOPES rather than the stations. Inverting them recovers the stations:
 * eta_s from the outboard slope and the crank-to-tip chord fall, then eta_o by
 * subtracting the inboard panel's width. Ignoring the carry-through and
 * tapering from the centreline instead loses 2.6% of the reference area.
 *
 * **Nothing else.** Sweeps for the tails are not in the solution -- they are
 * deck constants -- so they stay parameters here and are marked as such.
 */
export function deckFromSolve(sol, overrides = {}) {
  const g = (k) => {
    const v = sol[k];
    if (typeof v !== 'number') throw new Error(`solve has no numeric ${k}`);
    return v;
  };
  const b = g('Wing_b'), cRoot = g('Wing_c_root');
  const cBreak = g('Wing_c_break'), cTip = g('Wing_c_tip');
  const etaBreak = 1 - (cBreak - cTip) / (g('Wing_dc_dy_out') * b / 2);
  const etaRoot = etaBreak - g('Wing_box_deta_inn');

  return {
    ...B737_TASOPT,
    name: 'Boeing 737 (from a solve)',
    fuseLength: g('Fuse_l_fuse'), fuseRadius: g('Fuse_R_fuse'),
    noseLength: g('Fuse_l_nose'), coneLength: g('Fuse_l_cone'),

    wingSpan: b, wingRootChord: cRoot,
    wingEtaRoot: etaRoot, wingKink: etaBreak,
    wingCrankRatio: cBreak / cRoot, wingTipRatio: cTip / cBreak,
    // Given outright here, so no quarter-chord conversion is needed for the
    // wing -- only for the tails, whose sweeps the solution does not carry.
    wingSweepLE: Math.atan(g('Wing_tan_Lambda_LE')) / DEG,
    wingQuarterX: g('Wing_x_w'),

    htSpan: g('HT_b_ht'), htRootChord: g('HT_c_root_ht'),
    htTaper: g('HT_lambda_ht'), htArm: g('HT_l_ht'),

    vtHeight: g('VT_b_vt'), vtRootChord: g('VT_c_root_vt'),
    vtTaper: g('VT_lambda_vt'), vtArm: g('VT_l_vt'),
    /**
     * How many fins. Absent on a conventional solve, which means one.
     *
     * It settles what `VT_S_vt` counts, and the two readings differ by a factor
     * of two. The planform answers it: b_vt, c_root_vt and lambda_vt reproduce
     * S_vt exactly on their own, so S_vt is ONE surface and `n_vt` says how many
     * of them there are -- the total is the product, not the quotient.
     */
    finCount: typeof sol.n_vt === 'number' ? sol.n_vt : 1,
    /**
     * Tail LEADING EDGE stations, when the solve carries them.
     *
     * Unambiguous where an arm is not: `x_w + l_ht` and `x_ht_le + c_root/4`
     * disagree by 0.78 m on the D8, so the arms are measured to a reference
     * this file was guessing at. Null when absent, and the arms are used.
     */
    htLE: typeof sol.x_ht_le === 'number' ? sol.x_ht_le : null,
    vtLE: typeof sol.x_vt_le === 'number' ? sol.x_vt_le : null,

    engineX: g('x_eng'), engineY: g('y_eng'),
    nacelleDia: g('LG_d_nacelle'), nacelleLength: g('l_nacelle'),

    mainX: g('LG_x_m'), mainY: g('LG_y_m'), mainStrut: g('LG_l_m'),
    mainTyreIn: g('LG_d_t_m'),
    noseX: g('LG_x_n'), noseStrut: g('LG_l_n'), noseTyreIn: g('LG_d_t_n'),

    /**
     * The cabin, as the solve describes it.
     *
     * `x_shell1` and `x_shell2` bound the PRESSURE SHELL, which is where the
     * passengers are and therefore where windows can be. They come out equal to
     * the nose length and to the start of the tailcone on this solve, which is
     * not a coincidence and not something to rely on -- a solve that moved a
     * bulkhead would separate them, and reading the shell directly means the
     * windows follow it rather than following the barrel.
     *
     * Seat pitch is NOT a variable in the solve; it is the shell divided by the
     * rows, which comes out at 0.8637 m -- 34.0 inches, to a tenth. That it
     * lands on a round number in the units seats are actually specified in is
     * the check that this is the right reading of the two.
     */
    /**
     * The section. A tube has w = R and no web; a double bubble has both a
     * lobe radius and a web, and is wider than it is tall. Carried for every
     * solve because both files hold all of them, and reading them is what tells
     * one configuration from the other.
     */
    fuseHalfWidth: g('Fuse_w_fuse'), fuseHalfHeight: g('Fuse_h_fuse'),
    lobeRadius: g('Fuse_R_fuse'), webHalfWidth: g('Fuse_w_db'),

    cabinStart: g('Fuse_x_shell1'), cabinEnd: g('Fuse_x_shell2'),
    cabinRows: g('Fuse_n_rows'), passengers: g('Fuse_n_pass'),
    seatsAbreast: g('Fuse_n_pass') / g('Fuse_n_rows'),
    seatPitch: g('Fuse_l_shell') / g('Fuse_n_rows'),

    /** What the solve said its own areas were, to check the assembly against. */
    solvedAreas: { wing: g('Wing_S'), horizontalTail: g('HT_S_ht'),
                   // All the fins together, which is what an assembly has.
                   verticalTail: (typeof sol.n_vt === 'number' ? sol.n_vt : 1) * g('VT_S_vt') },
    ...overrides,
  };
}

/**
 * A Boeing 737-800, from the SPaircraft `optimal737` deck.
 *
 * Solved with `spaircraft/conventional.py` -- an optimised airframe on the
 * TASOPT D8.2 core, not the delivered aeroplane -- and every number below is
 * read straight off that solution. The sweeps are the deck's own constants
 * (`SWEEP_W`, `SWEEP_HT`, `SWEEP_VT`), which are QUARTER-CHORD angles.
 *
 * The areas are not listed because they are not inputs: span and chords
 * determine them, and the assembled surfaces reproduce Wing_S 116.649,
 * HT_S_ht 19.095 and VT_S_vt 15.859 to five figures. That is the check that the
 * conversion below is right.
 */
export const B737_TASOPT = {
  name: 'Boeing 737-800 (SPaircraft optimal737)',

  // Fuselage. l_nose + l_shell + l_cone = l_fuse exactly.
  fuseLength:     36.5787,   // Fuse_l_fuse
  fuseRadius:      1.8548,   // Fuse_R_fuse
  noseLength:      6.0960,   // Fuse_l_nose
  coneLength:      6.1827,   // Fuse_l_cone

  // Wing. A plain trapezoid: the deck has no crank.
  wingSpan:       35.8140,   // Wing_b
  wingRootChord:   5.6645,   // Wing_c_root
  wingEtaRoot:     0.0000,   // carry-through end; 0 means none
  wingKink:         null,    // crank station; null means a plain trapezoid
  wingCrankRatio:  1.0000,   // crank chord / root
  wingTipRatio:    0.1500,   // tip chord / crank (/ root without a crank)
  wingSweepLE:      29.03,   // degrees, at the LEADING edge
  wingQuarterX:   16.5458,   // Wing_x_w -- root quarter chord, aft of the nose

  // Horizontal tail.
  htSpan:          8.7396,   // HT_b_ht
  htRootChord:     3.4959,   // HT_c_root_ht
  htTaper:         0.2500,   // HT_lambda_ht
  htSweepC4:        25.0,    // SWEEP_HT
  htArm:          17.0451,   // HT_l_ht, wing to tail

  // Vertical tail. b_vt is a HEIGHT and S_vt is one surface: A_vt = b^2/S = 2.
  vtHeight:        5.6319,   // VT_b_vt
  vtRootChord:     4.3322,   // VT_c_root_vt
  vtTaper:         0.3000,   // VT_lambda_vt
  vtSweepC4:        25.0,    // SWEEP_VT
  vtArm:          15.9482,   // VT_l_vt

  // Engines.
  engineX:        17.6323,   // x_eng
  engineY:         4.8768,   // y_eng
  nacelleDia:      1.8831,   // LG_d_nacelle
  nacelleLength:   2.5842,   // l_nacelle

  // Undercarriage. Tyre diameters are in INCHES in the deck; everything else
  // is metres. Converting them here rather than at the call site is the whole
  // reason this table exists.
  mainX:          18.7609,   // LG_x_m
  mainY:           4.8768,   // LG_y_m
  mainStrut:       3.1176,   // LG_l_m
  mainTyreIn:     42.5135,   // LG_d_t_m
  noseX:           6.0960,   // LG_x_n
  noseStrut:       2.1909,   // LG_l_n
  noseTyreIn:     34.0108,   // LG_d_t_n

  // Cabin. Seat pitch is derived, not read: l_shell / n_rows.
  cabinStart:      5.1820,   // Fuse_x_shell1
  cabinEnd:       31.0920,   // Fuse_x_shell2
  cabinRows:         30.0,   // Fuse_n_rows
  passengers:       180.0,   // Fuse_n_pass
  seatsAbreast:       6.0,   // n_pass / n_rows
  seatPitch:      0.86367,   // l_shell / n_rows

  /* ---- not from the deck -------------------------------------------- */
  // A sizing solve is two-dimensional about these. They are choices.
  wingRootY:      -0.70,     // wing root chord height, in fuselage radii
  engineDrop:      1.42,     // engine axis below the local wing, metres
  rearSpar:        0.65,     // chord fraction the pylon's aft attachment takes
  htRaise:         0.38,     // horizontal tail above the local body centre, radii
  wingDihedral:     3.0,     // degrees
  htDihedral:       6.0,
  tcRoot: 0.135, tcCrank: 0.120, tcTip: 0.105,   // wing thickness
  tcTail: 0.10,                  // both tails
};

/**
 * Leading-edge sweep of a surface quoted at its quarter chord.
 *
 * The chord falls linearly along the span, so the leading edge sweeps back
 * FASTER than the quarter chord by a quarter of that fall:
 *
 *     tan(LE) = tan(c/4) + k * cRoot * (1 - taper) / span
 *
 * with k = 1/2 for a surface whose `span` is tip to tip and 1/4 for a fin,
 * whose span is a height. Getting that factor wrong on the fin moves its
 * leading edge by most of a metre and looks entirely plausible.
 */
function leadingEdgeSweep(sweepC4, rootChord, taper, span, mirrored = true) {
  const k = mirrored ? 0.5 : 0.25;
  return Math.atan(Math.tan(sweepC4 * DEG) + k * rootChord * (1 - taper) / span) / DEG;
}

/** Assemble an aeroplane from a deck. Returns a group, nose tip at the origin. */
export function conventionalAircraft(deck = B737_TASOPT, opts = {}) {
  const d = { ...B737_TASOPT, ...deck };
  const g = new THREE.Group();
  const R = d.fuseRadius;
  const parts = {};

  /* ---- fuselage ------------------------------------------------------- */
  const fuse = jetlinerFuselage({
    radius: R,
    fineness: d.fuseLength / (2 * R),
    noseD: d.noseLength / (2 * R),
    shape: { tailD: d.coneLength / (2 * R) },
    detail: opts.detail ?? false,
  });
  g.add(fuse); parts.fuselage = fuse;
  const u = fuse.userData;

  /* ---- wing ----------------------------------------------------------- */
  // The deck's wing is a plain trapezoid, so `kink: null` and the taper is the
  // ordinary tip-over-root. A real 737's cranked trailing edge is not in the
  // solve and is not invented here.
  /**
   * The inboard planform: one straight taper, not a carry-through and a break.
   *
   * The solve puts the wing-box break at eta_root, which works out at the
   * fuselage's full RADIUS -- and that is only where a wing leaves the body if
   * it is mounted at mid height. This one is low, as a 737's is, so it emerges
   * where the section is narrower: 1.392 m against the break's 1.855. That left
   * 0.46 m of constant-chord carry-through in the open, ending in a 26 degree
   * corner in the trailing edge that read as the wing clipping into the body.
   *
   * So the mould line runs a single taper from the centreline to the crank,
   * with a root chord chosen to enclose exactly the area the carry-through and
   * taper enclosed between them. Everything outboard of the crank is untouched,
   * the wing sits where it sat -- placement below still uses the solve's
   * c_root, which is what x_w is quoted against -- and the areas still
   * reproduce the solve.
   *
   * What it costs is that the built root chord is no longer Wing_c_root. That
   * is deliberate and worth being plain about: c_root is a structural number
   * about the wing box, this is the outer mould line, and the two only have to
   * agree where the box is actually inside the aerofoil. The area, which is the
   * aerodynamic quantity, is preserved exactly.
   */
  const cKinkSolve = d.wingRootChord * d.wingCrankRatio;
  const inboardArea = d.wingRootChord * d.wingEtaRoot
    + 0.5 * (d.wingRootChord + cKinkSolve) * (d.wingKink - d.wingEtaRoot);
  const straightRoot = 2 * inboardArea / d.wingKink - cKinkSolve;

  const wing = liftingSurface({
    span: d.wingSpan,
    rootChord: straightRoot,
    etaRoot: 0,
    kink: d.wingKink,
    // Restated against the new root so the crank chord itself does not move.
    crankRatio: cKinkSolve / straightRoot,
    tipRatio: d.wingTipRatio,
    taperRatio: d.wingKink == null ? d.wingTipRatio : null,
    sweep: d.wingSweepLE,
    dihedral: d.wingDihedral,
    twistRoot: 0, twistTip: -3,
    rootThickness: d.tcRoot, crankThickness: d.tcCrank, tipThickness: d.tcTip,
    root: '2412', kinkFoil: null, tip: '2410',
  });
  const wingRootLE = d.wingQuarterX - 0.25 * d.wingRootChord;
  const wingY = d.wingRootY * R;
  wing.position.set(0, wingY, -wingRootLE);
  g.add(wing); parts.wing = wing;

  /* ---- horizontal tail ------------------------------------------------ */
  // The arm runs from the wing's reference station to the tail's, so it is
  // added to the same station it was measured from.
  const ht = liftingSurface({
    kink: null,
    span: d.htSpan,
    rootChord: d.htRootChord,
    taperRatio: d.htTaper,
    sweep: leadingEdgeSweep(d.htSweepC4, d.htRootChord, d.htTaper, d.htSpan),
    dihedral: d.htDihedral,
    twistRoot: 0, twistTip: 0,
    thickness: d.tcTail, symmetric: true,
  });
  const htQuarterX = d.wingQuarterX + d.htArm;
  const htRootLE = htQuarterX - 0.25 * d.htRootChord;
  // Vertically: above the body's own section centre on the tailcone. The deck
  // has no opinion -- a tail arm is a longitudinal number -- so the raise is a
  // choice like the wing height, and marked as one.
  const htY = u.shapeAt(-htQuarterX).yc + d.htRaise * R;
  ht.position.set(0, htY, -htRootLE);
  g.add(ht); parts.horizontalTail = ht;

  /* ---- vertical tail -------------------------------------------------- */
  const vt = verticalTail({
    height: d.vtHeight,
    rootChord: d.vtRootChord,
    taperRatio: d.vtTaper,
    sweep: leadingEdgeSweep(d.vtSweepC4, d.vtRootChord, d.vtTaper, d.vtHeight, false),
    thickness: d.tcTail,
    cant: 0,
  });
  const vtQuarterX = d.wingQuarterX + d.vtArm;
  const vtRootLE = vtQuarterX - 0.25 * d.vtRootChord;
  vt.position.set(0, u.crownAt(-vtQuarterX), -vtRootLE);
  g.add(vt); parts.verticalTail = vt;

  /* ---- engines and pylons --------------------------------------------- */
  // Sized so the finished nacelle matches the deck's diameter: the podded
  // turbofan reports its own rMax, so the fan radius is scaled to hit it rather
  // than guessed and checked afterwards.
  const probe = turbofan({ rFan: 1, bypassRatio: 9 });
  const rFan = (d.nacelleDia / 2) / probe.userData.rMax;
  const wingAtEngine = wingY + d.engineY * Math.tan(d.wingDihedral * DEG);
  const engineY = wingAtEngine - d.engineDrop;

  // The pylon's top chord is the WING's, at the engine's own station: forward
  // attachment on the leading edge, aft on the rear spar. Taking it from the
  // wing rather than from a fraction of the fan radius is what keeps the two
  // agreeing when either moves -- a pylon whose top is a fixed multiple of the
  // engine slides along the wing every time the engine is resized.
  // Both taken from the WING rather than recomputed, so a cranked planform and
  // a carry-through are handled without this knowing about either.
  const etaEngine = 2 * d.engineY / d.wingSpan;
  const wingLeAt = wingRootLE + wing.userData.at(etaEngine).xLE;
  const chordAtEngine = wing.userData.at(etaEngine).chord;
  const topFwd = d.engineX - wingLeAt;                  // pod frame, +Z forward
  const topAft = topFwd - d.rearSpar * chordAtEngine;

  parts.engines = [];
  for (const side of [1, -1]) {
    const pod = new THREE.Group();
    const eng = turbofan({ rFan, bypassRatio: 9 });
    pod.add(eng);
    // Built in the pod's frame with the engine at its origin, so the wing is
    // however far above that the two heights differ by. The pylon takes its
    // stations in fan radii, hence the division.
    pod.add(underMountPylon(eng, {
      engineZ: 0,
      attachY: (wingAtEngine - engineY) / rFan,
      topZ0: topFwd / rFan,
      topZ1: topAft / rFan,
    }));
    pod.position.set(side * d.engineY, engineY, -d.engineX);
    // Tagged so a viewer can find a pod from a click without knowing how the
    // assembly is put together.
    pod.userData.isEnginePod = true;
    pod.userData.side = side;
    pod.name = side > 0 ? 'starboardPod' : 'portPod';
    g.add(pod); parts.engines.push(pod);
  }

  /* ---- undercarriage --------------------------------------------------- */
  // Both legs attach where they physically would -- main to the wing, nose to
  // the fuselage keel -- and where the wheels then land is a RESULT. Placing
  // the nose gear at whatever height makes it reach the main gear's ground
  // would hide the one thing worth knowing here: whether the deck's two strut
  // lengths agree about the attitude the aeroplane sits at.
  const mainTyre = d.mainTyreIn * IN / 2;
  const mkMain = () => landingGear({
    wheels: 2, rTire: mainTyre, rWheel: mainTyre * 0.55,
    lStrut: d.mainStrut, rStrut: 0.10,
  });
  const mainAttachY = wingY + d.mainY * Math.tan(d.wingDihedral * DEG);
  const mainContact = mainAttachY + mkMain().userData.contactY;
  parts.gear = [];
  for (const side of [1, -1]) {
    const lg = mkMain();
    lg.position.set(side * d.mainY, mainAttachY, -d.mainX);
    // Which way this leg retracts is a property of where it is, so it is
    // recorded here rather than re-derived by whoever animates it.
    lg.userData.gearKind = 'main';
    lg.userData.side = side;
    lg.userData.retract = { axis: 'z', angle: -side * Math.PI / 2 };
    lg.name = side > 0 ? 'starboardGear' : 'portGear';
    g.add(lg); parts.gear.push(lg);
  }
  const noseTyre = d.noseTyreIn * IN / 2;
  const nose = landingGear({
    wheels: 2, rTire: noseTyre, rWheel: noseTyre * 0.55,
    lStrut: d.noseStrut, rStrut: 0.075,
  });
  const noseAttachY = u.keelAt(-d.noseX);
  nose.position.set(0, noseAttachY, -d.noseX);
  nose.userData.gearKind = 'nose';
  nose.userData.side = 0;
  // Forward, not sideways: a nose leg swings about the lateral axis into a bay
  // ahead of it. Negative about X takes the leg from hanging down to pointing
  // forward.
  nose.userData.retract = { axis: 'x', angle: -Math.PI / 2 };
  nose.name = 'noseGear';
  g.add(nose); parts.gear.push(nose);
  const noseContact = noseAttachY + nose.userData.contactY;

  // The two legs do not reach the same plane, so the aeroplane sits at an
  // angle. Rotating the whole assembly about the nose tip by that angle puts
  // every wheel on one ground plane; the angle itself is reported, because it
  // is a statement about the deck rather than about the drawing.
  const wheelbase = d.mainX - d.noseX;
  const pitch = Math.atan2(noseContact - mainContact, wheelbase);
  if (opts.sitOnGround !== false) g.rotation.x = pitch;
  const ground = mainContact + d.mainX * Math.sin(pitch);

  Object.assign(g.userData, {
    deck: d, parts, ground,
    /**
     * Static attitude, positive nose-up: the angle at which the deck's two
     * strut lengths put the aeroplane on its wheels. Zero would mean the two
     * agree exactly, which nothing in the solve requires them to.
     */
    groundAttitude: pitch / DEG,
    wheelbase, mainContact, noseContact,
    /** Areas the assembled surfaces came out at -- compare against the solve. */
    areas: {
      wing: wing.userData.area,
      horizontalTail: ht.userData.area,
      verticalTail: vt.userData.area,
    },
    leadingEdgeSweeps: {
      wing: wing.userData.sweep, horizontalTail: ht.userData.sweep,
      verticalTail: vt.userData.sweep,
    },
    /** Where the pylon meets the wing, in the pod's frame -- taken from the
     *  wing's own leading edge and rear spar, not from the engine's size. */
    pylonTop: { forward: topFwd, aft: topAft, chordAtEngine },
    noseGearAttachY: noseAttachY,
  });
  return g;
}

export const aircraft = { boeing737: conventionalAircraft };
