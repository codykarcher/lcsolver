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
  wingTaper:       0.1500,   // Wing_lambda, tip over root
  wingSweepC4:      26.0,    // SWEEP_W, degrees
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

  /* ---- not from the deck -------------------------------------------- */
  // A sizing solve is two-dimensional about these. They are choices.
  wingRootY:      -0.70,     // wing root chord height, in fuselage radii
  engineDrop:      1.42,     // engine axis below the local wing, metres
  wingDihedral:     3.0,     // degrees
  htDihedral:       6.0,
  tcRoot: 0.135, tcTip: 0.105,   // wing thickness
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
  const wing = liftingSurface({
    kink: null,
    span: d.wingSpan,
    rootChord: d.wingRootChord,
    taperRatio: d.wingTaper,
    sweep: leadingEdgeSweep(d.wingSweepC4, d.wingRootChord, d.wingTaper, d.wingSpan),
    dihedral: d.wingDihedral,
    twistRoot: 0, twistTip: -3,
    rootThickness: d.tcRoot, tipThickness: d.tcTip,
    root: '2412', tip: '2410',
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
  // Vertically: on the tailcone, at the body's own section centre there.
  const htY = u.shapeAt(-htQuarterX).yc;
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
  parts.engines = [];
  for (const side of [1, -1]) {
    const pod = new THREE.Group();
    const eng = turbofan({ rFan, bypassRatio: 9 });
    pod.add(eng);
    // The pylon is built in the pod's frame with the engine at its origin, so
    // the wing underside is however far above that the two heights differ by.
    pod.add(underMountPylon(eng, {
      engineZ: 0,
      attachY: (wingAtEngine - engineY) / rFan,
    }));
    pod.position.set(side * d.engineY, engineY, -d.engineX);
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
    g.add(lg); parts.gear.push(lg);
  }
  const noseTyre = d.noseTyreIn * IN / 2;
  const nose = landingGear({
    wheels: 2, rTire: noseTyre, rWheel: noseTyre * 0.55,
    lStrut: d.noseStrut, rStrut: 0.075,
  });
  const noseAttachY = u.keelAt(-d.noseX);
  nose.position.set(0, noseAttachY, -d.noseX);
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
    noseGearAttachY: noseAttachY,
  });
  return g;
}

export const aircraft = { boeing737: conventionalAircraft };
