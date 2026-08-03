/**
 * A D8 assembled from a sizing solve.
 *
 * The same parts as the conventional aeroplane, arranged the way the D8 puts
 * them: a wide flat double-bubble body, a low wing, TWIN fins on the aft
 * upper body with the engines between them, and the horizontal tail carried on
 * top of the fins.
 *
 * Everything the solve determines is read from it and nothing is retyped. What
 * the solve does not determine -- where the wing sits on the body's height, how
 * far apart the fins stand, how deeply the nacelles are let into the upper
 * skin -- is a choice, and is collected in D8_CHOICES where it can be seen to
 * be one.
 */
import * as THREE from 'three';
import { d8Fuselage } from './fuselage.js';
import { liftingSurface, verticalTail } from './wing.js';
import { bareTurbofan } from './engines.js';
import { landingGear } from './landing_gear.js';
import { carveDuct, carveInto } from './carve.js';

const DEG = Math.PI / 180;
const IN = 0.0254;

/**
 * The numbers a sizing solve is two-dimensional about.
 *
 * Read against the general arrangement rather than invented: the fins stand
 * outboard of the nacelles, the nacelles sit half into the upper skin, and the
 * tailplane bridges the fin tips.
 */
export const D8_CHOICES = {
  wingRootY:     -0.55,   // wing root chord height, in body half-heights
  finY:           0.62,   // fin root, as a fraction of the body's half-width
  finCant:       14.0,    // degrees outboard from vertical
  /**
   * Engine axis height, as a fraction of the BODY's height up from its keel.
   *
   * Referred to the cabin's height rather than the local one: the afterbody is
   * closing where the engines sit, and a fraction of a collapsing number is not
   * a place. Set against the arrangement drawing, not derived.
   */
  engineHeight:   0.70,
  /**
   * The afterbody's trailing-edge half-width, in engine outer extents.
   *
   * One puts the walls of the channel exactly on the outside of the engines.
   * The cradle needs this: where the body runs wider than what it holds, a ray
   * out of the section crosses skin, the opening, then skin again, and a
   * section written as one radius per angle cannot say that.
   */
  tailSpan:       1.00,
  planTaperA:     1.0,    // the plan narrows EARLY -- the depth does not
  planTaperB:     2.0,
  runStart:       1.5,    // where the run begins, in tailcone lengths off the tail
  tailHold:       4.0,    // how squarely the afterbody holds its depth aft
  tailTrough:     0.30,   // valley between the lobes, in local half-heights
  troughWidth:    0.55,   // angular half-width of that valley, radians
  htOnFins:       true,   // tailplane carried on the fin tips, not the body
  wingDihedral:   2.0,
  wingSweepC4:   20.0,
  htSweepC4:     22.0,
  vtSweepC4:     30.0,
  tcRoot: 0.145, tcCrank: 0.125, tcTip: 0.105, tcTail: 0.09,
  /**
   * Carve the engine duct out of the afterbody.
   *
   * A toggle, and defaulted on, because the shape underneath it is the one
   * that was arrived at by hand and is not to be disturbed: turning this off
   * gives back exactly that body, vertex for vertex, rather than some earlier
   * approximation of it. See `carve.js` for why this cannot be a section.
   */
  ductCarve:      true,
  /** Clearance between the duct's walls and the engines they hold. */
  ductGap:        0.04,
  /**
   * How deep into the body the duct's lip is blended, in metres.
   *
   * Zero gives the bare cut, which meets the skin at a right angle along the
   * median of the rim and folds back on itself at 158 degrees at the worst.
   *
   * Small, because the thing it is rounding is small: where the walls come
   * through the upper skin they are a few centimetres tall, so a radius of the
   * 0.30 m that first looked right swallowed the wall whole and came out worse
   * than no blend at all.
   *
   * 0.08 is the largest that still closes. The flare moves the cut outboard, and
   * past this the duct's own sheet starts running out of body to be clipped
   * against before the skin's cut edge does: at 0.10 ten of the seam's edges
   * open up and the worst gap goes from 18 to 45 mm.
   */
  ductBlend:      0.08,
  /**
   * Mesh density, as a multiplier on every count that drives it.
   *
   * One knob rather than four, because the four have to move together: a fine
   * skin carved against a coarse duct sheet is not a better aeroplane, it is
   * the same seam mismatch with more triangles on one side of it.
   *
   * 1.0 is the density everything above was judged at. It costs about half a
   * second a build, nearly all of it the carve, which is fine for looking at
   * one aeroplane and too slow to sit inside a solver's loop.
   */
  quality:        1.0,
};

/** Leading-edge sweep from a quarter-chord one. `k` is 1/2 tip-to-tip, 1/4 for a fin. */
function leadingEdgeSweep(sweepC4, rootChord, taper, span, mirrored = true) {
  const k = mirrored ? 0.5 : 0.25;
  return Math.atan(Math.tan(sweepC4 * DEG) + (k * rootChord * (1 - taper)) / span) / DEG;
}

export function d8Aircraft(deck, opts = {}) {
  const d = { ...D8_CHOICES, ...deck };
  /**
   * How finely everything is meshed, from one number.
   *
   * The skin's stations and segments, and the duct sheet's grid, all scale
   * together. Rounded to even numbers because the section's meshing pairs
   * points across the symmetry plane and an odd count puts a seam down the
   * middle of the aeroplane.
   */
  const q = Math.max(0.15, opts.quality ?? d.quality);
  const grain = (n) => 2 * Math.max(3, Math.round((n * q) / 2));
  const g = new THREE.Group();
  const parts = {};

  /* ---- fuselage ------------------------------------------------------- */
  // `radius` is the body's HALF-HEIGHT and `cabinWidth` its width over that,
  // which is exactly the pair the solve reports.
  const halfH = d.fuseHalfHeight, halfW = d.fuseHalfWidth;
  // 55% of the body's height up from its keel.
  const engineAxisY = -halfH + d.engineHeight * 2 * halfH;
  /**
   * Where the engine's nose actually falls, worked out before the body is
   * built, because the body's run has to finish there.
   *
   * MEASURED, not `engineX - length/2`: the turbofan is not centred on its own
   * origin, so half its length forward of the station is 1.28 m ahead of its
   * real leading edge -- and the run finished that far early, flattening off
   * before it reached the thing it was climbing to meet.
   */
  const probe0 = bareTurbofan({ rFan: 1, bypassRatio: 9 });
  const rFanFor = (d.nacelleDia / 2) / probe0.userData.rMax;
  const engineNoseX = d.engineX - new THREE.Box3()
    .setFromObject(bareTurbofan({ rFan: rFanFor, bypassRatio: 9 })).max.z;
  const fuse = d8Fuselage({
    nSeg: grain(64),
    nStation: grain(140),
    radius: halfH,
    length: d.fuseLength,
    noseD: d.noseLength / (2 * halfH),
    shape: {
      cabinWidth: halfW / halfH,
      noseWidth: halfW / halfH,
      tailD: d.coneLength / (2 * halfH),
      /**
       * The afterbody narrows in plan until its trailing edge just spans the
       * engines.
       *
       * Left at the cabin's width -- which is what the standalone body does,
       * and is right for a body with nothing on the back of it -- the planform
       * is a constant-width slab and the engines sit on a shelf that runs on
       * past them to either side. Taking the width from the engines instead
       * makes the back of the aeroplane end where they do, so the afterbody
       * reads as the thing carrying them rather than as a slab they happen to
       * be on. Derived from the deck, not chosen: it is the engine's outer
       * extent, in the half-heights this parameter is measured in.
       */
      tailWidth: (d.tailSpan * (d.engineY + d.nacelleDia / 2)) / halfH,
      /**
       * The afterbody holds its DEPTH back to the engines, then closes.
       *
       * The smoothstep the bare body uses falls away from the moment the cabin
       * ends, which is right when there is nothing to carry: it had the body
       * 0.23 m deep where the engines sit, 12 per cent of full, so they perched
       * on an edge and the dish they are supposed to sit in had no room to
       * exist. A power law holds it at 77 per cent there and does its closing
       * in the last fifth instead.
       */
      tailLaw: 'power', tailA: d.tailHold, tailB: 0.70,
      /**
       * Where the afterbody's centreline ends up: in the CHANNEL.
       *
       * A section is one radius per angle about that centreline, so the
       * centreline has to be inside the shape it describes. Left at its own
       * default the body closes at 0.35 of its height while the channel, with
       * the engines at 0.70, floors at 0.47 -- the origin ends up below the
       * floor, outside the section, and the whole afterbody collapses to
       * nothing. Aiming it at the middle of the channel is not a tuning
       * choice; it is what keeps the description valid.
       */
      tailEdgeHeight: ((engineAxisY - (d.nacelleDia / 2 + 0.02) / 2) + halfH) / (2 * halfH),
      // The plan narrows on its own law, and earlier than the depth, so the
      // afterbody is no wider than the engines by the time it has to hold them.
      tailWidthA: d.planTaperA, tailWidthB: d.planTaperB,
      // And the afterbody closes into the channel that holds the engines: a
      // flat floor with sides rounding up at their own radius.
      channel: { x: d.engineY, y: engineAxisY, r: d.nacelleDia / 2 },
      /**
       * The run starts 1.5 tailcone-lengths off the tail -- so it begins well
       * forward of the cone, where there is length to climb in -- and runs
       * straight through to the body's TRAILING EDGE. Every metre it is given
       * is a degree it does not have to climb at: over the cone alone the keel
       * swept up at 48 degrees, to the engines' nose at 23, and to the trailing
       * edge at less again.
       */
      channelFromX: d.fuseLength - d.runStart * d.coneLength,
      channelToX: d.fuseLength,
    },
    detail: opts.detail ?? false,
  });
  g.add(fuse); parts.fuselage = fuse;
  const u = fuse.userData;

  /* ---- wing ----------------------------------------------------------- */
  // One straight taper inboard, area preserved, for the same reason as the
  // conventional aeroplane: the solve's wing-box break sits outboard of where a
  // low wing actually leaves a body, and the corner would be in the open.
  const cKink = d.wingRootChord * d.wingCrankRatio;
  const inboard = d.wingRootChord * d.wingEtaRoot
    + 0.5 * (d.wingRootChord + cKink) * (d.wingKink - d.wingEtaRoot);
  const straightRoot = (2 * inboard) / d.wingKink - cKink;

  const wing = liftingSurface({
    span: d.wingSpan, rootChord: straightRoot, etaRoot: 0,
    kink: d.wingKink, crankRatio: cKink / straightRoot, tipRatio: d.wingTipRatio,
    sweep: d.wingSweepLE, dihedral: d.wingDihedral,
    twistRoot: 0, twistTip: -3,
    rootThickness: d.tcRoot, crankThickness: d.tcCrank, tipThickness: d.tcTip,
    root: '2412', kinkFoil: null, tip: '2410',
  });
  const wingRootLE = d.wingQuarterX - 0.25 * d.wingRootChord;
  const wingY = d.wingRootY * halfH;
  wing.position.set(0, wingY, -wingRootLE);
  g.add(wing); parts.wing = wing;

  /* ---- the fins ------------------------------------------------------- */
  /**
   * Each fin is the solve's planform, unhalved, and there are `n_vt` of them.
   *
   * The solve settles this itself and it is worth being precise about, because
   * the two readings differ by a factor of two: b_vt, c_root_vt and lambda_vt
   * reproduce S_vt exactly on their own, so S_vt describes ONE surface and n_vt
   * says how many there are. The total is the product. Splitting the area
   * instead -- which is what this did before the solve carried n_vt -- builds
   * two fins that together come to what one was supposed to be.
   */
  const fin = {
    height: d.vtHeight, rootChord: d.vtRootChord, taper: d.vtTaper,
    area: 0.5 * d.vtHeight * d.vtRootChord * (1 + d.vtTaper),
  };
  // The solve gives the leading edge outright, which is the station that
  // actually places the surface; the arm is a moment length measured to a
  // reference this file would have to guess at.
  const vtRootLE = d.vtLE ?? (d.wingQuarterX + d.vtArm - 0.25 * fin.rootChord);
  const vtQuarterX = vtRootLE + 0.25 * fin.rootChord;
  /**
   * The fins are hung by their root TRAILING EDGE, on the body's back upper
   * corners.
   *
   * Which is where the two actually meet: the solve puts the fin's trailing
   * edge on the body's own trailing edge, and the channel puts the body's upper
   * corners there too, so the corner is a single point both of them already
   * own. Hanging them at the crown over their QUARTER chord instead -- which is
   * a station a good way forward, where the afterbody is still much deeper --
   * floated them well above it.
   *
   * The root's leading edge ends up inside the body, which is correct: the fin
   * is faired in forward and emerges aft.
   */
  const teCornerY = u.crownAt(-d.fuseLength);
  const teCornerX = u.halfWidthAt(-d.fuseLength);
  parts.verticalTails = [];
  // Placed symmetrically about the centreline, whatever the count.
  const sides = d.finCount >= 2 ? [1, -1] : [0];
  for (const side of sides) {
    const vt = verticalTail({
      height: fin.height, rootChord: fin.rootChord, taperRatio: fin.taper,
      sweep: leadingEdgeSweep(d.vtSweepC4, fin.rootChord, fin.taper, fin.height, false),
      thickness: d.tcTail,
      // Positive cant leans the tip to +x, so the port fin takes the negative
      // of it and the pair splay outboard rather than both leaning one way.
      cant: side * d.finCant,
    });
    vt.position.set(side * teCornerX, teCornerY, -vtRootLE);
    vt.name = side > 0 ? 'starboardFin' : 'portFin';
    vt.userData.side = side;
    g.add(vt); parts.verticalTails.push(vt);
  }

  /* ---- tailplane, carried on the fin tips ----------------------------- */
  const ht = liftingSurface({
    kink: null, span: d.htSpan, rootChord: d.htRootChord, taperRatio: d.htTaper,
    sweep: leadingEdgeSweep(d.htSweepC4, d.htRootChord, d.htTaper, d.htSpan),
    dihedral: 0, twistRoot: 0, twistTip: 0,
    thickness: d.tcTail, symmetric: true,
  });
  const htRootLE = d.htLE ?? (d.wingQuarterX + d.htArm - 0.25 * d.htRootChord);
  const htQuarterX = htRootLE + 0.25 * d.htRootChord;
  // Level with the fin tips when it rides on them, which is what makes the
  // empennage a pi rather than a cross.
  const finTipY = teCornerY + fin.height * Math.cos(d.finCant * DEG);
  const htY = d.htOnFins ? finTipY : u.crownAt(-htQuarterX);
  ht.position.set(0, htY, -htRootLE);
  g.add(ht); parts.horizontalTail = ht;

  /* ---- engines, on top of the body between the fins ------------------- */
  // Let into the upper skin rather than hung under a wing: a D8's nacelles sit
  // on the afterbody, which is the whole point of the configuration.
  // BARE, not podded. On a D8 the nacelle is not a separate body slung under a
  // wing -- the afterbody is the fairing, and the engine is let into it. A
  // podded turbofan brings its own cowl and reads as an engine parked on the
  // fuselage rather than built into it.
  const probe = bareTurbofan({ rFan: 1, bypassRatio: 9 });
  const rFan = (d.nacelleDia / 2) / probe.userData.rMax;
  // Length from the deck too, not left to the component's own proportions. A
  // D8's propulsor is short and fat -- 1.24 m long on a 1.68 m diameter, an
  // aspect of 0.74 where a podded engine is nearer 1.8 -- because the duct is
  // short when the fan is fed off the body. Letting the component choose gives
  // a 3.2 m engine that hangs a metre and a half past the tail.
  const rNac = d.nacelleDia / 2;
  parts.engines = [];
  for (const side of [1, -1]) {
    const pod = new THREE.Group();
    pod.add(bareTurbofan({ rFan, bypassRatio: 9 }));
    pod.position.set(side * d.engineY, engineAxisY, -d.engineX);
    pod.userData.isEnginePod = true;
    pod.userData.side = side;
    pod.name = side > 0 ? 'starboardPod' : 'portPod';
    g.add(pod); parts.engines.push(pod);
  }

  /* ---- the duct the engines sit in ------------------------------------ */
  /**
   * Cut AFTER everything is placed, and out of the mesh rather than out of the
   * section, so the body's shape is bit-for-bit what it was and everything
   * hung on it -- fins, tailplane, engines, decals -- still sits where it did.
   *
   * The duct's size is the engines', not a choice: its walls round up at their
   * radius, stand a clearance outboard of their outer extent, and its floor is
   * a clearance below them. It runs from the trailing edge, where it is
   * deepest, forward to the end of the cabin, where it closes on the crown and
   * the body is untouched.
   */
  if (opts.ductCarve ?? d.ductCarve) {
    const duct = carveDuct(fuse, {
      floor: engineAxisY - rNac - d.ductGap,
      /**
       * The walls stand exactly on the engines' outer extent -- no clearance,
       * where the floor gets `ductGap`.
       *
       * Because `tailSpan` is 1.00 there is nothing to spare. The body's
       * trailing edge is as wide as the engines and no wider, and the fins hang
       * off its back upper corners, 20 mm outboard of that. A duct even 40 mm
       * wider takes those corners with it and the fin roots come out floating
       * 189 mm above the trough with nothing under them. So the wall goes on
       * the engine line and the last 20 mm of body is what carries the fins.
       *
       * The engines then touch the wall at their equator. That is the whole
       * clearance the configuration has: a wider duct costs the fins their
       * mounting, and a wider BODY is a change to the shape.
       */
      halfWidth: d.engineY + rNac,
      cornerR: rNac + d.ductGap,
      toX: d.fuseLength,
      fromX: -u.cabinZ[1],
      /**
       * Level from the engines' NOSE aft, not just at the trailing edge. The
       * ramp is what closes the duct into the cabin roof, and if it is still
       * descending where the engines are then the floor cuts through them: at
       * the engine's own station it stood 270 mm above the bottom of the fan,
       * and a third of the nacelle stayed buried.
       */
      deepFrom: engineNoseX - d.ductGap,
      crown: u.crownAt(u.cabinZ[1]),
      blend: opts.ductBlend ?? d.ductBlend,
      nx: grain(160),
      nu: grain(72),
    });
    /**
     * The cut runs on through the fins.
     *
     * The fins stand 20 mm outboard of the duct's wall but they are 184 mm
     * thick at the root, so their inboard face pokes 92 mm through the wall,
     * 310 mm tall and 2 m long -- a ridge down the inside of the trough that
     * belongs to neither surface.
     *
     * Cut by the duct AND the body together, not by the duct alone. The duct's
     * wall is a plane with no top to it, so on its own it would go on slicing
     * the fin all the way to the tailplane; what wants removing is only what is
     * inside the body as well, which is exactly the part the carve exposed. The
     * hole it leaves sits in the plane of the wall, with the wall's own sheet
     * across it, so there is nothing to see through.
     */
    // The body FIRST. Only the fin's root is inside it, so for almost every
    // triangle three queries settle the matter, where asking the duct's two
    // faces first spent nine on the same answer.
    for (const vt of parts.verticalTails) {
      carveInto(vt, [(p) => u.depthInside(p.x, p.y, p.z), ...duct.faces]);
    }
  }

  /* ---- undercarriage --------------------------------------------------- */
  const mainTyre = d.mainTyreIn * IN / 2;
  const mkMain = () => landingGear({
    wheels: 2, rTire: mainTyre, rWheel: mainTyre * 0.55,
    lStrut: d.mainStrut, rStrut: 0.10,
  });
  /**
   * The main legs hang from the body's KEEL, where a D8 stows them.
   *
   * Which datum the solve sized its struts against is not in the solve, so it
   * has to be read off them. The test is that an aeroplane stands on its
   * wheels: whichever attachment puts all of them in one plane is the one the
   * strut lengths were written for.
   *
   * This solve's main leg reaches 0.158 m LESS than its nose leg and the keel
   * is level between the two stations, so keel-hung leaves 158 mm and 0.64
   * degrees. Hung from the wing instead -- 0.57 m above the keel there -- it
   * comes out 875 mm and 3.56 degrees nose-down.
   *
   * The previous solve said the opposite, and emphatically: its main leg
   * reached 0.881 m FURTHER than the nose, which no keel-hung leg could
   * absorb, and the wing was almost exactly that far above the keel. So this is
   * not a free choice being tuned -- the deck states it, and it has changed its
   * mind about where a D8 puts its gear. Left on the wing this aeroplane stands
   * on its nosewheel with the mains in the air.
   */
  const mainAttachY = u.keelAt(-d.mainX);
  const mainContact = mainAttachY + mkMain().userData.contactY;
  parts.gear = [];
  for (const side of [1, -1]) {
    const lg = mkMain();
    lg.position.set(side * d.mainY, mainAttachY, -d.mainX);
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
  nose.userData.retract = { axis: 'x', angle: -Math.PI / 2 };
  nose.name = 'noseGear';
  g.add(nose); parts.gear.push(nose);
  const noseContact = noseAttachY + nose.userData.contactY;

  /* ---- how it sits ----------------------------------------------------- */
  const wheelbase = d.mainX - d.noseX;
  const attitude = Math.atan((noseContact - mainContact) / wheelbase) / DEG;
  const ground = Math.min(noseContact, mainContact);
  if (opts.sitOnGround ?? true) g.position.y = -ground;

  Object.assign(g.userData, {
    isD8: true, deck: d, parts,
    areas: {
      wing: wing.userData.area,
      horizontalTail: ht.userData.area,
      // Every fin together, which is what the deck's solvedAreas counts.
      verticalTail: parts.verticalTails.reduce((t, f) => t + f.userData.area, 0),
    },
    fin, finCount: parts.verticalTails.length, engineAxisY,
    wheelbase, groundAttitude: attitude, ground,
    noseContact, mainContact,
    leadingEdgeSweeps: {
      wing: wing.userData.sweep,
      horizontalTail: ht.userData.sweep,
      verticalTail: parts.verticalTails[0].userData.sweep,
    },
  });
  return g;
}
