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
  ductGap:        0.03,   // clearance between the lid and the engines
  /**
   * How far the section centre stays below the lid.
   *
   * The centre has to be under the roof it describes, and it descends more
   * slowly than the lid does -- at 0.45 the lid overtook it just before the
   * engine face and that station read as empty. 0.80 keeps it clear the whole
   * way; it costs nothing visible, since the keel is a metre and a half below
   * either way.
   */
  lidClearance:   0.80,
  /**
   * How squarely the afterbody holds its depth aft.
   *
   * Lower than it was. With the centreline pushed down to stay under the lid,
   * holding the depth late leaves the body to close in the last stretch and the
   * underside necks in at 65 degrees; 0.8 spreads the close and brings it to 35.
   */
  tailHold:       0.8,
  tailTrough:     0.30,   // valley between the lobes, in local half-heights
  troughWidth:    0.55,   // angular half-width of that valley, radians
  htOnFins:       true,   // tailplane carried on the fin tips, not the body
  wingDihedral:   2.0,
  wingSweepC4:   20.0,
  htSweepC4:     22.0,
  vtSweepC4:     30.0,
  tcRoot: 0.145, tcCrank: 0.125, tcTip: 0.105, tcTail: 0.09,
};

/** Leading-edge sweep from a quarter-chord one. `k` is 1/2 tip-to-tip, 1/4 for a fin. */
function leadingEdgeSweep(sweepC4, rootChord, taper, span, mirrored = true) {
  const k = mirrored ? 0.5 : 0.25;
  return Math.atan(Math.tan(sweepC4 * DEG) + (k * rootChord * (1 - taper)) / span) / DEG;
}

export function d8Aircraft(deck, opts = {}) {
  const d = { ...D8_CHOICES, ...deck };
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
      // The plan narrows on its own law, and earlier than the depth, so the
      // afterbody is no wider than what it has to carry.
      tailWidthA: d.planTaperA, tailWidthB: d.planTaperB,
      /**
       * The lid: the top of the afterbody, descending from the cabin's crown to
       * just under the engines by the time it reaches their face, and holding
       * there. The engines then sit ON it rather than in a hole cut through it,
       * and their fairings are their own lofted pieces.
       */
      lid: {
        top: d.fuseHalfHeight,
        floor: engineAxisY - d.nacelleDia / 2 - d.ductGap,
        fromX: d.cabinEnd,
        toX: engineNoseX,
      },
      /**
       * And the body's centreline stays BELOW that lid.
       *
       * A section is one radius per angle about the centreline, so the centre
       * has to be under the roof it describes. Aimed into the engines -- which
       * is what it needed when they were cradled in the body -- it ends up
       * above the lid and every ray reads as empty.
       */
      tailEdgeHeight: ((engineAxisY - d.nacelleDia / 2 - d.ductGap - d.lidClearance)
                       + halfH) / (2 * halfH),
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

  /* ---- undercarriage --------------------------------------------------- */
  const mainTyre = d.mainTyreIn * IN / 2;
  const mkMain = () => landingGear({
    wheels: 2, rTire: mainTyre, rWheel: mainTyre * 0.55,
    lStrut: d.mainStrut, rStrut: 0.10,
  });
  // The main legs hang from the BODY, not the wing: a D8 stows them in the
  // fuselage, which is why the deck's y_m sits inboard of the wing's root.
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
