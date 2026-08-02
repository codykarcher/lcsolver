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
import { turbofan } from './engines.js';
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
  engineSink:     0.45,   // nacelle radius let into the upper skin
  htOnFins:       true,   // tailplane carried on the fin tips, not the body
  wingDihedral:   2.0,
  wingSweepC4:   20.0,
  htSweepC4:     22.0,
  vtSweepC4:     30.0,
  tcRoot: 0.145, tcCrank: 0.125, tcTip: 0.105, tcTail: 0.09,
};

/**
 * Twin fins, from a solve that describes one surface.
 *
 * The solve gives a single fin -- its span, root chord and taper reproduce its
 * own area exactly -- but the configuration carries two. Splitting the AREA and
 * holding the aspect ratio and taper is the reading that keeps the number the
 * solve actually cares about: two fins of half the area each, so the total is
 * what was sized. Building two of the given planform would silently double it.
 */
export function splitFin({ height, rootChord, taper }) {
  const area = 0.5 * height * rootChord * (1 + taper);
  const ar = (height * height) / area;
  const each = area / 2;
  const h = Math.sqrt(ar * each);
  return { height: h, rootChord: (2 * each) / (h * (1 + taper)), taper, area: each };
}

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
  const fuse = d8Fuselage({
    radius: halfH,
    length: d.fuseLength,
    noseD: d.noseLength / (2 * halfH),
    shape: {
      cabinWidth: halfW / halfH,
      noseWidth: halfW / halfH,
      tailD: d.coneLength / (2 * halfH),
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

  /* ---- twin fins ------------------------------------------------------ */
  const fin = splitFin({ height: d.vtHeight, rootChord: d.vtRootChord, taper: d.vtTaper });
  const vtQuarterX = d.wingQuarterX + d.vtArm;
  const vtRootLE = vtQuarterX - 0.25 * fin.rootChord;
  parts.verticalTails = [];
  for (const side of [1, -1]) {
    const vt = verticalTail({
      height: fin.height, rootChord: fin.rootChord, taperRatio: fin.taper,
      sweep: leadingEdgeSweep(d.vtSweepC4, fin.rootChord, fin.taper, fin.height, false),
      thickness: d.tcTail,
      // Positive cant leans the tip to +x, so the port fin takes the negative
      // of it and the pair splay outboard rather than both leaning one way.
      cant: side * d.finCant,
    });
    const y = side * d.finY * halfW;
    vt.position.set(y, u.crownAt(-vtQuarterX), -vtRootLE);
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
  const htQuarterX = d.wingQuarterX + d.htArm;
  const htRootLE = htQuarterX - 0.25 * d.htRootChord;
  // Level with the fin tips when it rides on them, which is what makes the
  // empennage a pi rather than a cross.
  const finTipY = u.crownAt(-vtQuarterX) + fin.height * Math.cos(d.finCant * DEG);
  const htY = d.htOnFins ? finTipY : u.crownAt(-htQuarterX);
  ht.position.set(0, htY, -htRootLE);
  g.add(ht); parts.horizontalTail = ht;

  /* ---- engines, on top of the body between the fins ------------------- */
  // Let into the upper skin rather than hung under a wing: a D8's nacelles sit
  // on the afterbody, which is the whole point of the configuration.
  const probe = turbofan({ rFan: 1, bypassRatio: 9 });
  const rFan = (d.nacelleDia / 2) / probe.userData.rMax;
  const rNac = d.nacelleDia / 2;
  parts.engines = [];
  for (const side of [1, -1]) {
    const pod = new THREE.Group();
    pod.add(turbofan({ rFan, bypassRatio: 9 }));
    pod.position.set(side * d.engineY,
                     u.crownAt(-d.engineX) + rNac * (1 - d.engineSink),
                     -d.engineX);
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
      // Both fins together, which is what the solve sized.
      verticalTail: 2 * parts.verticalTails[0].userData.area,
    },
    fin, wheelbase, groundAttitude: attitude, ground,
    noseContact, mainContact,
    leadingEdgeSweeps: {
      wing: wing.userData.sweep,
      horizontalTail: ht.userData.sweep,
      verticalTail: parts.verticalTails[0].userData.sweep,
    },
  });
  return g;
}
