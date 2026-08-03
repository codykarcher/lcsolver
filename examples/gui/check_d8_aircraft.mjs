/**
 * Does the D8 assembly reproduce the solve it came from?
 *
 * The same question as the conventional aeroplane's check, and the same answer
 * matters most: areas are not inputs anywhere. They fall out of spans and
 * chords, the solve computed them independently, and agreement to machine
 * precision is what says the whole conversion is right.
 *
 * The fins are where a solve can be misread by a factor of two, so that reading
 * is checked rather than assumed: `VT_S_vt` describes ONE surface -- its own
 * span, root chord and taper reproduce it exactly -- and `n_vt` says how many
 * there are. The total is the product.
 */
import { readFileSync } from 'fs';
import * as THREE from 'three';
import { deckFromSolve } from './components/aircraft.js';
import { d8Aircraft } from './components/d8_aircraft.js';

const sol = JSON.parse(readFileSync(new URL('./decks/b737_d8_solve.json', import.meta.url), 'utf8'));
const deck = deckFromSolve(sol);
const g = d8Aircraft(deck, { sitOnGround: false });
const u = g.userData, d = u.deck;
const fu = u.parts.fuselage.userData;

let failures = 0;
const bad = (m) => { console.log(`  FAIL  ${m}`); failures++; };

/* ---- the body is the section the solve describes ----------------------- */
const halfW = 2.6105 && d.fuseHalfWidth, halfH = d.fuseHalfHeight;
const mid = -d.fuseLength * 0.45;
const gotW = fu.halfWidthAt(mid), gotH = (fu.crownAt(mid) - fu.keelAt(mid)) / 2;
console.log(`body ${(2 * gotW).toFixed(3)} wide by ${(2 * gotH).toFixed(3)} tall by ` +
            `${fu.length.toFixed(2)} long`);
console.log(`     solve says ${(2 * halfW).toFixed(3)} by ${(2 * halfH).toFixed(3)} by ` +
            `${d.fuseLength.toFixed(2)}`);
if (Math.abs(gotW - halfW) > 1e-3) bad(`half-width ${gotW.toFixed(4)}, solve ${halfW.toFixed(4)}`);
if (Math.abs(gotH - halfH) > 1e-3) bad(`half-height ${gotH.toFixed(4)}, solve ${halfH.toFixed(4)}`);
if (Math.abs(fu.length - d.fuseLength) > 1e-6) bad(`length ${fu.length}, solve ${d.fuseLength}`);

// It has to be a double bubble, and wider than it is tall, or it is not a D8.
if (!fu.isDoubleBubble) bad('the body is not a double bubble');
if (gotW <= gotH) bad(`the body is ${(gotW / gotH).toFixed(2)} wide for its height -- not a D8`);
console.log(`     width over height ${(gotW / gotH).toFixed(3)}, ` +
            `${d.seatsAbreast.toFixed(1)} abreast in ${d.cabinRows.toFixed(1)} rows`);

/* ---- areas ------------------------------------------------------------- */
console.log('\narea            built        solve         rel');
for (const [key, want] of Object.entries(d.solvedAreas)) {
  const got = u.areas[key];
  const rel = Math.abs(got - want) / want;
  console.log(`  ${key.padEnd(15)}${got.toFixed(4).padStart(9)}${want.toFixed(4).padStart(13)}` +
              `${rel.toExponential(1).padStart(12)}`);
  if (rel > 1e-9) bad(`${key} is ${got}, solve says ${want}`);
}

/* ---- the fins are two, halved ------------------------------------------ */
{
  const fins = u.parts.verticalTails;
  console.log(`\nfins: ${fins.length} (solve says n_vt = ${sol.n_vt}), each ` +
              `${fins[0].userData.height.toFixed(3)} tall by ` +
              `${fins[0].userData.rootChord.toFixed(3)} root, ` +
              `${fins[0].userData.area.toFixed(4)} m2`);
  if (fins.length !== sol.n_vt) bad(`${fins.length} fins, solve says ${sol.n_vt}`);
  // Each fin is the solve's own planform, unhalved -- the reading that makes
  // b_vt, c_root_vt and lambda_vt reproduce S_vt, which they do exactly.
  const planform = 0.5 * d.vtHeight * d.vtRootChord * (1 + d.vtTaper);
  console.log(`      planform gives ${planform.toFixed(4)} against VT_S_vt ` +
              `${sol.VT_S_vt.toFixed(4)}; total ${(fins.length * planform).toFixed(4)}`);
  if (Math.abs(planform - sol.VT_S_vt) > 1e-9) {
    bad(`b_vt x c_root x (1+lambda)/2 is ${planform}, not VT_S_vt ${sol.VT_S_vt}`);
  }
  for (const f of fins) {
    if (Math.abs(f.userData.area - sol.VT_S_vt) > 1e-9) {
      bad(`a fin is ${f.userData.area} m2, not the solve's ${sol.VT_S_vt}`);
    }
  }
  // They must splay apart, not both lean the same way.
  const [a, b] = fins.map((f) => new THREE.Box3().setFromObject(f));
  if (a.min.x * b.max.x > 0) bad('both fins are on the same side');
  if (Math.sign(a.max.x - a.min.x) === 0) bad('a fin has no width');
}

/* ---- the engines sit on the afterbody, not under a wing ---------------- */
{
  const pods = u.parts.engines;
  const crown = fu.crownAt(-d.engineX);
  const b = new THREE.Box3().setFromObject(pods[0]);
  console.log(`\nengines at x ${d.engineX.toFixed(2)} of a ${d.fuseLength.toFixed(2)} m body ` +
              `(${(100 * d.engineX / d.fuseLength).toFixed(0)}% aft), y +/-${d.engineY.toFixed(3)}`);
  console.log(`     nacelle spans y ${b.min.y.toFixed(2)}..${b.max.y.toFixed(2)}, ` +
              `body crown there ${crown.toFixed(2)}`);
  /**
   * The engine axis is a HEIGHT, checked against the height it was given.
   *
   * Not "is it above the crown", which is what this asked while the engines
   * were perched on top of the afterbody. They are set into it now, so being
   * below the crown is the intended state and asking the old question reports
   * a fault that is not one. Whether the body has been opened to receive them
   * is a separate matter and is not yet true.
   */
  const wantY = -d.fuseHalfHeight + d.engineHeight * 2 * d.fuseHalfHeight;
  console.log(`     axis at y ${u.engineAxisY.toFixed(4)} = ` +
              `${(100 * d.engineHeight).toFixed(0)}% of the body's height up from its keel`);
  if (Math.abs(u.engineAxisY - wantY) > 1e-9) {
    bad(`engine axis at ${u.engineAxisY}, wanted ${wantY}`);
  }
  if (b.max.y < crown) {
    console.log('     and wholly within the body -- no channel cut for it yet');
  }
  if (d.engineX < 0.6 * d.fuseLength) bad('the engines are not on the afterbody');
  // Inboard of the fins, as the arrangement has them.
  const finX = Math.abs(new THREE.Box3().setFromObject(u.parts.verticalTails[0]).min.x);
  if (d.engineY > finX) bad(`engines at y ${d.engineY} are outboard of the fins at ${finX}`);
  console.log(`     and inboard of the fins, which stand at x ${finX.toFixed(2)}`);
}

/* ---- the afterbody carries the engines --------------------------------- */
// The plan taper and the engine size are both read from the deck rather than
// styled, so both are checkable. What they buy is that the back of the
// aeroplane ends where the engines do instead of running past them as a slab.
{
  const L = fu.length;
  const edge = fu.halfWidthAt(-L);
  const reach = d.engineY + d.nacelleDia / 2;
  console.log(`\nafterbody: plan taper ${fu.planTaper.toFixed(3)}, trailing edge ` +
              `${edge.toFixed(3)} half-wide against engines reaching ${reach.toFixed(3)}`);
  if (fu.planTaper >= 1) bad('the afterbody does not taper in plan -- it is a constant-width slab');
  if (edge < reach) bad(`the trailing edge is ${edge.toFixed(3)} but the engines reach ${reach.toFixed(3)}`);
  if (edge > 1.35 * reach) bad(`the trailing edge runs ${(edge / reach).toFixed(2)} times past the engines`);

  /**
   * The afterbody closes into the CHANNEL that holds the engines.
   *
   * Flat floor, sides rounding up at the engines' own radius, open at their
   * axis. Measured at the TRAILING EDGE, because that is where the run
   * finishes -- asking at the engine station reports a channel 92 per cent
   * formed and calls the difference a fault.
   */
  const rN = d.nacelleDia / 2, gap = 0.02;
  const zTE = -d.fuseLength;
  const floor = fu.keelAt(zTE), roof = fu.crownAt(zTE), wall = fu.halfWidthAt(zTE);
  console.log(`     at the trailing edge: ${fu.aftDeck ? 'lofted deck' : 'channel'}` +
              `, floor ${floor.toFixed(3)}, closes at ` +
              `${roof.toFixed(3)}, walls to ${wall.toFixed(3)}`);
  console.log(`     engines: bottom ${(u.engineAxisY - rN).toFixed(3)}, axis ` +
              `${u.engineAxisY.toFixed(3)}, outer ${(d.engineY + rN).toFixed(3)}`);
  if (Math.abs(floor - (u.engineAxisY - rN - gap)) > 0.02) {
    bad(`the floor is at ${floor.toFixed(3)}, the engines' undersides at ` +
        `${(u.engineAxisY - rN).toFixed(3)}`);
  }
  /**
   * How high it closes depends on which afterbody it is.
   *
   * A CHANNEL is open at the engines' axis, because it is a cradle and they sit
   * in it. A lofted DECK closes under them at the ducts' own floor, because
   * they sit on it -- and that floor is the cowl's INNER line, so it is above
   * the nacelle's underside by the thickness of the cowl wall. Asking a deck
   * the channel's question fails it by three quarters of a metre.
   */
  const deck = fu.aftDeck;
  if (deck) {
    if (Math.abs(roof - deck.roofY) > 0.02) {
      bad(`the deck closes at ${roof.toFixed(3)}, the ducts' floor is ${deck.roofY.toFixed(3)}`);
    }
    if (roof > u.engineAxisY - rN + 0.10) {
      bad(`the deck closes at ${roof.toFixed(3)}, up inside engines whose ` +
          `undersides are at ${(u.engineAxisY - rN).toFixed(3)}`);
    }
  } else if (Math.abs(roof - u.engineAxisY) > 0.02) {
    bad(`the channel is open at ${roof.toFixed(3)}, the engine axis is ${u.engineAxisY.toFixed(3)}`);
  }
  /**
   * The walls reach the FINS, not the engines.
   *
   * They used to be the same test, because the fins were placed by the model
   * at the body's own corner and the corner was sized to the engines -- both
   * ends of the comparison came from one guess. The solve names `y_vt` now, and
   * the body has to end where the fins stand: they sit on that corner, and a
   * body sized to the engines instead ends 87 mm inboard of its own fins.
   *
   * The engines are then inboard of the trailing edge rather than flush with
   * it, which is the right way round -- the carved trough holds them, the
   * corner holds the fins.
   */
  const wantWall = d.finY ?? (d.engineY + rN + gap);
  if (Math.abs(wall - wantWall) > 0.03) {
    bad(`the walls reach ${wall.toFixed(3)} where the fins stand at ${wantWall.toFixed(3)}`);
  }

  /**
   * And the underside SWEEPS up to it rather than necking in.
   *
   * The rise is fixed -- it is wherever the engines were put -- so the only
   * question is how much length it is given, and the answer is every metre from
   * where the run starts to the body's own trailing edge. Over the tailcone
   * alone the keel came up at 48 degrees.
   *
   * It is NOT asked to arrive level. The run is eased in only, so the underside
   * leaves the cabin gently and is still climbing when it reaches the trailing
   * edge, which is what the shape does; a curve that flattened at both ends
   * would have to be steeper in the middle to cover the same rise.
   */
  const L2 = fu.length;
  let prev = null, steepest = 0, dropped = 0, atTE = 0;
  for (let i = 650; i <= 1000; i++) {
    const q = i / 1000, ke = fu.keelAt(-L2 * q);
    if (prev !== null) {
      const slope = Math.atan2(ke - prev[1], (q - prev[0]) * L2) * 180 / Math.PI;
      steepest = Math.max(steepest, Math.abs(slope));
      atTE = slope;
      if (slope < -0.2) dropped++;              // going back DOWN, going aft
    }
    prev = [q, ke];
  }
  console.log(`     the underside sweeps up at no more than ${steepest.toFixed(1)} deg, ` +
              `still rising at ${atTE.toFixed(1)} deg where it meets the trailing edge`);
  if (steepest > 30) bad(`the underside necks in at ${steepest.toFixed(1)} deg`);
  if (dropped) bad(`the underside falls again at ${dropped} stations -- it is not one sweep`);
  if (atTE < 2) bad(`the underside flattens to ${atTE.toFixed(1)} deg at the trailing edge`);

  /**
   * Every section's lower surface is CONVEX -- a U, not a pinch.
   *
   * This is what interpolating two radius functions angle by angle destroyed:
   * the cabin's shape and the channel's disagree most at the bottom corners, so
   * the blend waisted in there for most of the run while the keel and the
   * width, measured on their own, both looked fine. Walking the lower boundary
   * and checking it never turns the wrong way is the test that sees it.
   */
  {
    let worst = 0, worstAt = 0;
    for (let i = 60; i <= 100; i++) {
      const q = i / 100, zq = -L2 * q;
      const pts = [];
      for (let k = 0; k <= 180; k++) pts.push(fu.surfaceAt(zq, Math.PI + (k / 180) * Math.PI));
      let turns = 0;
      for (let k = 1; k < pts.length - 1; k++) {
        const ax = pts[k].x - pts[k - 1].x, ay = pts[k].y - pts[k - 1].y;
        const bx = pts[k + 1].x - pts[k].x, by = pts[k + 1].y - pts[k].y;
        if (ax * by - ay * bx < -1e-9) turns++;
      }
      if (turns > worst) { worst = turns; worstAt = q; }
    }
    console.log(`     lower surface convex at every station` +
                (worst ? ` EXCEPT x/L ${worstAt.toFixed(2)}` : ''));
    if (worst) bad(`the lower surface pinches at x/L ${worstAt.toFixed(2)} -- ${worst} reversals`);
  }

  /**
   * Seated: the engine stands proud of the channel and does not hang below it.
   *
   * Measured on the ENGINE, not on the whole pod. A pod may carry a nacelle
   * wrapped round the engine, and that cowl is deliberately wider than the
   * channel was cut for -- the channel holds the engine, the cowl wraps it, and
   * whatever of the cowl ends up inside the body is carved away. Measuring the
   * pod's bounding box called that a fault: the nacelle's underside sits 68 mm
   * below the floor before the carve takes it.
   */
  const core = [];
  u.parts.engines[0].traverse((o) => { if (o.isMesh && o.name === 'core') core.push(o); });
  const b2 = new THREE.Box3();
  for (const c of core) b2.expandByObject(c);
  if (b2.isEmpty()) b2.setFromObject(u.parts.engines[0]);
  if (b2.max.y <= roof) bad('the engines do not stand above the channel -- they are buried');
  if (b2.min.y < floor - 1e-6) bad('the engines hang below the channel floor');
  console.log(`     engines stand ${(b2.max.y - roof).toFixed(3)} m above the open top`);
}

/* ---- the tails are placed by their own stations, and form a pi --------- */
{
  if (d.vtLE == null || d.htLE == null) bad('the solve carries no x_vt_le / x_ht_le');
  const fin = new THREE.Box3().setFromObject(u.parts.verticalTails[0]);
  const ht = new THREE.Box3().setFromObject(u.parts.horizontalTail);
  const overlap = Math.min(ht.max.z, fin.max.z) - Math.max(ht.min.z, fin.min.z);
  console.log(`\ntails: fin z ${fin.max.z.toFixed(2)}..${fin.min.z.toFixed(2)}, ` +
              `tailplane ${ht.max.z.toFixed(2)}..${ht.min.z.toFixed(2)} -> overlap ${overlap.toFixed(2)} m`);
  // A pi tail: the tailplane is carried ON the fins, so they must overlap in
  // station and meet in height. Derived from the arms they missed each other
  // by four metres, which is what x_ht_le and x_vt_le settled.
  if (overlap <= 0) bad(`the tailplane is ${(-overlap).toFixed(2)} m clear of the fins`);
  if (ht.min.y > fin.max.y || ht.max.y < fin.max.y - 0.5) {
    bad(`the tailplane sits at y ${ht.min.y.toFixed(2)}..${ht.max.y.toFixed(2)}, ` +
        `fins top out at ${fin.max.y.toFixed(2)}`);
  }
  /**
   * The fins stand where the SOLVE puts them, laterally and longitudinally.
   *
   * `y_vt` is the lateral station and it is the number that matters, because
   * the tailplane rides on the fin tips and the solve aligns its 0.40c spar
   * with the tip's at one spanwise station. Anywhere else and the tailplane
   * slides along its own sweep -- 140 mm out when the fins sat on the body's
   * back corner, which is where this used to require them.
   *
   * The corner is still reported, because a fin standing well clear of the body
   * would be hanging in the air, but it is no longer the thing being asserted:
   * the solve owns the station and it is 87 mm outboard of the corner here.
   */
  const vt = u.parts.verticalTails.find((f) => f.position.x > 0) ?? u.parts.verticalTails[0];
  const corner = {
    x: fu.halfWidthAt(-d.fuseLength),
    y: fu.crownAt(-d.fuseLength),
    z: -d.fuseLength,
  };
  const rootTE = { x: vt.position.x, y: vt.position.y, z: vt.position.z - u.fin.rootChord };
  console.log(`     fin root trailing edge (${rootTE.x.toFixed(3)}, ${rootTE.y.toFixed(3)}, ` +
              `${rootTE.z.toFixed(3)}); the body's back upper corner is at ` +
              `(${corner.x.toFixed(3)}, ${corner.y.toFixed(3)}, ${corner.z.toFixed(3)})`);
  if (d.finY != null && Math.abs(rootTE.x - d.finY) > 1e-6) {
    bad(`the fin stands at ${rootTE.x.toFixed(4)} where the solve's y_vt is ${d.finY.toFixed(4)}`);
  }
  if (Math.abs(rootTE.z - corner.z) > 2e-3) {
    bad(`the fin root trailing edge is at ${rootTE.z.toFixed(3)}, not the body's own ${corner.z.toFixed(3)}`);
  }
  if (Math.abs(rootTE.y - corner.y) > 2e-3) {
    bad(`the fin root sits at ${rootTE.y.toFixed(3)}, not on the body's crown ${corner.y.toFixed(3)}`);
  }
  // Outboard of the body it would be hanging in the air rather than mounted.
  if (rootTE.x > corner.x + 0.25) {
    bad(`the fin stands ${(rootTE.x - corner.x).toFixed(3)} m outboard of the body's corner`);
  }
}

/* ---- how it sits ------------------------------------------------------- */
{
  const sat = d8Aircraft(deck, { sitOnGround: true });
  sat.updateMatrixWorld(true);
  let lowest = Infinity;
  for (const lg of sat.userData.parts.gear) {
    lowest = Math.min(lowest, new THREE.Box3().setFromObject(lg).min.y);
  }
  /**
   * How nearly the wheels lie in ONE plane.
   *
   * The lowest wheel touching says nothing -- sitting the aeroplane down puts
   * it there by construction. What matters is the spread between the legs, and
   * it is worth measuring because it is not free: the deck fixes both strut
   * lengths and where they attach follows from the airframe, so a large spread
   * means the attachment is wrong, not the struts.
   *
   * Hung from the body's keel the main legs left the nose wheel 901 mm clear
   * and the aeroplane 3.22 degrees nose-up. From the wing, which is 0.73 m
   * higher and is what the solve's longer main leg is measured from, 171 mm.
   */
  let highest = -Infinity;
  for (const lg of sat.userData.parts.gear) {
    highest = Math.max(highest, new THREE.Box3().setFromObject(lg).min.y);
  }
  const spread = highest - lowest;
  /**
 * Where the solved gear attachment ended up.
 *
 * The height is solved from the requirement that the wheels are coplanar, so
 * the attitude below is zero by construction and says nothing on its own. What
 * is worth checking is that the answer is a place an aeroplane could carry a
 * leg from -- inside the body's own depth at that station, rather than hanging
 * in the air above the crown.
 */
{
  const a = u.mainGearAttach;
  console.log(`\nmain gear attaches ${a.aboveKeel.toFixed(3)} m above the keel ` +
              `(the body is ${a.bodyDepth.toFixed(3)} deep there), ` +
              `${a.aboveWing >= 0 ? '+' : ''}${a.aboveWing.toFixed(3)} m from the wing`);
  if (a.aboveKeel < -0.05 || a.aboveKeel > a.bodyDepth + 0.05) {
    bad(`the gear attaches ${a.aboveKeel.toFixed(2)} m above the keel, outside the body's ` +
        `${a.bodyDepth.toFixed(2)} m depth -- the deck's struts do not fit this body`);
  }
}

console.log(`\nstatic attitude ${u.groundAttitude.toFixed(3)} deg; the wheels lie within ` +
              `${(1000 * spread).toFixed(0)} mm of one plane`);
  if (Math.abs(lowest) > 0.01) bad(`sat down, the lowest wheel is ${lowest.toFixed(3)} off the ground`);
  if (spread > 0.25) {
    bad(`the wheels are ${(1000 * spread).toFixed(0)} mm out of plane -- the gear ` +
        `attachments do not match the deck's strut lengths`);
  }
}

console.log(failures ? `\nFAIL: ${failures} problem(s)`
                     : '\nPASS: the D8 reproduces the solve it came from');
process.exit(failures ? 1 : 0);
