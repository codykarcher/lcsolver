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
  // A centimetre, not zero: halfWidthAt scans the section at a finite number of
  // angles, so it under-reads a maximum falling between two of them.
  if (edge < reach - 1e-2) bad(`the trailing edge is ${edge.toFixed(3)} but the engines reach ${reach.toFixed(3)}`);
  if (edge > 1.35 * reach) bad(`the trailing edge runs ${(edge / reach).toFixed(2)} times past the engines`);

  /**
   * The afterbody is LIDDED, and the engines sit on it.
   *
   * Not a channel they are cradled in -- that was the earlier arrangement, and
   * a duct could not be carved from it because the body was shrink-wrapped to
   * them: 84 per cent of the section at the tail was already engine. The lid
   * descends from the cabin's crown to just under them instead, so the body
   * stops where they begin and nothing has to be cut out of it.
   *
   * The measure that matters is therefore how much of each engine is still
   * INSIDE the body. It should be none.
   */
  const rN = d.nacelleDia / 2;
  console.log(`     lid falls from ${d.fuseHalfHeight.toFixed(3)} at the cabin to ` +
              `${fu.crownAt(-d.engineX).toFixed(3)} at the engines, whose undersides ` +
              `are at ${(u.engineAxisY - rN).toFixed(3)}`);
  let worstBuried = 0;
  for (const x of [d.engineX - 0.3, d.engineX, d.engineX + 0.5, d.fuseLength]) {
    const zq = -x, sh = fu.shapeAt(zq);
    let inside = 0, n = 0;
    for (let i = 0; i < 720; i++) {
      const a = (i / 720) * 2 * Math.PI;
      const px = d.engineY + rN * Math.cos(a), py = u.engineAxisY + rN * Math.sin(a);
      const th = Math.atan2(py - sh.yc, px), r = Math.hypot(px, py - sh.yc);
      const sp = fu.surfaceAt(zq, th);
      if (r < Math.hypot(sp.x, sp.y - sh.yc) - 1e-6) inside++;
      n++;
    }
    worstBuried = Math.max(worstBuried, (100 * inside) / n);
  }
  console.log(`     at worst ${worstBuried.toFixed(0)}% of an engine's outline is inside the body`);
  if (worstBuried > 2) bad(`${worstBuried.toFixed(0)}% of an engine is still buried in the body`);

  // And every station has to remain a valid section: the centre under the roof.
  {
    let bad2 = 0;
    for (let i = 600; i <= 1000; i++) {
      const zq = -fu.length * (i / 1000), sh = fu.shapeAt(zq);
      if (!(fu.halfWidthAt(zq) > 0.2 && fu.crownAt(zq) > fu.keelAt(zq) + 1e-3
            && sh.yc < fu.crownAt(zq))) bad2++;
    }
    console.log(`     ${bad2 ? bad2 + ' stations invalid' : 'every station is a valid section'}`);
    if (bad2) bad(`${bad2} stations aft have the centre at or above the roof -- the section is empty there`);
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
  if (steepest > 40) bad(`the underside necks in at ${steepest.toFixed(1)} deg`);
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

  // Sitting ON the lid: the whole engine is above it, which "0% buried" above
  // already establishes. What is left to say is by how much, since an engine
  // resting exactly on the lid and one floating over it read the same in a
  // percentage.
  const b2 = new THREE.Box3().setFromObject(u.parts.engines[0]);
  const lidAtEngines = fu.crownAt(-d.engineX);
  console.log(`     the engine's underside sits ${(b2.min.y - lidAtEngines).toFixed(3)} m ` +
              `above the lid`);
  if (b2.min.y < lidAtEngines - 1e-6) bad('the engine dips below the lid');
  if (b2.min.y > lidAtEngines + 0.2) {
    bad(`the engine floats ${(b2.min.y - lidAtEngines).toFixed(2)} m above the lid`);
  }
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
   * The fins hang by their root TRAILING EDGE, on the body's back upper
   * corners -- all three coordinates, not just the station.
   *
   * The station is the solve's doing: it puts the fin's trailing edge on the
   * body's own. The other two are the channel's: it puts the body's upper
   * corners there. So the corner is one point that both already own, and the
   * fin either sits on it or does not.
   */
  const vt = u.parts.verticalTails.find((f) => f.position.x > 0) ?? u.parts.verticalTails[0];
  const corner = {
    x: fu.halfWidthAt(-d.fuseLength),
    y: fu.crownAt(-d.fuseLength),
    z: -d.fuseLength,
  };
  const rootTE = { x: vt.position.x, y: vt.position.y, z: vt.position.z - u.fin.rootChord };
  const off = Math.hypot(rootTE.x - corner.x, rootTE.y - corner.y, rootTE.z - corner.z);
  console.log(`     fin root trailing edge (${rootTE.x.toFixed(3)}, ${rootTE.y.toFixed(3)}, ` +
              `${rootTE.z.toFixed(3)}) against the body's back upper corner ` +
              `(${corner.x.toFixed(3)}, ${corner.y.toFixed(3)}, ${corner.z.toFixed(3)})`);
  if (off > 2e-3) bad(`the fin root trailing edge is ${(1000 * off).toFixed(0)} mm off the corner`);
}

/* ---- how it sits ------------------------------------------------------- */
{
  const sat = d8Aircraft(deck, { sitOnGround: true });
  sat.updateMatrixWorld(true);
  let lowest = Infinity;
  for (const lg of sat.userData.parts.gear) {
    lowest = Math.min(lowest, new THREE.Box3().setFromObject(lg).min.y);
  }
  console.log(`\nstatic attitude ${u.groundAttitude.toFixed(3)} deg, wheels within ` +
              `${(1000 * Math.abs(lowest)).toFixed(1)} mm of the ground`);
  if (Math.abs(lowest) > 0.01) bad(`sat down, the lowest wheel is ${lowest.toFixed(3)} off the ground`);
}

console.log(failures ? `\nFAIL: ${failures} problem(s)`
                     : '\nPASS: the D8 reproduces the solve it came from');
process.exit(failures ? 1 : 0);
