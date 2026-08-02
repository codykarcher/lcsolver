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
  // NOT "above the crown". That was the test when the engines sat on top of
  // the body; they are let into a channel now, so being below its crown is the
  // point. Whether they are seated is asked properly further down, by measuring
  // how much of the engine's own circle is skin.
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
  // The channels themselves set the trailing edge's width now, so the two are
  // equal by construction rather than merely close -- the flat region the fins'
  // trailing edges integrate into IS the outside of the channels.
  // A millimetre, not zero: halfWidthAt scans the section at a finite number of
  // angles, so it under-reads a maximum falling between two of them -- by 9
  // microns here, which is a property of the measurement and not of the body.
  if (edge < reach - 1e-3) bad(`the trailing edge is ${edge.toFixed(3)} but the engines reach ${reach.toFixed(3)}`);
  if (edge > 1.35 * reach) bad(`the trailing edge runs ${(edge / reach).toFixed(2)} times past the engines`);

  /**
   * The afterbody has to still BE there where the engines sit.
   *
   * The bare body's closing law falls away from the moment the cabin ends,
   * which is right when there is nothing to carry: it left 12 per cent of the
   * depth at the engine station, so they perched on an edge and the dish they
   * belong in had nowhere to exist.
   */
  const z = -d.engineX;
  const deep = (fu.crownAt(z) - fu.keelAt(z)) / 2;
  console.log(`     ${(100 * deep / d.fuseHalfHeight).toFixed(0)}% of full depth at the ` +
              `engine station (${deep.toFixed(3)} m against an engine radius of ` +
              `${(d.nacelleDia / 2).toFixed(3)})`);
  if (deep < 0.5 * d.fuseHalfHeight) {
    bad(`only ${(100 * deep / d.fuseHalfHeight).toFixed(0)}% of the depth survives to the engines`);
  }

  /**
   * The engine sits at 55 per cent of the body's height, in a channel the
   * afterbody has already made for it.
   *
   * Measured by asking how much of the engine's own circle IS skin: a channel
   * that wraps its outboard and lower faces is a receptacle, and a body that
   * merely passes near it is not. Nothing at all means the engine is either
   * buried in a core that never narrowed or floating clear of one that did.
   */
  const want = -d.fuseHalfHeight + d.engineHeight * 2 * d.fuseHalfHeight;
  console.log(`     engine axis y ${u.engineAxisY.toFixed(4)} = ` +
              `${(100 * d.engineHeight).toFixed(0)}% of the body's height up from the keel`);
  if (Math.abs(u.engineAxisY - want) > 1e-9) {
    bad(`engine axis at ${u.engineAxisY}, wanted ${want}`);
  }
  const rN = d.nacelleDia / 2;
  let on = 0, tot = 0;
  for (let i = 0; i < 720; i++) {
    const a = (i / 720) * 2 * Math.PI;
    const px = d.engineY + rN * Math.cos(a), py = u.engineAxisY + rN * Math.sin(a);
    if (px < 0) continue;
    tot++;
    const th = Math.atan2(py - fu.shapeAt(z).yc, px);
    const sp = fu.surfaceAt(z, th);
    if (Math.hypot(sp.x - px, sp.y - py) < 0.06) on++;
  }
  const pct = (100 * on) / tot;
  console.log(`     ${pct.toFixed(0)}% of the engine's circle is skin -- the channel wall`);
  if (pct < 25) bad(`only ${pct.toFixed(0)}% of the engine is in a channel`);
  if (pct > 90) bad(`${pct.toFixed(0)}% of the engine is skin -- it is buried, not seated`);

  // And the plan must narrow the whole way. The core passes inside where the
  // channels end up, so channels that form late leave a waist.
  let prev = null, pinch = 0;
  for (let i = 800; i <= 1000; i++) {
    const hw = fu.halfWidthAt(-fu.length * (i / 1000));
    if (prev !== null && hw > prev + 1e-4) pinch = Math.max(pinch, hw - prev);
    prev = hw;
  }
  console.log(`     plan narrows monotonically to ${(1000 * pinch).toFixed(1)} mm`);
  if (pinch > 0.01) bad(`the afterbody pinches in and swells back out by ${(1000 * pinch).toFixed(0)} mm`);
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
  // The fin's trailing edge lands on the body's own tail, which is what the
  // solve's station gives and is worth noticing if it ever stops being true.
  console.log(`     fin trailing edge ${(d.vtLE + d.vtRootChord).toFixed(3)}, ` +
              `body ends ${d.fuseLength.toFixed(3)}`);
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
