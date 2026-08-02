/**
 * Headless checks on a D8 double-bubble body.
 *
 * Separate from check_fuselage because most of the tube's claims are simply not
 * true of a D8 and forcing it through them would mean weakening them for the
 * tube. A D8's crown is SUPPOSED to fall aft -- that is the trough the engines
 * sit in -- and its section is not circular anywhere, so "crown = yc + r" is
 * wrong by construction.
 *
 * What is checked instead is the thing that makes it a D8 at all: that the
 * section really is three different shapes along the length, that each one is
 * the shape it claims to be, and that the morph between them leaves no ring.
 *
 *     node check_d8.mjs
 */
import * as THREE from 'three';
import { d8Fuselage } from './components/fuselage.js';

const CASES = [
  { name: 'D8 default', radius: 1.90 },
  { name: 'edge low',   radius: 1.90, shape: { tailEdgeHeight: 0.15 } },
  { name: 'edge high',  radius: 1.90, shape: { tailEdgeHeight: 0.80 } },
  { name: 'wider',      radius: 1.90, shape: { cabinWidth: 1.85 } },
  { name: 'narrow',     radius: 1.90, shape: { cabinWidth: 1.15 } },
  { name: 'flat roof',  radius: 1.90, shape: { cabinCrown: 0.00 } },
  { name: 'round roof', radius: 1.90, shape: { cabinCrown: 0.45 } },
  { name: 'small',      radius: 1.10, fineness: 8.0 },
];

let failures = 0;
const bad = (msg) => { console.log(`  FAIL  ${msg}`); failures++; };

for (const c of CASES) {
  const body = d8Fuselage({ radius: c.radius, fineness: c.fineness, shape: c.shape ?? {} });
  const u = body.userData, R = u.radius, L = u.length;
  const p = u.shapeParams;
  console.log(`${c.name}: L ${L.toFixed(2)}, half-height ${R.toFixed(2)}, ` +
              `cabin W/H ${p.cabinWidth}, crown ${p.cabinCrown}`);

  /* 1. the skin is a closed, outward solid --------------------------------- */
  // Same two tests as the tube, and they matter more here: a section that
  // changes along the body is far easier to loft into something inside out or
  // torn than a constant one.
  const geo = body.children[0].geometry;
  const pos = geo.getAttribute('position'), nrm = geo.getAttribute('normal');
  const idx = geo.getIndex().array;

  const wi = new Int32Array(pos.count), w = new Map();
  for (let i = 0; i < pos.count; i++) {
    const k = [pos.getX(i), pos.getY(i), pos.getZ(i)]
      .map((v) => Math.round(v * 1e6) + 0).join(',');
    if (!w.has(k)) w.set(k, w.size);
    wi[i] = w.get(k);
  }
  const edge = new Map();
  for (let t = 0; t + 2 < idx.length; t += 3) {
    const q = [wi[idx[t]], wi[idx[t + 1]], wi[idx[t + 2]]];
    if (q[0] === q[1] || q[1] === q[2] || q[0] === q[2]) continue;
    for (let k = 0; k < 3; k++) {
      const a = q[k], b = q[(k + 1) % 3], key = a < b ? `${a},${b}` : `${b},${a}`;
      edge.set(key, (edge.get(key) || 0) + 1);
    }
  }
  const open = [...edge.values()].filter((n) => n !== 2).length;
  if (open) bad(`skin has ${open} boundary edges`);

  let vol = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  const cc = new THREE.Vector3(), n = new THREE.Vector3();
  for (let t = 0; t + 2 < idx.length; t += 3) {
    a.fromBufferAttribute(pos, idx[t]);
    b.fromBufferAttribute(pos, idx[t + 1]);
    cc.fromBufferAttribute(pos, idx[t + 2]);
    vol += a.dot(n.crossVectors(b, cc)) / 6;
  }
  if (vol <= 0) bad(`enclosed volume ${vol.toFixed(2)} -- wound inward`);

  /* 2. r(z) still means HALF-HEIGHT ---------------------------------------- */
  // The whole library rests on this: one size distribution and one centreline
  // law serve every section because every section is normalised to unit height.
  // The double bubble gets it for free -- its highest point is a lobe apex, at
  // height exactly 1 -- but only if it is not divided by the height at the
  // CENTRELINE, which is the valley and would make the body 12% too tall.
  const zc = -L * 0.45;
  const halfHeight = (u.crownAt(zc) - u.keelAt(zc)) / 2;
  if (Math.abs(halfHeight / R - 1) > 1e-3)
    bad(`cabin half-height ${halfHeight.toFixed(3)} but radius is ${R.toFixed(3)}`);

  /* 3. the cabin roof is FLAT and slightly convex -------------------------- */
  // The OML is not the double bubble. That is the thing most descriptions of
  // this aeroplane get backwards, and it is what this file did first: modelling
  // the bubbles as the outer shape puts a crease down the top of an aeroplane
  // with a flat roof. The bubbles are the pressure vessel; the roof is a
  // fairing over them.
  const wantWidth = p.cabinWidth;
  const gotWidth = 2 * u.halfWidthAt(zc) / (u.crownAt(zc) - u.keelAt(zc));
  if (Math.abs(gotWidth - wantWidth) > 2e-3)
    bad(`cabin width/height ${gotWidth.toFixed(3)}, asked ${wantWidth}`);

  // Height across the roof, by x rather than by angle -- "flat" is a statement
  // about the shape in space, and a statement about equal angles is not the
  // same statement on a section this wide.
  //
  // Probed as a fraction of the FLAT's own extent, w - 1, not of the half-width.
  // The flat top of a stadium only spans |x| <= w - 1, which at w = 1.55 is a
  // third of the half-width -- so probing at half the half-width lands out on
  // the arc and reports a dead flat roof as dropping 2.6%.
  const flatHalf = Math.max(1e-6, (p.cabinWidth - 1) * R);
  const roofAt = (z, frac) => {
    const sh = u.shapeAt(z), want = frac * flatHalf;
    let lo = 0, hi = Math.PI / 2;                 // x falls as theta rises
    for (let i = 0; i < 60; i++) {
      const m = (lo + hi) / 2;
      if (sh.r * u.section(m, z) * Math.cos(m) > want) lo = m; else hi = m;
    }
    const th = (lo + hi) / 2;
    return sh.yc + sh.r * u.section(th, z) * Math.sin(th);
  };
  const peak = u.crownAt(zc);
  const midDrop = (peak - roofAt(zc, 0.7)) / (peak - u.shapeAt(zc).yc);
  if (midDrop > 0.02 + 0.12 * p.cabinCrown)
    bad(`roof drops ${(100 * midDrop).toFixed(2)}% across the flat -- not flat`);
  if (p.cabinCrown > 0 && midDrop < 1e-5)
    bad(`crown is ${p.cabinCrown} but the roof is dead flat`);
  if (p.cabinCrown === 0 && midDrop > 1e-9)
    bad(`crown is 0 but the roof drops ${(100 * midDrop).toFixed(4)}%`);

  // THE SHOULDERS ARE SEMICIRCLES. This is the claim that failed before -- a
  // superellipse gets the flat roof right and leaves the shoulders visibly
  // boxy. A stadium's side is an arc of the pressure lobe itself, so at half
  // the height the surface must stand at o + sqrt(3)/2 with o = w - 1. The
  // crown blend pulls it in slightly, which is the only licence allowed.
  const shoulderAt = (frac) => {
    const sh = u.shapeAt(zc);
    let lo = 0, hi = Math.PI / 2;
    for (let i = 0; i < 60; i++) {
      const m = (lo + hi) / 2;
      if (sh.r * u.section(m, zc) * Math.sin(m) / R < frac) lo = m; else hi = m;
    }
    const th = (lo + hi) / 2;
    return sh.r * u.section(th, zc) * Math.cos(th) / R;
  };
  const gotShoulder = shoulderAt(0.5);
  const wantShoulder = (p.cabinWidth - 1) + Math.sqrt(0.75);
  const slack = 0.02 + 0.25 * p.cabinCrown;
  if (Math.abs(gotShoulder - wantShoulder) > slack)
    bad(`shoulder at half height is ${gotShoulder.toFixed(3)}R, ` +
        `a semicircular side gives ${wantShoulder.toFixed(3)}R`);

  // Convex, and highest on the centreline: no valley, which is what the bubble
  // section produced and what a fairing exists to remove.
  let dip = 0;
  for (let f = 0; f <= 0.9; f += 0.05) dip = Math.max(dip, roofAt(zc, f) - peak);
  if (dip > 1e-6) bad(`roof rises ${dip.toFixed(4)} off the centreline -- a valley`);

  /* 4. it is three shapes, not one ---------------------------------------- */
  // Elliptical at the point, flat-topped over the cabin, dished aft. If any two
  // of those measure the same the morph is not doing anything.
  const valley = (z) => u.shapeAt(z).yc + u.shapeAt(z).r * u.section(Math.PI / 2, z);

  // Asked of the SECTION rather than of the body. At 6% of the length the morph
  // toward the cabin has already begun, so measuring the body there and calling
  // it "the nose ellipse" reports the blend and blames the ellipse.
  const noseW = u.section(0, -1e-9) / u.section(Math.PI / 2, -1e-9);
  if (Math.abs(noseW - p.noseWidth) > 2e-3)
    bad(`nose section is ${noseW.toFixed(3)} wide per unit tall, asked ${p.noseWidth}`);
  // Compared shape against shape, not aspect against aspect: the nose and the
  // cabin can be the same width and still be quite different sections, since
  // one is an ellipse and the other is flat-topped. Aspect alone called that
  // inert when it plainly is not.
  let secDiff = 0;
  for (let i = 0; i <= 180; i++) {
    const th = (i / 180) * Math.PI * 2;
    secDiff = Math.max(secDiff, Math.abs(u.section(th, -1e-9) - u.section(th, zc)));
  }
  if (secDiff < 0.02)
    bad(`nose and cabin sections differ by only ${secDiff.toFixed(4)} -- the morph is inert`);

  // Aft, the roof is dished for the engines: the centreline drops below the
  // shoulders. Over the cabin it does not.

  /* 4a. the tail closes on a HORIZONTAL LINE ------------------------------- */
  // Not on a point, not on a face, and not on a channel. The height goes to
  // nothing while the width does not, which is a line -- and it is only
  // expressible because the section is allowed to ask the size distribution how
  // tall the body is, rather than being a fixed multiple of it.
  const tailEdge = u.tailEdge;
  const edgeRatio = tailEdge.halfWidth / tailEdge.halfHeight;
  if (edgeRatio < 20)
    bad(`trailing edge is ${edgeRatio.toFixed(1)} wide per unit thick -- a face, not a line`);
  // And it sits where it was asked to, measured up from the KEEL. The residual
  // thickness has to be accounted for in the hold or the line lands tipR short,
  // which is small enough to go unnoticed and wrong every time.
  if (Math.abs(tailEdge.heightFraction - p.tailEdgeHeight) > 1e-4)
    bad(`trailing edge sits at ${tailEdge.heightFraction.toFixed(4)} of the height, ` +
        `asked ${p.tailEdgeHeight}`);
  // No taper in plan: the top view is a constant-width slab from the cabin to
  // the trailing edge, so the half-width may not move over the whole aft body.
  let planDrift = 0;
  for (let i = 0; i <= 120; i++) {
    const z = u.cabinZ[1] - (L + u.cabinZ[1]) * (i / 120);
    planDrift = Math.max(planDrift, Math.abs(u.halfWidthAt(z) - u.halfWidthAt(zc)));
  }
  if (u.planTaper === 1 && planDrift > 1e-6)
    bad(`half-width moves ${planDrift.toFixed(4)} over the aft body -- it tapers in plan`);
  // And nothing on the way there may dish: no channel down the aft deck.
  let dished = 0;
  for (let i = 0; i <= 200; i++) {
    const z = u.cabinZ[1] - (L + u.cabinZ[1]) * (i / 200);
    if (u.crownAt(z) - valley(z) > 1e-6) dished++;
  }
  if (dished) bad(`aft deck is dished at ${dished} stations -- a channel`);

  /* 4b. the point sits above the axis -------------------------------------- */
  // The one thing about this nose that is not simply "wider than a tube's". A
  // tube drops its point for the view over the nose; a D8 lifts it for moment.
  // Sign errors here are invisible on a body this wide, so measure it.
  const tipY = u.shapeAt(0).yc;
  if (Math.abs(tipY - p.tipRise * R) > 1e-9)
    bad(`point at ${tipY.toFixed(3)}, tipRise ${p.tipRise} asks ${(p.tipRise * R).toFixed(3)}`);
  if (p.tipRise > 0 && tipY <= 0)
    bad(`tipRise is positive but the point is at or below the axis`);
  let outside = 0;
  for (let i = 0; i <= 300; i++) {
    const z = -u.noseLength * i / 300;
    if (u.crownAt(z) > R + 1e-6 || u.keelAt(z) < -R - 1e-6) outside++;
  }
  if (outside) bad(`nose leaves the +/-R envelope at ${outside} stations`);

  /* 4c. no sliver triangles at the closing line --------------------------- */
  // A ring thin enough to read as a line is a ring thin enough for its cap fan
  // to collapse. The nose apex is excluded: there the ring really does close on
  // a point and the fan is degenerate by construction, as it is on a tube.
  //
  // Measured by SHAPE, not by size. A sliver is a triangle that is THIN, and
  // thinness is its area against its own longest edge squared -- about 0.43 for
  // an equilateral one, near zero for a splinter. Comparing area to the body's
  // bounding box instead condemns any small triangle, and on a body that closes
  // to a line every triangle at the trailing edge is small: that version failed
  // 10 faces whose areas were 1.1e-5 against a 1.4e-5 threshold derived from a
  // 36 m diagonal, which is not a statement about anything.
  let slivers = 0, thinnest = 1;
  {
    const pos2 = geo.getAttribute('position'), ix = geo.getIndex().array;
    const va = new THREE.Vector3(), vb = new THREE.Vector3(), vcc = new THREE.Vector3();
    const vn = new THREE.Vector3();
    for (let t = 0; t + 2 < ix.length; t += 3) {
      va.fromBufferAttribute(pos2, ix[t]);
      vb.fromBufferAttribute(pos2, ix[t + 1]);
      vcc.fromBufferAttribute(pos2, ix[t + 2]);
      if ((va.z + vb.z + vcc.z) / 3 > -0.02 * L) continue;      // the nose apex
      const area = vn.crossVectors(vb.clone().sub(va), vcc.clone().sub(va)).length() / 2;
      const longest = Math.max(va.distanceTo(vb), vb.distanceTo(vcc), vcc.distanceTo(va));
      if (longest < 1e-12) { slivers++; continue; }
      const q = area / (longest * longest);
      thinnest = Math.min(thinnest, q);
      if (q < 2e-3) slivers++;
    }
  }
  if (slivers) bad(`${slivers} sliver triangles aft of the nose apex`);

  /* 5. the morph leaves no ring ------------------------------------------- */
  // A jump in the SECTION is a crease running all the way round the body, which
  // is the one defect this construction can produce that a tube cannot. Walk
  // the half-width and the valley depth and look for a step.
  // Measured on the SHAPE, not on the surface. The surface moves arbitrarily
  // fast at the point -- the nose taper has a vertical tangent there by
  // construction -- so an absolute rate reports the nose tip as a crease at
  // every setting, which it is not. Scale-free ratios have no such excuse.
  const shapeOf = (z) => {
    let wide = 0, tall = 0;
    for (let i = 0; i <= 180; i++) {
      const th = (i / 180) * Math.PI * 2, sc = u.section(th, z);
      wide = Math.max(wide, Math.abs(sc * Math.cos(th)));
      tall = Math.max(tall, sc * Math.sin(th));
    }
    return [wide / tall, u.section(Math.PI / 2, z) / tall];   // aspect, valley
  };
  // Looked for as a SPIKE against its own neighbours, not against a fixed rate.
  // The aft aspect ratio legitimately runs away -- that is what closing on a
  // line means -- so any absolute bound is either useless or condemns the tail.
  // A ring is a step: one interval changing far more than the ones around it.
  const N = 600, steps = [];
  let prev = shapeOf(0);
  for (let i = 1; i <= N; i++) {
    const z = -L * i / N, now = shapeOf(z);
    steps.push(Math.abs(now[0] - prev[0]) + Math.abs(now[1] - prev[1]));
    prev = now;
  }
  // Against its immediate NEIGHBOURS, not against the median of the whole body.
  // Most of a fuselage is parallel, so the median step is essentially zero and
  // any ratio to it is meaningless -- it reported 2e16 at the tail, where the
  // section is changing fast but perfectly smoothly. A ring is a step that is
  // large while the steps either side of it are small; that is a local test.
  let spike = 0, worstZ = 0;
  for (let i = 4; i < steps.length - 4; i++) {
    let local = 0;
    for (let k = -4; k <= 4; k++) if (k) local += steps[i + k];
    local /= 8;
    const ratio = steps[i] / Math.max(local, 1e-12);
    if (ratio > spike) { spike = ratio; worstZ = -L * (i + 1) / N; }
  }
  if (spike > 8)
    bad(`one station changes ${spike.toFixed(1)}x its neighbours at ` +
        `z ${worstZ.toFixed(1)} -- a ring`);


  console.log(`  closed, volume ${vol.toFixed(1)}, half-height ${halfHeight.toFixed(3)}`);
  console.log(`  width/height: nose section ${noseW.toFixed(2)}, cabin ${gotWidth.toFixed(2)}`);
  console.log(`  nose-to-cabin section change ${secDiff.toFixed(3)}`);
  console.log(`  roof drops ${(100 * midDrop).toFixed(3)}% across the flat; ` +
              `shoulder ${gotShoulder.toFixed(3)}R vs semicircle ${wantShoulder.toFixed(3)}R`);
  console.log(`  trailing edge ${(2 * tailEdge.halfWidth).toFixed(2)} wide x ` +
              `${(2 * tailEdge.halfHeight).toFixed(3)} thick (W/T ${edgeRatio.toFixed(0)}), ` +
              `at ${(100 * tailEdge.heightFraction).toFixed(1)}% of height from the keel`);
  console.log(`  plan taper ${u.planTaper.toFixed(3)}, ` +
              `half-width drifts ${planDrift.toFixed(5)} aft, ${dished} dished stations`);
  console.log(`  point at y ${tipY.toFixed(3)} (${p.tipRise} half-heights up), ` +
              `${outside} stations outside the envelope`);
  console.log(`  worst section spike ${spike.toFixed(2)}x its neighbours, ` +
              `thinnest face ${thinnest.toFixed(4)} (equilateral is 0.43)`);
}

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: three sections, one closed body, no rings');
process.exit(failures ? 1 : 0);
