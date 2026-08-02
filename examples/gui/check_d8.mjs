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
  { name: 'wider',      radius: 1.90, shape: { cabinWidth: 1.85, trough: 0.40 } },
  { name: 'rounder',    radius: 1.90, shape: { cabinWidth: 1.15, cabinFlat: 2.3 } },
  { name: 'boxy',       radius: 1.90, shape: { cabinFlat: 5.5 } },
  { name: 'small',      radius: 1.10, fineness: 8.0 },
];

let failures = 0;
const bad = (msg) => { console.log(`  FAIL  ${msg}`); failures++; };

for (const c of CASES) {
  const body = d8Fuselage({ radius: c.radius, fineness: c.fineness, shape: c.shape ?? {} });
  const u = body.userData, R = u.radius, L = u.length;
  const p = u.shapeParams;
  console.log(`${c.name}: L ${L.toFixed(2)}, half-height ${R.toFixed(2)}, ` +
              `cabin W/H ${p.cabinWidth}, flat ${p.cabinFlat}, trough ${p.trough}`);

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
  const roofAt = (z, frac) => {
    const sh = u.shapeAt(z), want = frac * u.halfWidthAt(z);
    let lo = 0, hi = Math.PI / 2;                 // x falls as theta rises
    for (let i = 0; i < 60; i++) {
      const m = (lo + hi) / 2;
      if (sh.r * u.section(m, z) * Math.cos(m) > want) lo = m; else hi = m;
    }
    const th = (lo + hi) / 2;
    return sh.yc + sh.r * u.section(th, z) * Math.sin(th);
  };
  const peak = u.crownAt(zc);
  const midDrop = (peak - roofAt(zc, 0.5)) / (peak - u.shapeAt(zc).yc);
  // Against the superellipse's own closed form rather than a fixed threshold.
  // At half the width, |x/w|^n = 2^-n, so the roof is at (1 - 2^-n)^(1/n) --
  // 2.4% down at n = 3.6, 9.4% at n = 2.3. A fixed bound would just be a
  // statement about which exponent I happened to pick as the default.
  const wantDrop = 1 - Math.pow(1 - Math.pow(0.5, p.cabinFlat), 1 / p.cabinFlat);
  if (Math.abs(midDrop - wantDrop) > 3e-3)
    bad(`roof drops ${(100 * midDrop).toFixed(2)}% at half-width, ` +
        `exponent ${p.cabinFlat} gives ${(100 * wantDrop).toFixed(2)}%`);
  if (midDrop < 1e-4) bad(`roof is dead flat, not slightly convex`);

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
  const za = -L * p.uTrough;
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
  const aftDish = (u.crownAt(za) - valley(za)) / (u.crownAt(za) - u.keelAt(za));
  const cabDish = (u.crownAt(zc) - valley(zc)) / (u.crownAt(zc) - u.keelAt(zc));
  if (p.trough > 0.05 && aftDish < 0.04)
    bad(`aft roof is dished only ${(100 * aftDish).toFixed(1)}% -- no seat for an engine`);
  if (cabDish > 0.01)
    bad(`cabin roof is already dished by ${(100 * cabDish).toFixed(1)}% -- the trough has leaked forward`);

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
  let worstStep = 0, worstZ = 0;
  const N = 600;
  let prev = shapeOf(0);
  for (let i = 1; i <= N; i++) {
    const z = -L * i / N, now = shapeOf(z);
    const rate = (Math.abs(now[0] - prev[0]) + Math.abs(now[1] - prev[1])) / (1 / N);
    if (rate > worstStep) { worstStep = rate; worstZ = z; }
    prev = now;
  }
  if (worstStep > 6.0)
    bad(`section shape changes at ${worstStep.toFixed(2)} per unit length at ` +
        `z ${worstZ.toFixed(1)} -- a ring`);

  /* 5b. the pressure vessel fits inside the skin --------------------------- */
  // The flat roof is a fairing over two round tubes. If a tube pokes through it
  // the aeroplane is not describable, and on a body this wide it happens at the
  // shoulders where nobody is looking.
  const withVessel = d8Fuselage({
    radius: c.radius, fineness: c.fineness, shape: c.shape ?? {}, vessel: true });
  const vs = withVessel.children.find((ch) => ch.userData.isPressureVessel);
  let clearance = Infinity;
  if (!vs) bad('vessel: true produced no pressure vessel');
  else {
    const vp = vs.geometry.getAttribute('position');
    for (let i = 0; i < vp.count; i++) {
      const x = vp.getX(i), y = vp.getY(i), z = vp.getZ(i);
      const sh = u.shapeAt(z);
      if (sh.r < 1e-9) continue;
      const th = Math.atan2(y - sh.yc, x);
      clearance = Math.min(clearance,
        sh.r * u.section(th, z) - Math.hypot(x, y - sh.yc));
    }
    if (clearance < 0)
      bad(`pressure vessel pokes ${(-clearance).toFixed(3)} through the skin`);
  }

  /* 6. the nacelle seat is on the surface --------------------------------- */
  // It is published so an engine can be placed ON the body. If it is not
  // actually on the body it is worse than useless.
  for (const side of [1, -1]) {
    const seat = u.nacelleSeat(side);
    const sh = u.shapeAt(seat.z);
    const th = Math.atan2(seat.point.y - sh.yc, seat.point.x);
    const surf = sh.r * u.section(th, seat.z);
    const got = Math.hypot(seat.point.x, seat.point.y - sh.yc);
    if (Math.abs(got - surf) > 1e-6 * R)
      bad(`nacelle seat ${side > 0 ? 'right' : 'left'} is ${(got - surf).toFixed(4)} off the skin`);
    if (seat.point.y < valley(seat.z) - 1e-6)
      bad(`nacelle seat is below the valley floor`);
  }

  console.log(`  closed, volume ${vol.toFixed(1)}, half-height ${halfHeight.toFixed(3)}`);
  console.log(`  width/height: nose section ${noseW.toFixed(2)}, cabin ${gotWidth.toFixed(2)}`);
  console.log(`  nose-to-cabin section change ${secDiff.toFixed(3)}`);
  console.log(`  roof drops ${(100 * midDrop).toFixed(2)}% by half-width ` +
              `(exponent ${p.cabinFlat} gives ${(100 * wantDrop).toFixed(2)}%); ` +
              `dished ${(100 * cabDish).toFixed(1)}% at the cabin, ` +
              `${(100 * aftDish).toFixed(1)}% aft`);
  console.log(`  vessel clearance inside the skin: ${clearance.toFixed(3)}`);
  console.log(`  point at y ${tipY.toFixed(3)} (${p.tipRise} half-heights up), ` +
              `${outside} stations outside the envelope`);
  console.log(`  fastest SHAPE change ${worstStep.toFixed(2)} per unit length ` +
              `at z ${worstZ.toFixed(1)}`);
}

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: three sections, one closed body, no rings');
process.exit(failures ? 1 : 0);
