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
  { name: 'wide',       radius: 1.90, shape: { bubble: 0.62, trough: 0.40 } },
  { name: 'round',      radius: 1.90, shape: { bubble: 0.10, trough: 0.10 } },
  { name: 'small',      radius: 1.10, fineness: 8.0 },
];

let failures = 0;
const bad = (msg) => { console.log(`  FAIL  ${msg}`); failures++; };

for (const c of CASES) {
  const body = d8Fuselage({ radius: c.radius, fineness: c.fineness, shape: c.shape ?? {} });
  const u = body.userData, R = u.radius, L = u.length;
  const p = u.shapeParams;
  console.log(`${c.name}: L ${L.toFixed(2)}, half-height ${R.toFixed(2)}, ` +
              `bubble ${p.bubble}, trough ${p.trough}`);

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

  /* 3. the cabin section is the double bubble it claims to be -------------- */
  // Half-width of the union of two unit lobes offset by o is exactly 1 + o.
  const wantWidth = 1 + p.bubble;
  const gotWidth = u.halfWidthAt(zc) / R;
  if (Math.abs(gotWidth - wantWidth) > 2e-3)
    bad(`cabin half-width ${gotWidth.toFixed(3)}R, two lobes at ${p.bubble} give ${wantWidth}`);

  // And it has a valley: the top is not flat, or it is not a double bubble.
  const valley = (z) => u.shapeAt(z).yc + u.shapeAt(z).r * u.section(Math.PI / 2, z);
  const cabinValley = (u.crownAt(zc) - valley(zc)) / R;
  const wantValley = 1 - Math.sqrt(1 - p.bubble * p.bubble);
  if (Math.abs(cabinValley - wantValley) > 2e-3)
    bad(`cabin valley ${cabinValley.toFixed(3)}R deep, two lobes give ${wantValley.toFixed(3)}`);

  /* 4. it is three shapes, not one ---------------------------------------- */
  // Elliptical at the point, double bubble over the cabin, open aft. If any two
  // of those measure the same the morph is not doing anything.
  // Asked of the SECTION rather than of the body. At 6% of the length the morph
  // toward the bubble has already begun, so measuring the body there and calling
  // it "the nose ellipse" reports the blend and blames the ellipse.
  const aspect = (z) => u.section(0, z) / u.section(Math.PI / 2, z);
  const za = -L * p.uTrough;
  const noseW = aspect(-1e-9);
  if (Math.abs(noseW - p.noseWidth) > 2e-3)
    bad(`nose section is ${noseW.toFixed(3)} wide per unit tall, asked ${p.noseWidth}`);
  if (Math.abs(noseW - gotWidth) < 0.05)
    bad(`nose and cabin sections are the same shape -- the morph is inert`);

  const aftValley = (u.crownAt(za) - valley(za)) / (u.crownAt(za) - u.keelAt(za));
  const cabValley = (u.crownAt(zc) - valley(zc)) / (u.crownAt(zc) - u.keelAt(zc));
  if (p.trough > 0.05 && aftValley <= cabValley * 1.2)
    bad(`aft valley ${aftValley.toFixed(3)} is no deeper than the cabin's ${cabValley.toFixed(3)}`);

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
  console.log(`  width/height: nose section ${noseW.toFixed(2)}, cabin ${gotWidth.toFixed(2)} ` +
              `(two lobes give ${wantWidth.toFixed(2)})`);
  console.log(`  valley below the lobes: cabin ${(100 * cabValley).toFixed(1)}%, ` +
              `aft ${(100 * aftValley).toFixed(1)}% of height`);
  console.log(`  fastest SHAPE change ${worstStep.toFixed(2)} per unit length ` +
              `at z ${worstZ.toFixed(1)}`);
}

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: three sections, one closed body, no rings');
process.exit(failures ? 1 : 0);
