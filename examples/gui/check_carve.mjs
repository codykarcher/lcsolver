/**
 * Does the duct carve do what it claims, and only that?
 *
 * Three questions, and the third matters most. The carve exists to expose the
 * engines, and it has to leave a closed shell -- but the binding condition is
 * that NOTHING ELSE MOVES. That body was arrived at by hand over many attempts
 * and every earlier way of making a duct changed it, so "the rest is untouched"
 * is not a nicety here, it is the requirement, and it is checked literally:
 * every triangle of the original that lies clear of the duct must still be in
 * the carved mesh, the same three vertices, bit for bit.
 *
 * Each check is run against deliberately broken geometry as well. Twice this
 * session a check passed on a body that was visibly wrong, so a check that has
 * not been shown to fail is not evidence.
 */
import { readFileSync } from 'fs';
import * as THREE from 'three';
import { deckFromSolve } from './components/aircraft.js';
import { d8Aircraft } from './components/d8_aircraft.js';
import { bareTurbofan, turbofan } from './components/engines.js';
import { ductVolume, clipTriangles } from './components/carve.js';

const sol = JSON.parse(readFileSync(new URL('./decks/b737_d8_solve.json', import.meta.url), 'utf8'));
const deck = deckFromSolve(sol);

let failures = 0;
const bad = (m) => { console.log(`  FAIL  ${m}`); failures++; };

/**
 * Geometries are thrown away as soon as each comparison is done with them.
 *
 * At the quality this now runs at an aeroplane is a quarter of a million
 * triangles, and this builds four of them; holding all four at once ran the
 * heap out mid-check, which reads as a broken carve rather than a broken test.
 */
const drop = (g) => g.traverse((o) => { if (o.isMesh) o.geometry.dispose(); });

const plain = d8Aircraft(deck, { sitOnGround: false, ductCarve: false });
const cut = d8Aircraft(deck, { sitOnGround: false, ductCarve: true });
const fu = cut.userData.parts.fuselage.userData;
/**
 * The cut, as the aeroplane built it.
 *
 * Taken from the aeroplane rather than rebuilt from a spec: the cut is now two
 * regions -- the hull of the two ducts aft, a rectangle forward -- and there is
 * no single set of numbers a checker could reassemble it from without copying
 * the construction and being wrong in its own way.
 */
const duct = fu.duct;

/**
 * Walk a mesh's triangles, handing each one to `fn` as a key and three points.
 *
 * Streamed rather than collected. At this density the skin is 176,000
 * triangles and building the list twice over -- once per aeroplane, four
 * strings each -- ran the heap out before any check had finished.
 */
function eachTriangle(geo, fn) {
  const pos = geo.getAttribute('position');
  const index = geo.getIndex();
  const n = index ? index.count : pos.count;
  const at = index ? (i) => index.getX(i) : (i) => i;
  const v = new THREE.Vector3();
  const c = ['', '', ''], xyz = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let t = 0; t + 2 < n; t += 3) {
    for (let k = 0; k < 3; k++) {
      v.fromBufferAttribute(pos, at(t + k));
      c[k] = `${v.x.toFixed(6)},${v.y.toFixed(6)},${v.z.toFixed(6)}`;
      xyz[k * 3] = v.x; xyz[k * 3 + 1] = v.y; xyz[k * 3 + 2] = v.z;
    }
    fn(c.slice().sort().join('|'), xyz);
  }
}

/* ---- 1. everything clear of the duct is untouched ----------------------- */
// The test is on triangles wholly clear of the duct by a margin, since one
// sitting exactly on the boundary is legitimately allowed to be split.
const after = new Set();
eachTriangle(cut.userData.parts.fuselage.userData.skinMesh.geometry, (k) => after.add(k));
// Kept for the control below, which has to run after the uncarved aeroplane
// has been released.
const plainSkin = plain.userData.parts.fuselage.userData.skinMesh.geometry.clone();
const p = new THREE.Vector3();
let total = 0, clear = 0, lost = 0;
eachTriangle(plain.userData.parts.fuselage.userData.skinMesh.geometry, (k, xyz) => {
  total++;
  for (let i = 0; i < 3; i++) {
    if (!(duct.depth(p.set(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2])) < -0.02)) return;
  }
  clear++;
  if (!after.has(k)) lost++;
});
console.log(`skin ${total} triangles, ${clear} clear of the duct`);
if (lost) bad(`${lost} triangles clear of the duct were changed`);

/* ---- 2. the engines are no longer buried -------------------------------- */
/**
 * Sampled on the nacelles' own outer surface, in aircraft coordinates, and
 * asked whether that point is still in solid material -- inside the body's
 * section AND outside the duct. Sampling the axis instead would pass on a duct
 * far too narrow to clear the fan.
 */
const d = cut.userData.deck;
const rN = d.nacelleDia / 2, axis = cut.userData.engineAxisY;
// The engine's extent is MEASURED off the built pod. The turbofan is not
// centred on its own origin, so `engineX +/- length/2` misses it by over a
// metre, and the sampling would then miss the part that is actually buried.
const pod = cut.userData.parts.engines[0];
pod.updateMatrixWorld(true);
const box = new THREE.Box3().setFromObject(pod);
const [xNose, xTail] = [-box.max.z, -box.min.z];
console.log(`nacelle runs x ${xNose.toFixed(2)} to ${xTail.toFixed(2)}; the cut is the ducts' ` +
            `hull, from the tail forward to their throat at x ${fu.duct.throatX.toFixed(2)}`);
/**
 * How far the nacelle reaches into solid material, at its worst.
 *
 * DEPTH, not a count of points. Counting says 1.0% of the nacelle is buried
 * where the truth is that the duct's wall stands exactly on the engine's
 * equator and touches it along a line -- 0.00 mm deep. That is the clearance
 * the configuration has, not a fault, and a check that cannot tell tangency
 * from burial would either fail on a correct duct or have to be loosened until
 * it stopped catching real ones.
 */
function deepestBurial(volume) {
  let worst = -Infinity;
  for (let i = 0; i <= 80; i++) {
    const x = xNose + (xTail - xNose) * (i / 80);
    for (let j = 0; j < 96; j++) {
      const th = (j / 96) * Math.PI * 2;
      const px = d.engineY + rN * Math.cos(th), py = axis + rN * Math.sin(th);
      p.set(px, py, -x);
      if (fu.depthInside(px, py, -x) <= 0) continue;      // not in the body
      // Inside the body, how far from getting out through the duct.
      worst = Math.max(worst, volume ? -volume.depth(p) : fu.depthInside(px, py, -x));
    }
  }
  return worst === -Infinity ? 0 : worst;
}
/**
 * The engine as a CYLINDER of its widest radius, over its whole length.
 *
 * Fair for the bare engine this was written against, and wrong for a nacelle:
 * a cowl is that wide only at one station and tapers away from it, so the
 * cylinder pokes 7 mm through a duct wall the nacelle itself clears by 0. The
 * cylinder is kept for the BEFORE reading, where it says something true and
 * useful -- how much body there was to get out of the way -- and the AFTER
 * reading is taken from the engine's own geometry, in check 8.
 */
const wasBuried = deepestBurial(null);
console.log(`a cylinder of the engine's widest radius sat ${(1000 * wasBuried).toFixed(0)} mm ` +
            `into the body before the carve`);
if (wasBuried < 0.1) bad('nothing was buried to begin with -- the test proves nothing');

/* ---- 3. the shell is closed along the cut ------------------------------- */
/**
 * The skin's new boundary and the floor's boundary have to be the same curve.
 * Counted as open edges: an edge used once is a boundary, and every boundary
 * edge of one sheet should land on the other. Compared by proximity rather than
 * identity, because the two sheets cut on DIFFERENT surfaces -- the skin on the
 * duct, the floor on the body -- and meet on the curve where those cross.
 */
/**
 * `near` restricts the walk to triangles inside a box, which is what makes this
 * affordable on the fine skin: 176,000 triangles carry half a million edges,
 * and all but a few hundred of them are nowhere near the duct. Safe because the
 * box is drawn a metre clear of the cut, so an edge wrongly called a boundary
 * for want of its neighbour is far outside the region the result is read in.
 */
function boundary(geo, near = null) {
  const pos = geo.getAttribute('position');
  const index = geo.getIndex();
  const n = index ? index.count : pos.count;
  const at = index ? (i) => index.getX(i) : (i) => i;
  const v = new THREE.Vector3();
  const key = (i) => {
    v.fromBufferAttribute(pos, at(i));
    return `${v.x.toFixed(5)},${v.y.toFixed(5)},${v.z.toFixed(5)}`;
  };
  const uses = new Map();
  const seg = new Map();
  const c = new THREE.Vector3();
  for (let t = 0; t + 2 < n; t += 3) {
    if (near) {
      c.set(0, 0, 0);
      for (let k = 0; k < 3; k++) c.add(v.fromBufferAttribute(pos, at(t + k)));
      if (!near(c.multiplyScalar(1 / 3))) continue;
    }
    const k = [key(t), key(t + 1), key(t + 2)];
    for (let e = 0; e < 3; e++) {
      const a = k[e], b = k[(e + 1) % 3];
      const id = a < b ? `${a}/${b}` : `${b}/${a}`;
      uses.set(id, (uses.get(id) || 0) + 1);
      if (!seg.has(id)) {
        const pa = a.split(',').map(Number), pb = b.split(',').map(Number);
        seg.set(id, {
          a: new THREE.Vector3(...pa), b: new THREE.Vector3(...pb),
          mid: new THREE.Vector3(...pa).lerp(new THREE.Vector3(...pb), 0.5),
        });
      }
    }
  }
  return [...uses].filter(([, c]) => c === 1).map(([id]) => seg.get(id));
}

/**
 * Distance from a point to a segment.
 *
 * Point-to-POINT between edge midpoints was the first measure and it reads
 * half an edge length even when two boundaries lie exactly on each other --
 * 66 mm on this mesh, which is indistinguishable from a real 66 mm crack. The
 * two sheets are meshed at different resolutions and their vertices have no
 * reason to coincide, so only distance to the other CURVE means anything.
 */
// Written without allocating: this runs a few million times now that the
// meshes are ten times finer, and three Vector3s per call was most of the cost.
const toSegment = (q, s) => {
  const abx = s.b.x - s.a.x, aby = s.b.y - s.a.y, abz = s.b.z - s.a.z;
  const l2 = abx * abx + aby * aby + abz * abz;
  const t = l2 < 1e-18 ? 0 : Math.min(1, Math.max(0,
    ((q.x - s.a.x) * abx + (q.y - s.a.y) * aby + (q.z - s.a.z) * abz) / l2));
  const dx = q.x - (s.a.x + abx * t), dy = q.y - (s.a.y + aby * t), dz = q.z - (s.a.z + abz * t);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};
/**
 * Drop edges that are only topologically open.
 *
 * Cutting one surface at a time means a triangle can be split while its
 * neighbour, already outside a different surface, is kept whole. The shared
 * edge is then one segment on one side and two collinear segments on the
 * other, so edge-counting calls all three a boundary -- and there is no hole,
 * the two sides lie exactly on each other. Counting those as gaps put 26 false
 * open edges in a mesh with none, which is the kind of number that sends you
 * looking for a fault that is not there.
 *
 * Detected by the thing that defines them: the midpoint of a T-junction edge
 * lies exactly on another boundary edge.
 */
function realHoles(edges) {
  return edges.filter((s, i) =>
    !edges.some((o, j) => j !== i && toSegment(s.mid, o) < 1e-4));
}
// Narrowed to the duct's own rim BEFORE the T-junction pass, which compares
// every edge against every other: on a mesh this fine the skin has thousands of
// open edges elsewhere, and sorting those out first turned a quadratic over all
// of them into one over the few hundred that matter.
/**
 * Measured at the cut's VERTICES, not at its edge midpoints.
 *
 * A midpoint is not on the cut. It is the middle of a chord across a curved
 * rim, so it sits off that rim by the chord's sagitta -- which is a property of
 * how finely the skin is meshed, not of whether there is a hole. That made the
 * measure read the skin's density: the same geometry gave 5 strays with a fine
 * skin and 11 with a coarse one, and tuning the duct against it chased
 * something that was never there.
 *
 * The cut's vertices ARE on it, exactly, since that is where the clipper solved
 * for the crossing. If one of them is far from the duct's rim, there is a hole.
 */
const nearDuct = (q) => duct.depth(q) > -1.0 && -q.z > duct.fromX - 1;
const skinEdge = realHoles(
  boundary(cut.userData.parts.fuselage.userData.skinMesh.geometry, nearDuct)
    .filter((s) => duct.depth(s.mid) > -0.05));   // only the edge the carve made
/**
 * All of the cut's surface, not just its walls.
 *
 * The cut is three pieces now -- the swept wall, the step where the rectangle
 * meets the hull, and the cap across the forward end -- and a skin edge closed
 * by any of them is closed. Comparing against the wall alone reported 69
 * unclosed vertices with nothing within 1.6 m, which was the cap it was not
 * looking at.
 */
const floorEdge = [
  ...boundary(fu.duct.mesh.geometry),
  ...(fu.duct.step ? boundary(fu.duct.step.geometry) : []),
  ...(fu.duct.front ? boundary(fu.duct.front.geometry) : []),
];
console.log(`cut edge: ${skinEdge.length} open edges on the skin, ${floorEdge.length} on the floor`);
const nearest = (q, set) => set.reduce((m, s) => Math.min(m, toSegment(q, s)), Infinity);
const TOL = 0.01;                                // 10 mm on a 30 m aeroplane
const cutPoints = new Map();
for (const s of skinEdge) {
  for (const q of [s.a, s.b]) cutPoints.set(`${q.x.toFixed(5)},${q.y.toFixed(5)},${q.z.toFixed(5)}`, q);
}
const gaps = [...cutPoints.values()].map((q) => nearest(q, floorEdge));
const strays = gaps.filter((g) => g > TOL);
console.log(`worst gap ${gaps.length ? Math.max(...gaps).toFixed(4) : 'n/a'} m, ` +
            `${strays.length} of ${gaps.length} cut vertices further than ${TOL * 1000} mm from the duct`);
if (!skinEdge.length) bad('the carve left no open edge at all -- it cut nothing');
if (strays.length > gaps.length * 0.02) bad(`${strays.length} of ${gaps.length} cut vertices are unclosed`);

/* ---- 4. the duct's surface faces into the duct --------------------------- */
/**
 * Every triangle of the floor and walls must face the void, which is up off the
 * floor and inboard off the walls.
 *
 * Worth its own check because the failure is silent: backface culling makes a
 * wrongly wound sheet INVISIBLE, not inside-out, so the duct would simply look
 * as though the carve had left a hole -- which sends you looking at the
 * clipping, where nothing is wrong.
 */
{
  const pos = fu.duct.mesh.geometry.getAttribute('position');
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = new THREE.Vector3(), cen = new THREE.Vector3();
  let wrong = 0, total = 0;
  for (let i = 0; i + 2 < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i);
    b.fromBufferAttribute(pos, i + 1);
    c.fromBufferAttribute(pos, i + 2);
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
    n.crossVectors(b.clone().sub(a), c.clone().sub(a));
    if (n.lengthSq() < 1e-18) continue;
    /**
     * Toward the nearer duct axis, not "up and inboard".
     *
     * That was right for a trough open at the top. This cut has a ROOF, where
     * into the void is down, so one direction cannot say which way a face
     * should look -- it called every roof face inside out, 4,565 of them, on a
     * surface that is wound correctly.
     */
    const into = new THREE.Vector3(
      Math.sign(cen.x) * duct.spacing - cen.x,
      (duct.centreY ? duct.centreY(-cen.z) : duct.axisY) - cen.y, 0);
    total++;
    if (n.dot(into) < 0) wrong++;
  }
  console.log(`duct surface: ${total} triangles, ${wrong} facing away from the void`);
  if (wrong > total * 0.02) bad(`${wrong} of ${total} duct faces are wound inside out`);
}

/* ---- 5. the body still shades smooth ------------------------------------ */
/**
 * Clipping unshares every vertex -- the skin's 8,961 become 50,286 -- and a
 * mesh with no shared vertices gets one normal per FACE if you recompute them,
 * which shades a smooth body as a field of flat panels. Nothing about the shape
 * changes, so the carve looks like it wrecked the mesh when the mesh is right.
 *
 * Measured as the spread of each triangle's three vertex normals: flat shading
 * makes them identical, so a mesh where nearly every triangle has a spread of
 * zero is faceted no matter how fine it is.
 */
function flatFraction(geo, where = null) {
  const n = geo.getAttribute('normal');
  if (!n) return 1;
  const pos = geo.getAttribute('position');
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const q = new THREE.Vector3(), cen = new THREE.Vector3();
  let flat = 0, total = 0;
  for (let i = 0; i + 2 < n.count; i += 3) {
    if (where) {
      cen.set(0, 0, 0);
      for (let k = 0; k < 3; k++) cen.add(q.fromBufferAttribute(pos, i + k));
      if (!where(cen.multiplyScalar(1 / 3))) continue;
    }
    a.fromBufferAttribute(n, i);
    b.fromBufferAttribute(n, i + 1);
    c.fromBufferAttribute(n, i + 2);
    total++;
    if (a.distanceTo(b) < 1e-6 && a.distanceTo(c) < 1e-6) flat++;
  }
  return total ? flat / total : 1;
}
// The duct is measured only on its CORNER ROUNDS. Its floor and its walls are
// flat by construction, so identical normals there are correct, and counting
// them made a correctly shaded sheet read as 17% faceted. The arcs are the part
// that has to be smooth, and the part that shows if it is not.
/**
 * The arcs at the two ends of the hull -- the part that has to shade smooth.
 *
 * Written against the hull's own geometry: a point is on an arc when it is
 * beyond the flat span, which is what `spacing` measures. The old test asked
 * for `halfWidth` and `cornerR`, numbers the cut no longer has, so it selected
 * on NaN and reported a smooth sheet as 100 per cent faceted.
 */
const onTheRound = (q) => Math.abs(q.x) > fu.duct.spacing + 0.05
  && -q.z >= fu.duct.throatX;      // the hull; forward of it the sides ARE flat
for (const [what, geo, where] of [
  ['skin', cut.userData.parts.fuselage.userData.skinMesh.geometry, null],
  ['duct corner rounds', fu.duct.mesh.geometry, onTheRound],
]) {
  const f = flatFraction(geo, where);
  console.log(`${what}: ${(100 * f).toFixed(1)}% of triangles are flat-shaded`);
  if (f > 0.10) bad(`${what} is ${(100 * f).toFixed(0)}% faceted -- normals were not carried through`);
}

/* ---- 6. the cut runs on through the fins -------------------------------- */
/**
 * The fins stand 20 mm outboard of the duct's wall and are 184 mm thick at the
 * root, so before the cut their inboard face came 92 mm through the wall as a
 * ridge down the inside of the trough. Nothing should now reach into the void.
 *
 * Measured as `min(into the duct, into the body)`, because that product is what
 * the cut removes: material inside the duct but outside the body was never
 * there to remove, and material inside the body but outside the duct is buried
 * and belongs where it is.
 */
function finReach(craft) {
  const cu = craft.userData, cf = cu.parts.fuselage.userData;
  const vol = cf.duct ?? duct;
  craft.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(craft.matrixWorld).invert();
  let worst = 0;
  for (const vt of cu.parts.verticalTails) {
    vt.traverse((o) => {
      if (!o.isMesh) return;
      const m = new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld);
      const a = o.geometry.getAttribute('position'), q = new THREE.Vector3();
      for (let i = 0; i < a.count; i++) {
        q.fromBufferAttribute(a, i).applyMatrix4(m);
        worst = Math.max(worst, Math.min(vol.depth(q), cf.depthInside(q.x, q.y, q.z)));
      }
    });
  }
  return worst;
}
const finWas = finReach(plain), finNow = finReach(cut);
console.log(`fins reach ${(1000 * finWas).toFixed(0)} mm into the trough before, ` +
            `${(1000 * finNow).toFixed(1)} mm after`);
if (finNow > 1e-3) bad(`the fins still reach ${(1000 * finNow).toFixed(0)} mm into the trough`);
if (finWas < 0.02) bad('the fins never reached into the trough -- the test proves nothing');

/* ---- 7. the lip is blended, not square ---------------------------------- */
/**
 * Measured as the turn in the surface normal just inside the seam, not at the
 * seam itself. At the seam the sheet is GIVEN the body's normal, so it reads
 * zero whether or not there is a blend there -- the question is whether the
 * turn is then spread over a fillet or taken in one step by the next row of
 * triangles.
 */
function lipTurn(craft) {
  const cu = craft.userData, cf = cu.parts.fuselage.userData;
  if (!cf.duct) return null;
  const geo = cf.duct.mesh.geometry;
  const pos = geo.getAttribute('position'), nrm = geo.getAttribute('normal');
  const q = new THREE.Vector3(), n = new THREE.Vector3();
  const h = 1e-4, angs = [];
  for (let i = 0; i < pos.count; i++) {
    q.fromBufferAttribute(pos, i);
    const a = cf.depthInside(q.x, q.y, q.z);
    if (!(a > 0.005 && a < 0.05)) continue;          // just inside the seam
    n.fromBufferAttribute(nrm, i);
    const g = new THREE.Vector3(
      cf.depthInside(q.x + h, q.y, q.z) - cf.depthInside(q.x - h, q.y, q.z),
      cf.depthInside(q.x, q.y + h, q.z) - cf.depthInside(q.x, q.y - h, q.z),
      cf.depthInside(q.x, q.y, q.z + h) - cf.depthInside(q.x, q.y, q.z - h));
    if (g.lengthSq() < 1e-18) continue;
    g.normalize().negate();
    angs.push(Math.acos(Math.max(-1, Math.min(1, g.dot(n)))) * 180 / Math.PI);
  }
  angs.sort((a, b) => a - b);
  return angs.length ? angs[Math.floor(angs.length / 2)] : null;
}
/**
 * The uncarved aeroplane is released first.
 *
 * Nothing after this point needs it, and at render density each aeroplane is a
 * hundred thousand triangles held as objects while it is carved. Keeping a
 * fourth one alive to compare against ran the heap out mid-check, which reads
 * as a broken carve rather than a test that asked for too much.
 */
drop(plain);
/**
 * Only meaningful when the deck asks for a blend.
 *
 * A sharp lip is a legitimate setting and is the current one -- the flare was
 * tearing the mesh where it grazed the skin, and a square edge that holds
 * together beats a rounded one that does not. Asserting that the blend turns
 * the lip when the blend is switched off is asserting that a thing does what it
 * has been told not to.
 */
const wantsBlend = (cut.userData.deck.ductBlend ?? 0) > 0;
const square = wantsBlend ? d8Aircraft(deck, { sitOnGround: false, ductBlend: 0 }) : null;
const turnSquare = square ? lipTurn(square) : null;
const turnBlend = lipTurn(cut);
if (square) drop(square);
if (!wantsBlend) {
  console.log(`lip: sharp by choice -- the normal turns ${turnBlend.toFixed(0)} deg within ` +
              `50 mm of the seam`);
} else {
  console.log(`lip: the normal turns ${turnSquare.toFixed(0)} deg within 50 mm of the seam ` +
              `unblended, ${turnBlend.toFixed(0)} deg blended`);
  if (turnBlend > 0.8 * turnSquare) bad(`the blend barely turns the lip (${turnBlend.toFixed(0)} vs ${turnSquare.toFixed(0)} deg)`);
  // Near square, not exactly: the median is taken over a meshed rim, so it lands
  // a degree or two either side of 90 depending on where the rows fall.
  if (turnSquare < 80) bad('the unblended lip was not square -- the test proves nothing');
}

/* ---- 8. the nacelles keep only what the body does not already fill ------- */
/**
 * The engines are nacelled now, and the nacelle is cut back to whatever is not
 * inside solid body -- solid meaning inside the body AND outside the duct,
 * since the trough is not body any more.
 *
 * With the duct carved this turns out to remove nothing, and that is the right
 * answer rather than a broken test: the duct's cradle was built to hold a
 * cylinder of exactly `nacelleDia`, so a nacelle of that diameter clears it.
 * The control below is therefore run with the duct OFF, where the same nacelle
 * is 69 per cent buried and there is plenty to take away.
 */
function nacelleReach(craft, useDuct) {
  const cu = craft.userData, cf = cu.parts.fuselage.userData;
  const vol = useDuct && cf.duct ? cf.duct : null;
  craft.updateMatrixWorld(true);
  const inv = new THREE.Matrix4().copy(craft.matrixWorld).invert();
  let worst = 0;
  for (const pod of cu.parts.engines) {
    pod.traverse((o) => {
      if (!o.isMesh) return;
      const m = new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld);
      const a = o.geometry.getAttribute('position'), q = new THREE.Vector3();
      for (let i = 0; i < a.count; i++) {
        q.fromBufferAttribute(a, i).applyMatrix4(m);
        /**
         * The body's own trailing edge does not count.
         *
         * `depthInside` steps there -- the section has a real width right up to
         * the last station and nothing at all aft of it -- so a cut placed on
         * that step lands on a face the body has no depth behind. 175 vertices
         * of 168,570 sit exactly on it and read as 39 mm buried in a body that
         * ends where they are. Skipped for the same reason tangency was: it is
         * a boundary, not material.
         */
        if (Math.abs(-q.z - cu.deck.fuseLength) < 1e-3) continue;
        const inBody = cf.depthInside(q.x, q.y, q.z);
        if (!(inBody > 0)) continue;
        // Inside the body -- but only solid if it is also outside the duct.
        const outOfDuct = vol ? -vol.depth(q) : inBody;
        worst = Math.max(worst, Math.min(inBody, outOfDuct));
      }
    });
  }
  return worst;
}
{
  /**
   * The ENGINE must be the size and in the place the deck put it, and the
   * nacelle must be bigger, because it wraps it.
   *
   * Asserting the nacelle's own radius against `nacelleDia` was the earlier
   * reading and it is what made the engine move: satisfying it means shrinking
   * the fan so a cowl fits inside the same envelope. The deck sizes the engine;
   * the cowl goes round the outside.
   */
  const bare = bareTurbofan({ rFan: 1, bypassRatio: 9 });
  const rf = (d.nacelleDia / 2) / bare.userData.rMax;
  const fanR = rf * bare.userData.rMax;
  const nacR = turbofan({ rFan: rf, bypassRatio: 9 }).userData.nacelleMaxRadius;
  const left = nacelleReach(cut, true);
  console.log(`engine radius ${fanR.toFixed(3)} m against the deck's ${(d.nacelleDia / 2).toFixed(3)}; ` +
              `nacelle wraps it at ${nacR.toFixed(3)} m; ` +
              `${(1000 * left).toFixed(1)} mm of it left in solid body`);
  if (Math.abs(fanR - d.nacelleDia / 2) > 1e-6) {
    bad(`the engine is ${fanR.toFixed(3)} m where the deck says ${(d.nacelleDia / 2).toFixed(3)}`);
  }
  if (!(nacR > fanR * 1.05)) bad('the nacelle does not wrap the engine -- it has replaced it');
  // A few millimetres survive where a triangle bulges through between corners
  // the clipper judged both outside. They are inside the body, hidden by it.
  if (left > 0.01) bad(`the nacelle is still ${(1000 * left).toFixed(0)} mm inside solid body`);
}

/* ---- negative controls -------------------------------------------------- */
/**
 * Each check, run against geometry it is supposed to reject. A check that has
 * never been seen to fail is not evidence that anything is right.
 */
console.log('\nnegative controls');
{
  // A carve that moves the untouched region: nudge one far-forward vertex.
  plainSkin.getAttribute('position').setY(0, plainSkin.getAttribute('position').getY(0) + 0.5);
  const moved = new Set();
  eachTriangle(plainSkin, (k) => moved.add(k));
  let n = 0;
  eachTriangle(cut.userData.parts.fuselage.userData.skinMesh.geometry, (k, xyz) => {
    for (let i = 0; i < 3; i++) {
      if (!(duct.depth(p.set(xyz[i*3], xyz[i*3+1], xyz[i*3+2])) < -0.02)) return;
    }
    if (!moved.has(k)) n++;
  });
  console.log(`  a moved vertex loses ${n} clear triangles` + (n ? '  ok' : '  NOT DETECTED'));
  if (!n) bad('control: a moved vertex was not detected as a change');
}
{
  // A duct too shallow to reach the engines leaves them buried.
  const shallow = { depth: (p) => fu.duct.depth(p) - 0.9 * rN };
  const f = deepestBurial(shallow);
  console.log(`  a duct stopping short leaves ${(1000 * f).toFixed(0)} mm buried` +
              (f > 1e-3 ? '  ok' : '  NOT DETECTED'));
  if (!(f > 1e-3)) bad('control: a too-shallow duct was not detected');
}
{
  // A floor built for a duct 30 cm narrower cannot close the skin's cut.
  const wrong = { depth: (p) => fu.duct.depth(p) - 0.3 };
  const g = clipTriangles(fu.duct.mesh.geometry, (q) => wrong.depth(q) + 0.3 - 0.3);
  const e = boundary(g);
  const pts = [...cutPoints.values()];
  const n = pts.filter((q) => nearest(q, e) > TOL).length;
  console.log(`  a floor 300 mm narrow leaves ${n} of ${pts.length} vertices open` +
              (n > pts.length * 0.02 ? '  ok' : '  NOT DETECTED'));
  if (!(n > pts.length * 0.02)) bad('control: a narrow floor was not detected');
}

{
  // Reversing the winding must be caught, or the check is decoration.
  const g = fu.duct.mesh.geometry.clone();
  const arr = g.getAttribute('position').array;
  for (let i = 0; i + 8 < arr.length; i += 9) {
    for (let k = 0; k < 3; k++) {
      const t = arr[i + 3 + k]; arr[i + 3 + k] = arr[i + 6 + k]; arr[i + 6 + k] = t;
    }
  }
  const pos = g.getAttribute('position');
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = new THREE.Vector3(), cen = new THREE.Vector3();
  let wrong = 0, total = 0;
  for (let i = 0; i + 2 < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i); b.fromBufferAttribute(pos, i + 1); c.fromBufferAttribute(pos, i + 2);
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
    n.crossVectors(b.clone().sub(a), c.clone().sub(a));
    if (n.lengthSq() < 1e-18) continue;
    total++;
    if (n.dot(new THREE.Vector3(-Math.sign(cen.x) * 0.01, 0.01, 0)) < 0) wrong++;
  }
  console.log(`  reversing the winding turns ${wrong} of ${total} faces the wrong way` +
              (wrong > total * 0.02 ? '  ok' : '  NOT DETECTED'));
  if (!(wrong > total * 0.02)) bad('control: a reversed winding was not detected');
}

{
  // Recomputing normals on the carved mesh -- the thing that caused it -- must
  // be caught, or the check cannot tell smooth from faceted.
  const g = cut.userData.parts.fuselage.userData.skinMesh.geometry.clone();
  g.computeVertexNormals();
  const f = flatFraction(g);
  console.log(`  recomputed normals leave ${(100 * f).toFixed(0)}% flat-shaded` +
              (f > 0.10 ? '  ok' : '  NOT DETECTED'));
  if (!(f > 0.10)) bad('control: face-normal shading was not detected');
}

{
  // With no duct the body is whole, the nacelle is deep inside it, and the
  // carve has real work to do. Both halves are checked: that it WAS buried,
  // and that afterwards it is not.
  const whole = d8Aircraft(deck, { sitOnGround: false, ductCarve: false, nacelles: false });
  const wf = whole.userData.parts.fuselage.userData;
  const probe = turbofan({ rFan: 1, bypassRatio: 9 });
  const rf = (d.nacelleDia / 2) / probe.userData.nacelleMaxRadius;
  const raw = new THREE.Group();
  raw.add(turbofan({ rFan: rf, bypassRatio: 9 }));
  raw.position.set(d.engineY, whole.userData.engineAxisY, -d.engineX);
  raw.updateMatrixWorld(true);
  let tot = 0, inside = 0;
  raw.traverse((o) => {
    if (!o.isMesh) return;
    const a = o.geometry.getAttribute('position'), q = new THREE.Vector3();
    for (let i = 0; i < a.count; i++) {
      q.fromBufferAttribute(a, i).applyMatrix4(o.matrixWorld);
      tot++;
      if (wf.depthInside(q.x, q.y, q.z) > 0) inside++;
    }
  });
  drop(whole);
  const cutWhole = d8Aircraft(deck, { sitOnGround: false, ductCarve: false, nacelles: true });
  const cf = cutWhole.userData.parts.fuselage.userData;
  cutWhole.updateMatrixWorld(true);
  const inv2 = new THREE.Matrix4().copy(cutWhole.matrixWorld).invert();
  let tot2 = 0, in2 = 0;
  for (const pod of cutWhole.userData.parts.engines) {
    pod.traverse((o) => {
      if (!o.isMesh) return;
      const m = new THREE.Matrix4().multiplyMatrices(inv2, o.matrixWorld);
      const a = o.geometry.getAttribute('position'), q = new THREE.Vector3();
      for (let i = 0; i < a.count; i++) {
        q.fromBufferAttribute(a, i).applyMatrix4(m);
        tot2++;
        if (cf.depthInside(q.x, q.y, q.z) > 1e-3) in2++;
      }
    });
  }
  drop(cutWhole);
  /**
   * Measured as the FRACTION left inside, not the depth of the deepest scrap.
   *
   * The clipper decides on vertex signs, so a triangle whose corners are both
   * outside can still bulge through the skin between them, and a handful of
   * those survive -- under one per cent of the cowl, and every one of them
   * inside the body, where the body's own skin hides it. Asking for zero depth
   * fails a carve that has done its job.
   */
  const before = 100 * inside / tot, after = 100 * in2 / tot2;
  console.log(`  with no duct a nacelle is ${before.toFixed(0)}% buried, ` +
              `and carves back to ${after.toFixed(1)}%` +
              (before > 20 && after < 2 ? '  ok' : '  NOT DETECTED'));
  if (!(before > 20)) bad('control: the nacelle was never buried -- the test proves nothing');
  if (!(after < 2)) bad(`control: the carve left ${after.toFixed(1)}% of the nacelle inside the body`);
}

console.log(failures ? `\n${failures} FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
