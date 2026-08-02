import * as THREE from 'three';
/**
 * Does a pylon intrude on the fan or stator row?
 *
 * The zone is the ANNULUS between the core skin and the duct wall, over the
 * span of the fan case -- not a disc, because inside the core radius is core
 * structure the pylon is supposed to reach, and not unbounded in z, because
 * ahead of the case there is nothing to hit.
 */
export function intrusions(engine, pylonObj, engineZ = 0) {
  const u = engine.userData;
  // Aft limit is the aftmost VANE, not the case: the case trails the blading
  // by 0.04 fan radii, and it is the blading that must be missed.
  const caseAft = u.vaneAft ?? u.caseAft;
  const caseFwd = 0.18 * (u.rFan ?? 1), DUCT = 1.012 * (u.rFan ?? 1);
  const w = u.coreWall;
  const coreAt = (z) => {
    for (let i = 0; i < w.length - 1; i++) {
      const [z0, r0] = w[i], [z1, r1] = w[i + 1];
      if (z <= z0 && z >= z1) return r0 + (r1 - r0) * ((z0 - z) / ((z0 - z1) || 1));
    }
    return z > w[0][0] ? w[0][1] : w[w.length - 1][1];
  };
  let hits = 0, worst = null;
  pylonObj.updateMatrixWorld(true);
  pylonObj.traverse((o) => {
    if (!o.isMesh) return;
    const g = o.geometry.getAttribute('position');
    const v = new THREE.Vector3();
    for (let i = 0; i < g.count; i++) {
      v.fromBufferAttribute(g, i); o.localToWorld(v);
      const z = v.z - engineZ;
      if (z <= caseAft + 1e-6 || z > caseFwd + 1e-6) continue;
      const r = Math.hypot(v.x, v.y);
      if (r > coreAt(z) + 1e-6 && r < DUCT - 1e-6) {
        hits++; if (!worst || r < worst.r) worst = { r, z, part: o.name };
      }
    }
  });
  return { hits, worst };
}
