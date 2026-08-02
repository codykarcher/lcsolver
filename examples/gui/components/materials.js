/**
 * Shared materials for the component library.
 *
 * One place for the palette so that a strut looks like a strut wherever it
 * appears, and so a change of look is a change in one file rather than a sweep
 * through every component. Components should never construct their own
 * materials -- ask here.
 *
 * Materials are created once and shared across every mesh that uses them.
 * three.js is happy with that (a material carries no per-mesh state) and it
 * keeps the shader-program count down: the whole gear library draws with five
 * programs regardless of how many wheels are on screen.
 */
import * as THREE from 'three';

/** Rubber. Nearly black, and *not* pure black -- pure black shows no form. */
export const tire = new THREE.MeshStandardMaterial({
  color: 0x1c1c1f, roughness: 0.92, metalness: 0.0,
});

/** Machined aluminium rim. */
export const rim = new THREE.MeshStandardMaterial({
  color: 0xb8bcc2, roughness: 0.38, metalness: 0.85,
});

/** The polished oleo piston -- the bright band you see on a real gear. */
export const piston = new THREE.MeshStandardMaterial({
  color: 0xe8ebef, roughness: 0.12, metalness: 1.0,
});

/** Painted structure: the strut's outer cylinder, forks, bogie beams. */
export const structure = new THREE.MeshStandardMaterial({
  color: 0x6f7885, roughness: 0.55, metalness: 0.6,
});

/** Darker steel for axles, pins and pivots, so joints read as joints. */
export const hardware = new THREE.MeshStandardMaterial({
  color: 0x3f4650, roughness: 0.45, metalness: 0.9,
});

/* ---- engines ---------------------------------------------------------- */

/** Engine casing: light, slightly warm metal. */
export const casing = new THREE.MeshStandardMaterial({
  color: 0xc3c8ce, roughness: 0.34, metalness: 0.92,
});

/** Fan and compressor blades -- titanium, brighter than the case. */
export const blade = new THREE.MeshStandardMaterial({
  color: 0xd9dde3, roughness: 0.22, metalness: 1.0,
  side: THREE.DoubleSide,
});

/** Anything downstream of the burner: discoloured, dark, barely reflective. */
export const hot = new THREE.MeshStandardMaterial({
  color: 0x4a4038, roughness: 0.74, metalness: 0.75,
});

/** Spinners, cowl noses, propeller blades -- painted rather than bare. */
export const painted = new THREE.MeshStandardMaterial({
  color: 0x23262b, roughness: 0.48, metalness: 0.15,
});

/** Accessories, gearboxes, crankcases: cast and unpolished. */
export const accessory = new THREE.MeshStandardMaterial({
  color: 0x878d95, roughness: 0.72, metalness: 0.55,
});

/** Cylinder barrels and cooling fins on a piston engine. */
export const finned = new THREE.MeshStandardMaterial({
  color: 0x555b62, roughness: 0.85, metalness: 0.5,
});

/** Copper-ish: motor windings seen through a vented can. */
export const winding = new THREE.MeshStandardMaterial({
  color: 0xa8632f, roughness: 0.55, metalness: 0.8,
});

export const all = { tire, rim, piston, structure, hardware,
                     casing, blade, hot, painted, accessory, finned, winding };

/** Free every material. Call when tearing down a scene you built. */
export function dispose() {
  for (const m of Object.values(all)) m.dispose();
}
