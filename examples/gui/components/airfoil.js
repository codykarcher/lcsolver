/**
 * Aerofoil sections.
 *
 * A section here is a closed loop of points in chord-normalised coordinates:
 * x from 0 at the leading edge to 1 at the trailing edge, y up, ordered from
 * the TRAILING EDGE forward over the upper surface, round the nose, and back
 * along the lower surface. That is the order every coordinate file in the world
 * uses (Selig order), so a section read off disk drops straight in.
 *
 * Every section is built on the SAME chordwise distribution, which is what lets
 * a wing blend between two different aerofoils by interpolating point for
 * point. Two sections with the same point count but different x stations cannot
 * be blended at all -- the result is a shape neither of them has -- so the
 * distribution lives here rather than in the caller.
 *
 * The distribution is cosine in x, clustered at both ends. That is not the same
 * choice as the fuselage's rings, which are spaced evenly along their outline:
 * a fuselage section has no distinguished points, and an aerofoil has two. Nearly
 * all of an aerofoil's curvature is in the first few percent of chord, and a
 * trailing edge is a feature that has to be resolved rather than averaged over.
 */

/** Chordwise stations, cosine-clustered at leading and trailing edges. */
function chordStations(n) {
  const xs = [];
  for (let i = 0; i < n; i++) xs.push((1 - Math.cos(Math.PI * i / (n - 1))) / 2);
  return xs;
}

/**
 * A NACA 4-digit section: `naca4('2412')`.
 *
 *   first digit   maximum camber, in per cent of chord
 *   second        where that camber is, in tenths of chord
 *   last two      maximum thickness, in per cent of chord
 *
 * The thickness polynomial's last coefficient is -0.1036 rather than the
 * original -0.1015. The original leaves the trailing edge 0.21% of chord thick,
 * which is correct to the definition and a nuisance in a solid model: the two
 * surfaces never meet, so the section is not a closed loop and the wing has a
 * slot down its trailing edge. -0.1036 closes it and moves nothing else by more
 * than a thousandth of chord.
 */
export function naca4(code, n = 80) {
  const s = String(code).padStart(4, '0');
  const m = parseInt(s[0], 10) / 100;         // camber
  const p = parseInt(s[1], 10) / 10;          // camber position
  const t = parseInt(s.slice(2), 10) / 100;   // thickness

  const halfThickness = (x) => 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x
    - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
  const camber = (x) => {
    if (m === 0 || p === 0) return { y: 0, slope: 0 };
    return x < p
      ? { y: m / (p * p) * (2 * p * x - x * x), slope: 2 * m / (p * p) * (p - x) }
      : { y: m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x * x),
          slope: 2 * m / ((1 - p) ** 2) * (p - x) };
  };
  const surface = (x, sign) => {
    const { y, slope } = camber(x);
    const th = Math.atan(slope), yt = halfThickness(x);
    return [x - sign * yt * Math.sin(th), y + sign * yt * Math.cos(th)];
  };

  const xs = chordStations(n), pts = [];
  for (let i = n - 1; i >= 0; i--) pts.push(surface(xs[i], 1));    // TE -> LE, upper
  for (let i = 1; i <= n - 2; i++) pts.push(surface(xs[i], -1));   // LE -> TE, lower
  return Object.assign(pts, {
    name: `NACA ${s}`, n, thickness: t, camber: m, camberAt: p,
  });
}

/**
 * Put an arbitrary section onto the standard distribution so it can be blended.
 *
 * Takes points in Selig order and re-reads them at this file's own chordwise
 * stations, upper and lower separately. Without this, a measured aerofoil and a
 * NACA one cannot appear on the same wing.
 */
export function resample(points, n = 80) {
  const lead = points.reduce((best, p, i) => (p[0] < points[best][0] ? i : best), 0);
  const at = (chain, x) => {
    for (let i = 0; i < chain.length - 1; i++) {
      const [x0, y0] = chain[i], [x1, y1] = chain[i + 1];
      if ((x - x0) * (x - x1) <= 0 && Math.abs(x1 - x0) > 1e-12) {
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
      }
    }
    return chain[chain.length - 1][1];
  };
  const upper = points.slice(0, lead + 1).slice().reverse();   // LE -> TE
  const lower = points.slice(lead);                            // LE -> TE
  const xs = chordStations(n), pts = [];
  for (let i = n - 1; i >= 0; i--) pts.push([xs[i], at(upper, xs[i])]);
  for (let i = 1; i <= n - 2; i++) pts.push([xs[i], at(lower, xs[i])]);
  return Object.assign(pts, { name: 'resampled', n });
}

/** Maximum thickness of a section, as a fraction of chord. */
export function thicknessOf(section) {
  const n = section.n;
  let t = 0;
  for (let i = 1; i < n - 1; i++) t = Math.max(t, section[i][1] - section[section.length - i][1]);
  return t;
}

/** Whatever the caller gave us, as a section: a NACA code, or points. */
export function asSection(spec, n = 80) {
  if (Array.isArray(spec)) return spec.n === n ? spec : resample(spec, n);
  return naca4(spec, n);
}
