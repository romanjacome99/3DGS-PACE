/* Snapshot timeline helpers.
 *
 * The set of snapshot instants is a property of the data, not of the page: the 30k protocol replays
 * written by scripts/run_web_snapshots.sh carry init, t5s..t600s and a `final` tag at the 30k
 * iteration cap, and a run that hits the cap early simply has no late tags. Deriving the timeline
 * from the manifest keeps the slider honest instead of pinning it to whatever the build happened to
 * produce when the page was written.
 */

/** Wall-clock seconds a tag stands for; `final` sorts last because it is the iteration cap, not an instant. */
export function tagSeconds(tag) {
  if (tag === 'init') return 0;
  if (tag === 'final') return Infinity;
  return +tag.slice(1, -1);
}

/** Short label for the slider ticks and the film-strip headers. */
export function tagLabel(tag) {
  if (tag === 'init') return 'init';
  if (tag === 'final') return 'end of run';
  return tagSeconds(tag) + ' s';
}

/** Ordered union of the snapshot tags present anywhere in a scene (backends and methods differ). */
export function timelineFor(sceneData) {
  const seen = new Set(['init']);
  for (const be of Object.values(sceneData.backends || {})) {
    for (const m of Object.values(be.methods || {})) {
      for (const s of m.snapshots || []) seen.add(s.tag);
    }
  }
  return [...seen].sort((a, b) => tagSeconds(a) - tagSeconds(b));
}
