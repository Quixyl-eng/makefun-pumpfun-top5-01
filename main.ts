// pumpfun-top5-forecast-01 entry (Node 22, standard library only).
//
// Derived from the make.fun house baseline (MIT, see LICENSE and NOTICE): the same
// /input and /output wiring, the log10(cap in SOL) = log10(lamports) - 9 size, and
// the erfc approximation used for size bins. The forecast is not the baseline.
//
// Each ranked coin is scored from the snapshots the house actually gives us:
// market-cap level, SOL sitting in the venue (quote_reserves_lamports), bonding-curve
// progress (curve / migrating / pool), recency-weighted time near the top of the
// chart, momentum, and whether liquidity collapsed. There is no holder field in the
// history, so quote reserves and venue stand in for holder growth and curve progress.
// Place probabilities are a Plackett-Luce draw against an outsider bucket, mixed
// over every answer read. Each place's market cap is a fat-tailed normal in
// log10(SOL), centered on that rank's recent caps plus a shrunk drift, and the
// bins are averaged over the same reads. Equal scores break toward the smaller
// mint, compared as the 32 raw bytes, the same order the house uses for a cap tie.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const TEMP = 0.8;
const OUT_BASE = 2.2;
const OUT_GROWTH = 9.0;
const OUT_TAU_H = 1.05;
const MAX_CANDIDATES = 16;
const MICRO = 1_000_000;
const BASE58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
const mintBytesCache = new Map<string, Uint8Array | null>();

interface TaskFile {
  places?: number;
  output?: string;
  reads?: string[];
  chart_size?: number;
  history_until?: string;
  size?: { edges?: number[] };
}

interface RawRow {
  rank?: number;
  mint?: string;
  market_cap_lamports?: string | number;
  venue?: string;
  created_at?: string;
  complete?: boolean;
  mayhem?: boolean;
  quote_reserves_lamports?: string | number;
}

interface Mark {
  at: number;
  rows: RawRow[];
}

interface Obs {
  t: number;
  rank: number;
  logCap: number;
  quote: number;
  venue: string;
  created: number;
}

interface Coin {
  mint: string;
  obs: Obs[];
}

interface Feat {
  mint: string;
  quality: number;
  halfLife: number;
  staleHours: number;
  logCap: number;
  rank: number;
  collapsed: boolean;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as TaskFile;
const places = Number.isInteger(task.places) && (task.places as number) >= 1 && (task.places as number) <= 20 ? (task.places as number) : 5;
const edges = Array.isArray(task.size?.edges) && task.size.edges.length > 0 ? task.size.edges : defaultEdges();

const marks = loadMarks(readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8'));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const anchorMs = marks[marks.length - 1]!.at;
const horizons = readHorizons(task, anchorMs);

const coins = buildCoins(marks);
const feats = [...coins.values()].map((coin) => featurize(coin, anchorMs));
const rankedFeats = feats
  .filter((f) => Number.isFinite(f.quality) && Number.isFinite(f.halfLife))
  .sort((a, b) => b.quality - a.quality || compareMint(a.mint, b.mint));
const candidates = rankedFeats.filter((f) => !f.collapsed).slice(0, MAX_CANDIDATES);
if (candidates.length < places) {
  for (const f of rankedFeats) {
    if (candidates.length >= places) break;
    if (!candidates.some((c) => c.mint === f.mint)) candidates.push(f);
  }
}
if (candidates.length < places) throw new Error(`the history names ${candidates.length} coins, fewer than the ${places} places`);

const allocated = allocatePlackett(candidates, horizons, places);
const extras = rankedFeats.filter((f) => !candidates.some((c) => c.mint === f.mint));
const probabilities = probabilityRows(candidates, allocated.p, places, extras);
const top5 = pickTop(candidates, allocated.p, places);
const sizes = sizeRows(marks, edges, horizons, places);

const outputName = typeof task.output === 'string' && /^[A-Za-z0-9._-]{1,128}$/.test(task.output) ? task.output : 'forecast.json';
const outPath = join(OUTPUT_DIR, outputName);
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify({ probabilities, top5, sizes })}\n`);

/** 32-byte payload of a base58 mint, or null when the text is not one. */
function mintBytes(mint: string): Uint8Array | null {
  const cached = mintBytesCache.get(mint);
  if (cached !== undefined) return cached;
  let zeros = 0;
  while (zeros < mint.length && mint[zeros] === '1') zeros += 1;
  const size = ((mint.length - zeros) * 733) / 1000 + 1;
  const bin = new Uint8Array(Math.ceil(size));
  let length = 0;
  for (let i = zeros; i < mint.length; i++) {
    const digit = BASE58.indexOf(mint[i]!);
    if (digit < 0) {
      mintBytesCache.set(mint, null);
      return null;
    }
    let carry = digit;
    let j = 0;
    for (let k = bin.length - 1; (carry !== 0 || j < length) && k >= 0; k--, j++) {
      carry += 58 * bin[k]!;
      bin[k] = carry % 256;
      carry = Math.floor(carry / 256);
    }
    length = j;
  }
  let start = 0;
  while (start < bin.length && bin[start] === 0) start += 1;
  const out = new Uint8Array(zeros + (bin.length - start));
  out.set(bin.subarray(start), zeros);
  const decoded = out.length === 32 ? out : null;
  mintBytesCache.set(mint, decoded);
  return decoded;
}

/** Ascending order of the mint's 32 bytes. Text order only if one side does not decode. */
function compareMint(a: string, b: string): number {
  const left = mintBytes(a);
  const right = mintBytes(b);
  if (left && right) {
    for (let i = 0; i < 32; i++) if (left[i] !== right[i]) return left[i]! - right[i]!;
    return 0;
  }
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

function defaultEdges(): number[] {
  const out: number[] = [];
  for (let i = 0; i < 81; i++) out.push(Math.round((2 + i * 0.05) * 100) / 100);
  return out;
}

function loadMarks(text: string): Mark[] {
  const out: Mark[] = [];
  for (const line of text.split('\n')) {
    if (line === '') continue;
    let mark: { ok?: boolean; at?: string; rows?: RawRow[] };
    try {
      mark = JSON.parse(line) as { ok?: boolean; at?: string; rows?: RawRow[] };
    } catch {
      continue;
    }
    if (mark.ok !== true || !Array.isArray(mark.rows) || typeof mark.at !== 'string') continue;
    const at = Date.parse(mark.at);
    if (!Number.isFinite(at)) continue;
    out.push({ at, rows: mark.rows });
  }
  out.sort((a, b) => a.at - b.at);
  return out;
}

function readHorizons(file: TaskFile, anchor: number): number[] {
  const out: number[] = [];
  if (Array.isArray(file.reads)) {
    for (const read of file.reads) {
      const t = Date.parse(read);
      if (Number.isFinite(t)) out.push(Math.max(0, (t - anchor) / 3_600_000));
    }
  }
  if (out.length === 0) out.push(1);
  return out;
}

/** log10 of a non-negative integer, from a decimal string or a number. */
function log10Int(value: string | number | undefined): number {
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || value <= 0) return Number.NEGATIVE_INFINITY;
    return Math.log10(value);
  }
  if (typeof value !== 'string') return Number.NEGATIVE_INFINITY;
  const s = value.trim();
  if (!/^[0-9]+$/.test(s)) {
    const n = Number(s);
    return Number.isFinite(n) && n > 0 ? Math.log10(n) : Number.NEGATIVE_INFINITY;
  }
  const digits = s.replace(/^0+/, '');
  if (digits.length === 0) return Number.NEGATIVE_INFINITY;
  const take = Math.min(15, digits.length);
  return Math.log10(Number(digits.slice(0, take))) + (digits.length - take);
}

function lamportsToSol(value: string | number | undefined): number {
  const logLamports = log10Int(value);
  if (!Number.isFinite(logLamports)) return 0;
  return 10 ** (logLamports - 9);
}

function buildCoins(history: Mark[]): Map<string, Coin> {
  const byMint = new Map<string, Coin>();
  for (const mark of history) {
    const seen = new Set<string>();
    for (const row of mark.rows) {
      if (!row || typeof row.mint !== 'string' || row.mint.length < 32 || row.mint.length > 44) continue;
      if (seen.has(row.mint)) continue;
      seen.add(row.mint);
      let coin = byMint.get(row.mint);
      if (!coin) {
        coin = { mint: row.mint, obs: [] };
        byMint.set(row.mint, coin);
      }
      const created = typeof row.created_at === 'string' ? Date.parse(row.created_at) : mark.at;
      coin.obs.push({
        t: mark.at,
        rank: Number.isFinite(row.rank) && (row.rank as number) > 0 ? (row.rank as number) : 999,
        logCap: log10Int(row.market_cap_lamports) - 9,
        quote: lamportsToSol(row.quote_reserves_lamports),
        venue: row.venue === 'pool' || row.venue === 'migrating' || row.venue === 'curve' ? row.venue : 'curve',
        created: Number.isFinite(created) ? created : mark.at,
      });
    }
  }
  return byMint;
}

function featurize(coin: Coin, anchor: number): Feat {
  const obs = coin.obs;
  const last = obs[obs.length - 1]!;
  const logCap = Number.isFinite(last.logCap) ? last.logCap : Number.NEGATIVE_INFINITY;
  const logQuote = Math.log10(last.quote + 0.05);
  const venueScore = last.venue === 'pool' ? 1 : last.venue === 'migrating' ? 0.5 : 0.15;
  const memoryMs = 40 * 60 * 1000;
  let weightSum = 0;
  let top10 = 0;
  let top20 = 0;
  for (const o of obs) {
    const w = Math.exp(-(last.t - o.t) / memoryMs);
    weightSum += w;
    if (o.rank <= 10) top10 += w;
    if (o.rank <= 20) top20 += w;
  }
  const confidence = 1 - Math.exp(-obs.length / 1.6);
  const persist10 = (top10 / weightSum) * confidence + 0.12 * (1 - confidence);
  const persist20 = (top20 / weightSum) * confidence + 0.2 * (1 - confidence);
  const target = last.t - 30 * 60 * 1000;
  let ref = obs[0]!;
  for (const o of obs) if (Math.abs(o.t - target) < Math.abs(ref.t - target)) ref = o;
  const dtHours = Math.max(1 / 60, (last.t - ref.t) / 3_600_000);
  let momentum = obs.length === 1 || !Number.isFinite(ref.logCap) ? 0 : (logCap - ref.logCap) / dtHours;
  if (!Number.isFinite(momentum)) momentum = 0;
  momentum = Math.max(-2, Math.min(2.2, momentum));
  let maxLog = logCap;
  for (const o of obs) if (Number.isFinite(o.logCap) && o.logCap > maxLog) maxLog = o.logCap;
  const drawdown = Number.isFinite(logCap) && Number.isFinite(maxLog) ? logCap - maxLog : -1;
  const ageHours = Math.max(0, (last.t - last.created) / 3_600_000);
  const staleHours = Math.max(0, (anchor - last.t) / 3_600_000);
  const collapsed = last.quote < 0.25;
  const dead = last.quote < 1.5 && last.venue !== 'pool';
  // The chart replaces its leader every few marks. Current rank and youth
  // outweigh a long stay in the top 10, which is how yesterday's #1 lingers.
  const quality =
    1.05 * logCap +
    0.75 * logQuote +
    1.45 * venueScore +
    0.55 * persist10 +
    0.2 * persist20 +
    0.22 * momentum +
    1.05 * drawdown +
    0.08 * Math.min(obs.length, 5) +
    1.15 * Math.exp(-(last.rank - 1) / 3) -
    0.45 * Math.min(ageHours, 4) -
    (collapsed ? 4 : dead ? 1.8 : 0) -
    0.35 * staleHours;
  let halfLife =
    0.4 +
    1.55 * venueScore +
    0.5 * Math.max(0, logQuote) +
    1.25 * persist10 +
    0.3 * confidence +
    0.15 * Math.max(0, Math.min(momentum, 1));
  if (collapsed) halfLife = 0.1;
  else if (dead) halfLife *= 0.3;
  if (drawdown < -0.35) halfLife *= 0.4;
  if (obs.length === 1 && ageHours < 0.2) halfLife *= 0.72;
  halfLife *= Math.exp(-0.25 * staleHours);
  halfLife *= Math.exp(-0.55 * Math.min(ageHours, 3));
  return {
    mint: coin.mint,
    quality,
    halfLife: Math.max(0.1, halfLife),
    staleHours,
    logCap,
    rank: last.rank,
    collapsed,
  };
}

function weightsAt(list: Feat[], horizonHours: number): { w: number[]; out: number } {
  const w = list.map((f) => Math.exp(f.quality / TEMP) * Math.exp((-Math.LN2 * horizonHours) / f.halfLife));
  let maxW = 1e-12;
  for (const weight of w) if (weight > maxW) maxW = weight;
  const out = maxW * (OUT_BASE + OUT_GROWTH * (1 - Math.exp(-horizonHours / OUT_TAU_H)));
  return { w, out };
}

/**
 * At each answer read, draw places without replacement from a Plackett-Luce model.
 * Weight is strength decayed by that coin's half-life over the hours until the read.
 * The outsider bucket is coins the history never ranked. It is not removed when
 * drawn: many of them can arrive. One forecast is scored at every read, so the
 * reported probability is the mean of the draw over task.reads.
 */
function allocatePlackett(list: Feat[], horizonList: number[], nPlaces: number): { p: number[][] } {
  const n = list.length;
  if (n > 16) throw new Error(`refusing a Plackett-Luce field of ${n}`);
  const acc = Array.from({ length: nPlaces }, () => new Array<number>(n).fill(0));
  const states = 1 << n;
  for (const horizon of horizonList) {
    const { w, out } = weightsAt(list, horizon);
    let state = new Float64Array(states);
    state[0] = 1;
    for (let step = 0; step < nPlaces; step++) {
      const next = new Float64Array(states);
      for (let mask = 0; mask < states; mask++) {
        const mass = state[mask] ?? 0;
        if (mass < 1e-15) continue;
        let remaining = out;
        for (let i = 0; i < n; i++) if ((mask & (1 << i)) === 0) remaining += w[i] ?? 0;
        if (!(remaining > 0)) continue;
        for (let i = 0; i < n; i++) {
          if ((mask & (1 << i)) !== 0) continue;
          const take = (mass * (w[i] ?? 0)) / remaining;
          acc[step]![i] = (acc[step]![i] ?? 0) + take;
          const bit = mask | (1 << i);
          next[bit] = (next[bit] ?? 0) + take;
        }
        next[mask] = (next[mask] ?? 0) + (mass * out) / remaining;
      }
      state = next;
    }
  }
  const reads = horizonList.length;
  for (let k = 0; k < nPlaces; k++) for (let i = 0; i < n; i++) acc[k]![i] = (acc[k]![i] ?? 0) / reads;
  return { p: acc };
}

function toMicro(p: number): number {
  if (!Number.isFinite(p) || p <= 0) return 0;
  return Math.max(0, Math.floor(p * MICRO + 1e-6));
}

function probabilityRows(
  list: Feat[],
  p: number[][],
  nPlaces: number,
  extras: Feat[],
): Array<{ mint: string; place: number; p: number }> {
  const rows: Array<{ mint: string; place: number; micro: number }> = [];
  for (let k = 0; k < nPlaces; k++) {
    const chosen: Array<{ mint: string; micro: number }> = [];
    for (let i = 0; i < list.length; i++) {
      const micro = toMicro(p[k]![i]!);
      if (micro > 0) chosen.push({ mint: list[i]!.mint, micro });
    }
    chosen.sort((a, b) => b.micro - a.micro || compareMint(a.mint, b.mint));
    let sum = 0;
    const kept: Array<{ mint: string; micro: number }> = [];
    const named = new Set<string>();
    for (const row of chosen) {
      if (kept.length === 100) break;
      if (sum + row.micro > MICRO) continue;
      kept.push(row);
      named.add(row.mint);
      sum += row.micro;
    }
    // A p of 0 still names the coin, so it does not share the unnamed mass.
    // That mass is what scores a coin launched after the history ends. Spend the
    // spare name slots on coins that are on the recent chart but are not a real
    // top-5 threat, and leave the residue for newcomers.
    const zeros = extras
      .filter((f) => !named.has(f.mint) && (f.rank > 20 || f.collapsed || f.staleHours > 0.4))
      .sort((a, b) => b.rank - a.rank || compareMint(a.mint, b.mint));
    for (const feat of zeros) {
      if (kept.length === 100) break;
      kept.push({ mint: feat.mint, micro: 0 });
      named.add(feat.mint);
    }
    for (const row of kept) rows.push({ mint: row.mint, place: k + 1, micro: row.micro });
  }
  return rows.map((row) => ({ mint: row.mint, place: row.place, p: row.micro / MICRO }));
}

function pickTop(list: Feat[], p: number[][], nPlaces: number): string[] {
  const scored = list.map((f, i) => {
    let expected = 0;
    let mass = 0;
    for (let k = 0; k < nPlaces; k++) {
      expected += (k + 1) * p[k]![i]!;
      mass += p[k]![i]!;
    }
    expected += (nPlaces + 1) * Math.max(0, 1 - mass);
    return { mint: f.mint, expected, lead: p[0]![i]! };
  });
  scored.sort((a, b) => a.expected - b.expected || b.lead - a.lead || compareMint(a.mint, b.mint));
  const top: string[] = [];
  for (const row of scored) {
    if (top.length === nPlaces) break;
    if (!top.includes(row.mint)) top.push(row.mint);
  }
  if (top.length < nPlaces) {
    const fill = [...list].sort((a, b) => a.rank - b.rank || compareMint(a.mint, b.mint));
    for (const f of fill) {
      if (top.length === nPlaces) break;
      if (!top.includes(f.mint)) top.push(f.mint);
    }
  }
  return top;
}

/** erfc, Numerical Recipes' erfcc (fractional error under 1.2e-7), from the house baseline. */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

function above(x: number, mean: number, sd: number): number {
  if (x === Number.NEGATIVE_INFINITY) return 1;
  if (x === Number.POSITIVE_INFINITY) return 0;
  return 0.5 * erfc((x - mean) / (sd * Math.SQRT2));
}

function rankedRows(mark: Mark): RawRow[] {
  const rows = mark.rows.filter((row) => typeof row.mint === 'string' && Number.isFinite(row.rank));
  rows.sort((a, b) => (a.rank as number) - (b.rank as number) || compareMint(String(a.mint), String(b.mint)));
  return rows;
}

/** log10(cap in SOL) of the coin at `place` on each mark that ranked that many coins. */
function rankSeries(history: Mark[], place: number): Array<{ t: number; logCap: number }> {
  const series: Array<{ t: number; logCap: number }> = [];
  for (const mark of history) {
    const rows = rankedRows(mark);
    if (rows.length < place) continue;
    const logCap = log10Int(rows[place - 1]!.market_cap_lamports) - 9;
    if (Number.isFinite(logCap)) series.push({ t: mark.at, logCap });
  }
  return series;
}

/** Median log cap of the last mark's top 10, the level a long horizon is pulled toward. */
function anchorLogCap(history: Mark[]): number {
  for (let i = history.length - 1; i >= 0; i--) {
    const rows = rankedRows(history[i]!).slice(0, 10);
    const logs: number[] = [];
    for (const row of rows) {
      const logCap = log10Int(row.market_cap_lamports) - 9;
      if (Number.isFinite(logCap)) logs.push(logCap);
    }
    if (logs.length > 0) {
      logs.sort((a, b) => a - b);
      const mid = logs.length >> 1;
      return logs.length % 2 === 1 ? logs[mid]! : (logs[mid - 1]! + logs[mid]!) / 2;
    }
  }
  return 3;
}

/**
 * Mean, sd, tail weight and right-jump weight of log10(cap in SOL) at one horizon.
 * The slope uses the last hour of marks. Under 2.5 hours it keeps a shrinking
 * share of that pace, which is the practice window. Longer horizons saturate.
 * The final reads sit 7 to 13 hours out, so that pace is not compounded.
 */
function sizeMeanSd(series: Array<{ t: number; logCap: number }>, horizonHours: number, anchor: number): { mean: number; sd: number; wide: number; jump: number } {
  if (series.length === 0) return { mean: anchor, sd: 0.5, wide: 0.4, jump: 0.25 };
  const last = series[series.length - 1]!;
  const prev = series.length >= 2 ? series[series.length - 2]! : last;
  const level = 0.85 * last.logCap + 0.15 * prev.logCap;
  let driftPerHour = 0;
  const window = series.filter((point) => last.t - point.t <= 50 * 60 * 1000);
  const fit = window.length >= 2 ? window : series.slice(-6);
  if (fit.length >= 2) {
    const t0 = fit[0]!.t;
    let n = 0;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (const point of fit) {
      const x = (point.t - t0) / 3_600_000;
      n += 1;
      sx += x;
      sy += point.logCap;
      sxx += x * x;
      sxy += x * point.logCap;
    }
    const den = n * sxx - sx * sx;
    const slope = Math.abs(den) > 1e-9 ? (n * sxy - sx * sy) / den : 0;
    driftPerHour = (Number.isFinite(slope) ? slope : 0) * (n / (n + 1.5));
  }
  if (driftPerHour > 1.6) driftPerHour = 1.6;
  if (driftPerHour < -0.8) driftPerHour = -0.8;
  // The practice window is under two hours and the cap is still climbing.
  // A long final horizon saturates instead of compounding the opening ramp.
  const integrated =
    horizonHours < 2.5 ? (driftPerHour * horizonHours) / (1 + horizonHours / 2.2) : driftPerHour * (1 - Math.exp(-horizonHours / 2));
  const pull = 1 - Math.exp(-horizonHours / 14);
  const mean = (1 - 0.15 * pull) * (level + integrated) + 0.15 * pull * anchor;
  const sd = Math.sqrt(0.2 ** 2 + 0.14 ** 2 * horizonHours);
  const wide = 0.28 + 0.22 * (1 - Math.exp(-horizonHours / 2));
  const jump = 0.18 + 0.22 * (1 - Math.exp(-horizonHours / 1.1));
  return { mean, sd: Math.max(0.2, sd), wide, jump };
}

/**
 * Three pieces, averaged over the answer reads: a core on the projected cap,
 * a wide band for a coin that replaces this place, and a right-hand jump for
 * a new leader that prints a higher cap than the last mark's.
 */
function binMass(lo: number, hi: number, mean: number, sd: number, wide: number, jump: number): number {
  const narrow = above(lo, mean, sd) - above(hi, mean, sd);
  const tail = above(lo, mean, sd * 2.2) - above(hi, mean, sd * 2.2);
  const right = above(lo, mean + 0.32, sd * 1.7) - above(hi, mean + 0.32, sd * 1.7);
  const core = Math.max(0, 1 - wide - jump);
  return core * narrow + wide * tail + jump * right;
}

function sizeRows(history: Mark[], binEdges: number[], horizonList: number[], nPlaces: number): Array<{ place: number; bin: number; p: number }> {
  const anchor = anchorLogCap(history);
  const rows: Array<{ place: number; bin: number; p: number }> = [];
  // Place k cannot be priced above place k-1 at the same read. Keep one cap per horizon.
  const meanCap = horizonList.map(() => Number.POSITIVE_INFINITY);
  for (let place = 1; place <= nPlaces; place++) {
    const series = rankSeries(history, place);
    const acc = new Array<number>(binEdges.length + 1).fill(0);
    for (let h = 0; h < horizonList.length; h++) {
      let fitted = sizeMeanSd(series, horizonList[h]!, anchor);
      const gap = 0.04 * (place - 1);
      if (fitted.mean > meanCap[h]! - gap) fitted = { ...fitted, mean: meanCap[h]! - gap };
      meanCap[h] = fitted.mean + gap;
      const jump = place === 1 ? fitted.jump : fitted.jump * 0.45;
      for (let bin = 0; bin <= binEdges.length; bin++) {
        const lo = bin === 0 ? Number.NEGATIVE_INFINITY : binEdges[bin - 1]!;
        const hi = bin === binEdges.length ? Number.POSITIVE_INFINITY : binEdges[bin]!;
        acc[bin] = (acc[bin] ?? 0) + binMass(lo, hi, fitted.mean, fitted.sd, fitted.wide, jump);
      }
    }
    const reads = horizonList.length;
    let sum = 0;
    const bins: Array<{ bin: number; micro: number }> = [];
    for (let bin = 0; bin < acc.length; bin++) {
      const micro = toMicro((acc[bin] ?? 0) / reads);
      if (micro > 0 && sum + micro <= MICRO) {
        bins.push({ bin, micro });
        sum += micro;
      }
    }
    for (const bin of bins) rows.push({ place, bin: bin.bin, p: bin.micro / MICRO });
  }
  return rows;
}
