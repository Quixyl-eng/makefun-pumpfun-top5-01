# pumpfun-top5-forecast-01

Forecast which five pump.fun coins launched since the opening will be largest by market cap at the contest's answer reads, and which market-cap bin each place falls in.

The program is `main.ts`. Node 22 runs it with the standard library only. It does not use the network or an API key.

## Run

From a directory that holds this file, with the kit's `input/` (or the grader's `/input`):

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

When those variables are unset, it reads `/input` and writes `/output/forecast.json`. Two runs on the same history write the same bytes.

## Method

Every coin the snapshots rank gets a strength and a half-life from the fields the history actually carries:

- market cap, and how it has moved over the last half hour
- SOL in the venue (`quote_reserves_lamports`), which is the liquidity the curve or pool holds
- bonding-curve progress: `pool` outlasts `migrating`, which outlasts a coin still on the `curve`
- how often it has sat in the top 10 and top 20, down-weighted when it has only printed once
- a collapse in quote reserves, or a gap since it last appeared in a snapshot

The history has no holder counts. Quote reserves and venue are the stand-in for holder growth and curve progress.

Hours from the last snapshot to each answer read come from `task.json`, not from constants. A coin's weight decays with its own half-life over that gap, so a short rehearsal and the 7-to-13-hour final are not the same forecast. At each read the top places are a Plackett-Luce draw against one outsider bucket: coins launched after the history ends, and coins the snapshots never ranked. The bucket is not removed when it is drawn. The file reports the average of that draw over the seven reads. Each place also names, at probability 0, recent chart coins that are not a real top-5 threat. Naming them removes them from the unnamed share, so that share is left for coins launched after the history ends.

Size bins are a mixture of two normals in log10 of the cap in SOL. The center is that rank's latest level plus a drift fit on the last 45 minutes, and the drift saturates instead of running for the whole horizon. The wide component gets heavier as the horizon grows. A lower place is not given a higher mean than the place above it at the same read.

Ties break by the mint's 32 bytes, ascending.
