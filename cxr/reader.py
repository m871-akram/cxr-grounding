"""Pilot reader study (PLAN.md, section 7): 100 test-split images, 50 real and 50 edited, rated blind
by three clinicians who do not read chest X-rays every day.

    python -m cxr.reader select   # the images, fixed seed -> outputs/reader/key.csv, originals.txt
    python -m cxr.reader page     # the blinded rating page -> outputs/reader/rating_page.html

Runs on the Mac from the results/ tables and the downloaded images (no torch). The key, which says
which image is real or edited, stays in outputs/ (not in git) until every reader has answered; its
SHA-256 is recorded in PLAN.md before any reader sees an image.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json

import numpy as np
import pandas as pd
from PIL import Image

from . import DATA, OUTPUTS, RESULTS, ROOT

SEED = 20261001
GROUPS = [  # (group, number of images); edits first, from the smaller pools
    ("edit_cardiomegaly_checked", 15),  # audit additions that passed the three checks
    ("edit_growth_0.00", 10),           # dose run: "Cardiomegaly" inside the heart's own outline
    ("edit_growth_0.06", 10),           # dose run: the smallest enlargement
    ("edit_effusion_checked", 15),      # audit additions that passed the three checks
    ("real_cardiomegaly", 15),          # NIH label cardiomegaly, not effusion
    ("real_effusion", 15),              # NIH label effusion, not cardiomegaly
    ("real_normal", 20),                # NIH "No Finding"
]
READER = OUTPUTS / "reader"


def pools() -> dict[str, pd.DataFrame]:
    """Candidates of each group (test split): the image shown, its source film and patient, the
    intended finding, and MedGemma 4B's P(yes) where it was scored (re-scored tables, one <bos>)."""
    audit = pd.read_csv(RESULTS / "audit_4b_test.csv")
    measures = pd.read_csv(RESULTS / "h3_measures_test.csv")
    scores = pd.read_csv(RESULTS / "h3_scores_base_test.csv")
    p = scores.pivot_table(index=["kind", "image", "growth"], columns="question", values="p_yes", dropna=False)
    out = {}
    for finding in ["cardiomegaly", "effusion"]:
        e = audit[(audit["variant"] == "edit") & (audit["phrasing"] == 0) & (audit["finding"] == finding) & audit["valid"]]
        out[f"edit_{finding}_checked"] = pd.DataFrame({
            "image": e["image"], "patient_id": e["patient_id"], "path": [f"outputs/pairs/test/{finding}/edit/{i}" for i in e["image"]],
            "intended": finding, "growth": np.nan, f"p_yes_{finding}": e["p_yes"]})
    for g in [0.0, 0.06]:
        d = measures[(measures["kind"] == "edit") & (measures["growth"] == g)]
        out[f"edit_growth_{g:.2f}"] = pd.DataFrame({
            "image": d["image"], "patient_id": d["patient_id"], "path": [f"outputs/dose/cardiomegaly/growth{g:.2f}/{i}" for i in d["image"]],
            "intended": "cardiomegaly", "growth": g,
            "p_yes_cardiomegaly": [p.loc[("edit", i, g), "cardiomegaly_0"] for i in d["image"]]})
    files = pd.read_csv(DATA / "nih512" / "metadata.csv").set_index("image")["file"]  # where each original is
    r = measures[measures["kind"] == "real"]
    real = {"real_cardiomegaly": r[(r["cardiomegaly"] == 1) & (r["effusion"] == 0)],
            "real_effusion": r[(r["effusion"] == 1) & (r["cardiomegaly"] == 0)], "real_normal": r[r["no_finding"] == 1]}
    for group, d in real.items():
        key = [("real", i, np.nan) for i in d["image"]]
        out[group] = pd.DataFrame({
            "image": d["image"], "patient_id": d["patient_id"], "path": [f"data/nih512/{files[i]}" for i in d["image"]],
            "intended": {"real_cardiomegaly": "cardiomegaly", "real_effusion": "effusion", "real_normal": "none"}[group],
            "growth": np.nan, "p_yes_cardiomegaly": [p.loc[k, "cardiomegaly_0"] for k in key],
            "p_yes_effusion": [p.loc[k, "effusion_0"] for k in key]})
    return {g: d.sort_values("image").reset_index(drop=True) for g, d in out.items()}


def select(args: argparse.Namespace) -> None:
    """Draws the images group by group with one generator (SEED), never twice the same patient, so no
    original appears with its own edit (or two edits of one film); then gives them random ids."""
    rng = np.random.default_rng(SEED)
    candidates, used, picked = pools(), set(), []
    for group, n in GROUPS:
        pool, take = candidates[group], []
        for k in rng.permutation(len(pool)):  # a random order of the pool, one image per patient
            if pool.loc[k, "patient_id"] not in used:
                take.append(k)
                used.add(pool.loc[k, "patient_id"])
                if len(take) == n:
                    break
        assert len(take) == n, f"{group}: only {len(take)} patients left"
        picked.append(pool.loc[sorted(take)].assign(group=group))
    key = pd.concat(picked, ignore_index=True)
    key["kind"] = np.where(key["group"].str.startswith("edit"), "edited", "real")
    key.insert(0, "id", [f"img{k:03d}" for k in rng.permutation(len(key)) + 1])  # ids carry no group
    assert key["patient_id"].is_unique and len(key) == sum(n for _, n in GROUPS)
    READER.mkdir(parents=True, exist_ok=True)
    key.sort_values("id").to_csv(READER / "key.csv", index=False)
    (READER / "originals.txt").write_text("\n".join(key.loc[key["kind"] == "real", "path"]) + "\n")
    digest = hashlib.sha256((READER / "key.csv").read_bytes()).hexdigest()
    print(key.groupby("group").size().to_string(), f"\n{len(key)} images, {key['patient_id'].nunique()} patients; "
          f"key.csv SHA-256 {digest}", flush=True)


def page(args: argparse.Namespace) -> None:
    """The rating page: one image at a time, in a random order drawn for each reader, three questions,
    answers kept in the browser and exported as a CSV. Every image is re-encoded the same way (8-bit
    grayscale PNG, no metadata), so the files cannot tell a real film from an edit; the page holds
    only the random ids, never a file name, group or label."""
    key = pd.read_csv(READER / "key.csv")
    images = {}
    for row in key.itertuples():
        buffer = io.BytesIO()
        Image.open(ROOT / row.path).convert("L").save(buffer, format="PNG")
        images[row.id] = base64.b64encode(buffer.getvalue()).decode()
    html = (TEMPLATE.replace("__IMAGES__", json.dumps(images))
            .replace("__VERSION__", hashlib.sha256((READER / "key.csv").read_bytes()).hexdigest()[:12]))
    (READER / "rating_page.html").write_text(html, encoding="utf-8")
    print(f"{len(images)} images -> {READER / 'rating_page.html'} ({len(html) / 1e6:.1f} MB)", flush=True)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chest X-ray reading</title>
<style>
  :root { --bg: #0e0e0e; --panel: #1b1b1b; --ink: #f2f2f2; --muted: #a0a0a0; --accent: #2a78d6; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.4 system-ui, sans-serif; }
  main { display: flex; flex-wrap: wrap; gap: 24px; padding: 16px; justify-content: center; align-items: flex-start; }
  #xray { max-width: min(92vw, 820px); max-height: 88vh; background: #000; }
  .panel { background: var(--panel); padding: 20px; border-radius: 8px; width: 320px; max-width: 92vw; }
  h1 { font-size: 20px; margin: 0 0 12px; } p { color: var(--muted); }
  .q { margin: 18px 0; } .q b { display: block; margin-bottom: 8px; }
  button { font: inherit; padding: 10px 16px; margin: 0 8px 8px 0; border-radius: 6px; border: 1px solid #444;
           background: #262626; color: var(--ink); cursor: pointer; min-width: 92px; }
  button.on { background: var(--accent); border-color: var(--accent); }
  button:disabled { opacity: 0.4; cursor: default; }
  input { font: inherit; padding: 8px; width: 100%; margin: 8px 0 12px; border-radius: 6px; border: 1px solid #444;
          background: #262626; color: var(--ink); }
  .progress { color: var(--muted); margin-bottom: 8px; }
</style></head>
<body><main id="app"></main>
<script>
const IMAGES = __IMAGES__;
const VERSION = "__VERSION__";
const QUESTIONS = [["real_or_edited", "Is this image real or edited?", ["real", "edited"]],
                   ["cardiomegaly", "Cardiomegaly?", ["yes", "no"]],
                   ["pleural_effusion", "Pleural effusion?", ["yes", "no"]]];
const app = document.getElementById("app");
let state = null;

function load(reader) {  // progress is kept in this browser, so a reader can stop and come back
  try { return JSON.parse(localStorage.getItem("cxr-reading:" + VERSION + ":" + reader)); } catch (e) { return null; }
}
function save() {
  try { localStorage.setItem("cxr-reading:" + VERSION + ":" + state.reader, JSON.stringify(state)); } catch (e) {}
}
function shuffle(ids) {  // a new random order for each reader
  const a = ids.slice(), r = new Uint32Array(a.length);
  crypto.getRandomValues(r);
  for (let i = a.length - 1; i > 0; i--) { const j = r[i] % (i + 1); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function start() {
  app.innerHTML = `<div class="panel"><h1>Chest X-ray reading</h1>
    <p>You will see ${Object.keys(IMAGES).length} chest X-rays, one at a time. Some are real, some were edited by a
    computer program. For each image, answer three questions. There is no time limit; you can stop and
    come back later in the same browser. At the end, download your answers and send the file back.</p>
    <b>Your initials</b><input id="reader" autocomplete="off"><button id="go">Start</button></div>`;
  document.getElementById("go").onclick = () => {
    const reader = document.getElementById("reader").value.trim();
    if (!reader) return;
    state = load(reader) || {reader, order: shuffle(Object.keys(IMAGES)), answers: {}, position: 0};
    save(); show();
  };
}
function csv() {
  const rows = [["reader", "image_id", "position", ...QUESTIONS.map(q => q[0]), "answered_at", "page_version"]];
  const clean = s => String(s).replace(/[,"\\n\\r]/g, " ");
  state.order.forEach((id, k) => { const a = state.answers[id]; if (a) rows.push([clean(state.reader), id, k + 1,
    ...QUESTIONS.map(q => a[q[0]]), a.at, VERSION]); });
  return rows.map(r => r.join(",")).join("\\n") + "\\n";
}
function download() {
  const url = URL.createObjectURL(new Blob([csv()], {type: "text/csv"}));
  const link = document.createElement("a");
  link.href = url; link.download = "reading_" + state.reader.replace(/[^A-Za-z0-9]/g, "") + ".csv"; link.click();
  URL.revokeObjectURL(url);
}
function show() {
  const n = state.order.length;
  if (state.position >= n) {
    app.innerHTML = `<div class="panel"><h1>Done, thank you</h1><p>All ${n} images are answered.</p>
      <button id="dl" class="on">Download answers (CSV)</button></div>`;
    document.getElementById("dl").onclick = download; return;
  }
  const id = state.order[state.position], answer = Object.assign({}, state.answers[id] || {});
  app.innerHTML = `<img id="xray" alt="chest X-ray" src="data:image/png;base64,${IMAGES[id]}">
    <div class="panel"><div class="progress">Image ${state.position + 1} of ${n}</div>
    ${QUESTIONS.map(([k, text, options]) => `<div class="q"><b>${text}</b>${options.map(o =>
      `<button data-q="${k}" data-a="${o}">${o[0].toUpperCase() + o.slice(1)}</button>`).join("")}</div>`).join("")}
    <button id="next" disabled>Next</button>
    <p><button id="dl">Download answers so far</button></p></div>`;
  const next = document.getElementById("next");
  const refresh = () => {
    app.querySelectorAll("button[data-q]").forEach(b => b.classList.toggle("on", answer[b.dataset.q] === b.dataset.a));
    next.disabled = !QUESTIONS.every(q => answer[q[0]]);
  };
  app.querySelectorAll("button[data-q]").forEach(b => b.onclick = () => { answer[b.dataset.q] = b.dataset.a; refresh(); });
  next.onclick = () => { answer.at = new Date().toISOString(); state.answers[id] = answer; state.position += 1; save(); show(); };
  document.getElementById("dl").onclick = download;
  refresh();
}
start();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select", help="Draw the 100 images (fixed seed) -> outputs/reader/key.csv").set_defaults(func=select)
    sub.add_parser("page", help="Build the blinded rating page -> outputs/reader/rating_page.html").set_defaults(func=page)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
