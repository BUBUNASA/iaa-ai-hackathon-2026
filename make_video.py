"""Build the 5-minute video: slides (matplotlib) + narration (Windows SAPI voice via pyttsx3) -> MP4 (PyAV: H.264 + AAC).

Narration comes from VIDEO_SCRIPT.md: each **[Screen]** cue starts a slide and the text under it is spoken over that slide, so the
script stays the single source of truth. Charts are drawn from the repo's result files (p2_blend_results.csv,
outputs/p2_reference_premiums.csv, the raw developed-market policies).

    python make_video.py --slides   # render slides only (outputs/video/slides/*.png) for review
    python make_video.py            # slides + narration + outputs/video/NazaliBupa_IAA2026.mp4
"""
from __future__ import annotations

import argparse
import re
import textwrap
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "video"
SLIDES, AUDIO = OUT / "slides", OUT / "audio"
FPS, SAPI_RATE, PAD_S, MAX_S = 5, 0, 1.0, 298  # SAPI rate: -10..10, 0 = normal (~150 wpm)
NAVY, TEAL, GREY, RED, GREEN, AMBER, LIGHT = "#1f3b5c", "#2a9d8f", "#6c757d", "#c0392b", "#2e8b57", "#c47f17", "#f3f6f9"
W, H, DPI = 19.2, 10.8, 100
N_SLIDES = 10


# ---------------------------------------------------------------- script
def parse_script() -> list[dict]:
    chunks: list[dict] = []
    for line in (ROOT / "VIDEO_SCRIPT.md").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("**[Screen]**"):
            chunks.append({"cue": s.replace("**[Screen]**", "").strip(), "text": []})
        elif s and chunks and not s.startswith(("#", "---")):
            chunks[-1]["text"].append(s)
    for c in chunks:
        c["text"] = re.sub(r"[*_`]", "", " ".join(c["text"]).replace(" - ", ", "))
    return chunks


# ---------------------------------------------------------------- slide helpers
def base(title: str, n: int, sub: str | None = None):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.add_artist(plt.Rectangle((0, 0.885), 1, 0.115, transform=fig.transFigure, color=NAVY))
    fig.text(0.04, 0.942, title, color="white", fontsize=34, weight="bold", va="center")
    if sub:
        fig.text(0.04, 0.84, sub, fontsize=21, color=GREY, va="center", style="italic")
    fig.text(0.04, 0.025, "Team NazaliBupa  |  IAA AI-Assisted Hackathon 2026", fontsize=13, color=GREY)
    fig.text(0.96, 0.025, f"{n} / {N_SLIDES}", fontsize=13, color=GREY, ha="right")
    return fig


def bullets(fig, items: list[str], x: float, y: float, size: int = 22, chars: int = 44, color: str = "black") -> float:
    for t in items:
        s = textwrap.fill(t, chars)
        fig.text(x, y, "\u25a0", fontsize=size * 0.6, va="top", color=TEAL)
        fig.text(x + 0.022, y + 0.004, s, fontsize=size, va="top", color=color)
        y -= (s.count("\n") + 1) * size * 1.95 / 1080 + 0.028
    return y


def style(ax) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=16)


def table(fig, rect, header, rows, colw, size=18, height=2.8, colors=None):
    ax = fig.add_axes(rect)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, colWidths=colw, loc="upper center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    t.scale(1, height)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor(LIGHT)
        if colors and r > 0 and (r - 1, c) in colors:
            cell.get_text().set_color(colors[(r - 1, c)])
            cell.get_text().set_weight("bold")
    return t


# ---------------------------------------------------------------- slides
def s_title(n):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(NAVY)
    fig.text(0.5, 0.68, "Pricing a known market and a new one", ha="center", fontsize=52, color="white", weight="bold")
    fig.text(0.5, 0.585, "with AI, and with evidence", ha="center", fontsize=36, color="#cfe3f0")
    fig.text(0.5, 0.43, "Expected 2025 claim cost (the risk premium) for motor, home buildings and home contents",
             ha="center", fontsize=23, color="white")
    fig.text(0.5, 0.36, "Part 1: established market, 4 years of data      Part 2: new market, 2 years, 76% new customers",
             ha="center", fontsize=23, color="white")
    fig.text(0.5, 0.22, "One rule for every decision: adopt a change only if it wins on two separate holdout tests",
             ha="center", fontsize=25, color="#f4c56b", weight="bold")
    fig.text(0.5, 0.07, "Team NazaliBupa  |  IAA AI-Assisted Hackathon 2026", ha="center", fontsize=20, color="#cfe3f0")
    return fig


def s_leak(n):
    fig = base("Trap 1: a field that contains the answer", n, "customer_service_contacts: people call because they have claimed")
    ax = fig.add_axes([0.07, 0.13, 0.38, 0.6])
    vals = [0.06, 0.64]
    bars = ax.bar(["Without the field\n(honest model)", "With the field\n(leak)"], vals, color=[TEAL, RED], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=28, weight="bold")
    ax.set_ylabel("Validation score (Tweedie D2)", fontsize=18)
    ax.set_ylim(0, 0.75)
    style(ax)
    bullets(fig, ["Correlation 0.75 with the same policy's claim count",
                  "Zero contacts means essentially zero claims",
                  "2025 test data: a flat 0.40 on every line, against 0.5-0.75 in training",
                  "Removed, and an automated test stops it coming back",
                  "The same leak appears again in the new market"], 0.53, 0.74)
    return fig


def s_exposure(n):
    fig = base("Trap 2: policy length recorded in the wrong unit", n, "About 0.4% of training rows: 365 (days) or 1.014 (a 360-day year) instead of a fraction of a year")
    lines, days, b360, pct = ["Motor", "Buildings", "Contents"], [], [], []
    for ln in ("motor", "buildings", "contents"):
        e = pd.read_parquet(ROOT / "data" / "part2" / "developed" / f"policies_{ln}.parquet", columns=["exposure"])["exposure"]
        days.append(int((e > 2).sum()))
        b360.append(int(((e > 1.0001) & (e <= 2)).sum()))
        pct.append(float((e[e > 1.0001] - 1).sum() / e.clip(upper=1).sum()))
    ax = fig.add_axes([0.07, 0.14, 0.4, 0.58])
    x = np.arange(3)
    ax.bar(x - 0.2, days, 0.4, color=RED, label="Recorded in days (365, 292, ...)")
    ax.bar(x + 0.2, b360, 0.4, color=AMBER, label="360-day year (1.014)")
    for i in range(3):
        ax.text(x[i] - 0.2, days[i] + 40, f"{days[i]:,}", ha="center", fontsize=16)
        ax.text(x[i] + 0.2, b360[i] + 40, f"{b360[i]:,}", ha="center", fontsize=16)
        ax.text(x[i], max(days[i], b360[i]) + 420, f"+{pct[i]:.0%} phantom\ncover", ha="center", fontsize=17, color=RED, weight="bold")
    ax.set_xticks(x, lines, fontsize=18)
    ax.set_ylabel("Training rows affected", fontsize=18)
    ax.set_ylim(0, max(b360) * 1.6)
    ax.legend(fontsize=16, frameon=False, loc="upper right")
    style(ax)
    bullets(fig, ["Found by asking why a +0.02 holdout gain showed as only +0.0017 on the leaderboard",
                  "Fix: rebuild policy length from the policy dates",
                  "Guardrail: any length above one year, or inconsistent with the dates, blocks the submission",
                  "Caught on first load in the new market (452 rows)"], 0.53, 0.74)
    return fig


def s_part1(n):
    fig = base("Part 1: four validated steps", n, "Each step had to win on both the 2023 and the 2024 holdout")
    labels = ["v1\nbaseline", "v2\n+ claims still\nto be reported", "v3\nfrequency\nx severity", "v4\n+ cost trend",
              "v5\n+ per-peril\nmodels", "v6\n+ pricing\nfields"]
    pub = [0.0859, 0.0955, 0.1039, 0.1119, 0.1130, 0.1147]
    prv = [0.1025, 0.1101, 0.1160, 0.1226, 0.1265, 0.1260]
    ax = fig.add_axes([0.07, 0.2, 0.58, 0.56])
    x = np.arange(6)
    ax.bar(x - 0.2, pub, 0.4, color="#9db9d5", label="Public leaderboard")
    ax.bar(x + 0.2, prv, 0.4, color=NAVY, label="Final (private) leaderboard")
    for i in range(6):
        ax.text(x[i] - 0.2, pub[i] + 0.0012, f"{pub[i]:.3f}", ha="center", fontsize=13)
        ax.text(x[i] + 0.2, prv[i] + 0.0012, f"{prv[i]:.3f}", ha="center", fontsize=13, weight="bold")
    ax.set_xticks(x, labels, fontsize=14)
    ax.set_ylim(0.07, 0.138)
    ax.set_ylabel("Score (Tweedie D2)", fontsize=18)
    ax.legend(fontsize=16, frameon=False, loc="upper left")
    style(ax)
    bullets(fig, ["Biggest gain: claims valued at end-2024 are still being reported",
                  "Frequency x severity beat a single model in 6 of 6 tests",
                  "Cost trend estimated on earlier years only",
                  "Final: private 0.126, rank 43",
                  "v5 beat v6 privately: the pricing gain was smaller than it looked"], 0.69, 0.74, size=20, chars=30)
    return fig


def s_changed(n):
    fig = base("Part 2: what changed in the new market", n)
    rows = [["Claims history", "2021-24: ~163,000 claims", "2023-24: 9,258 claims (18x fewer)"],
            ["2025 customers", "15% new business", "76% new business; 50% aged 17-34"],
            ["Motor", "0.14 claims a year, ~3,800 each", "0.18 claims a year, ~2,200 each"],
            ["Buildings", "cost per year 432-449", "514, then 307 (fires = 45% of cost)"],
            ["Contents", "cost per year ~117", "87-91"],
            ["Data fields", "exact ages, credit, telematics", "age bands, free-text jobs, gender"],
            ["Same defects", "-", "policy length in days; the contacts leak"]]
    table(fig, [0.05, 0.2, 0.9, 0.6], ["", "Home market (Part 1)", "New market (Part 2)"], rows, [0.2, 0.36, 0.44],
          size=20, height=3.2, colors={(i, 2): NAVY for i in range(7)})
    fig.text(0.5, 0.12, "A model trained on the new market alone cannot learn much: its claim-size model stops after one step",
             ha="center", fontsize=22, color=AMBER, weight="bold")
    return fig


def s_transfer(n):
    fig = base("Part 2: borrow structure from the home market", n,
               "The home market supplies the risk pattern; the new market's own data corrects it")
    df = pd.read_csv(ROOT / "p2_blend_results.csv")
    models = [("EM", "New market only", RED), ("DEV_CAL", "Home model, recalibrated", "#9db9d5"),
              ("POOL", "Pooled, both markets", "#5d8bb8"), ("OFFSET", "Home model + new-market correction", NAVY),
              ("DEV+POOL+OFFSET", "Blend of the three (chosen)", TEAL)]
    lines = ["motor", "buildings", "contents"]
    x = np.arange(3)
    for k, (scheme, title) in enumerate([("time", "Test 1: predict 2024 from 2023"), ("group", "Test 2: 30% of policies held out")]):
        ax = fig.add_axes([0.07 + k * 0.47, 0.17, 0.41, 0.5])
        for j, (m, lab, col) in enumerate(models):
            vals = [df[(df.scheme == scheme) & (df.line == ln) & (df.model == m)]["D2"].iloc[0] for ln in lines]
            ax.bar(x + (j - 2) * 0.16, vals, 0.16, color=col, label=lab if k == 0 else None)
        ax.set_xticks(x, ["Motor", "Buildings", "Contents"], fontsize=18)
        ax.set_title(title, fontsize=21, pad=12)
        ax.set_ylabel("Score (Tweedie D2)", fontsize=16)
        ax.set_ylim(0, 0.13)
        style(ax)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.8), ncol=5, fontsize=15, frameon=False)
    fig.text(0.5, 0.075, "Borrowing from the home market beats the new-market-only model on every line, in both tests",
             ha="center", fontsize=22, color=GREEN, weight="bold")
    return fig


def s_rejected(n):
    fig = base("Part 2: four more ideas, same rule", n)
    rows = [["New-business loading", "helped one test, hurt the other; the factor flipped", "Rejected"],
            ["Cost trend from the home market", "lowered the score on every line (2024 test)", "Rejected"],
            ["Per-peril models (Part 1's best step)", "too few claims per peril: learned noise", "Rejected"],
            ["Insurer's own pricing fields", "small win on both tests for motor and buildings", "Adopted (v2)"]]
    table(fig, [0.05, 0.3, 0.9, 0.5], ["Idea", "What happened on the two tests", "Decision"], rows, [0.33, 0.5, 0.17],
          size=21, height=3.6, colors={(0, 2): RED, (1, 2): RED, (2, 2): RED, (3, 2): GREEN})
    fig.text(0.5, 0.2, "Finer splits help when data is rich, and hurt when it is thin", ha="center", fontsize=26, color=NAVY, weight="bold")
    return fig


def s_premium(n):
    fig = base("The business answer: the broker book is under-priced", n, "Office premium = expected claims + expenses + commission + tax + reinsurance + profit")
    ref = pd.read_csv(ROOT / "outputs" / "p2_reference_premiums.csv", dtype={"row_id": str})
    r = ref.iloc[0]
    parts = [("Claims", r.risk_premium, NAVY), ("Fixed exp.", r.fixed_expense, "#5d8bb8"),
             ("Variable exp.", r.variable_expense, "#9db9d5"), ("Commission", r.commission, AMBER),
             ("Premium tax", r.premium_tax, GREY), ("Profit", r.target_profit_margin, GREEN)]
    ax = fig.add_axes([0.07, 0.14, 0.36, 0.6])
    bottom = 0.0
    for lab, v, col in parts:
        ax.bar(0, v, 0.75, bottom=bottom, color=col)
        if v > 60:
            ax.text(0, bottom + v / 2, f"{lab}  {v:,.0f}", ha="center", va="center", fontsize=14, color="white", weight="bold")
        bottom += v
    ax.bar(1, r.written_premium, 0.75, color=RED)
    ax.text(0, bottom + 40, f"Needed {bottom:,.0f}", ha="center", fontsize=18, weight="bold")
    ax.text(1, r.written_premium + 40, f"Charged {r.written_premium:,.0f}", ha="center", fontsize=18, weight="bold", color=RED)
    ax.set_xticks([0, 1], [f"Technical price\n(policy {r.row_id}, {r.line})", "Price charged today"], fontsize=15)
    ax.set_ylabel("Currency units", fontsize=16)
    ax.set_ylim(0, bottom * 1.15)
    style(ax)
    ratio = ref.groupby("line")["office/charged"].mean().reindex(["motor", "contents", "buildings"])
    ax2 = fig.add_axes([0.53, 0.3, 0.4, 0.44])
    b = ax2.bar(["Motor", "Contents", "Buildings"], ratio.values, color=[NAVY, TEAL, AMBER], width=0.55)
    for bb, v in zip(b, ratio.values):
        ax2.text(bb.get_x() + bb.get_width() / 2, v + 0.08, f"{np.floor(v * 10 + 0.5) / 10:.1f}x", ha="center", fontsize=24, weight="bold")
    ax2.axhline(1, color=RED, ls="--", lw=2)
    ax2.text(2.35, 1.08, "priced right", color=RED, fontsize=15, ha="right")
    ax2.set_title("Needed price / price charged (20 reference policies)", fontsize=19)
    ax2.set_ylim(0, 4.8)
    style(ax2)
    fig.text(0.53, 0.2, textwrap.fill("v2 uses today's prices and forecasts slightly better, but a repricing built on it would be "
                                      "anchored to the price it replaces: v1 is recommended for pricing decisions.", 58),
             fontsize=18, va="top", color=NAVY)
    return fig


def s_ai(n):
    fig = base("How we used AI, and where we overruled it", n)
    rows = [["Data checks", "profiled the data; traced the score gap to days", "approved every exclusion and fix"],
            ["Modelling", "built models, holdout tests, 4 transfer methods", "set the two-test rule; chose on the worst case"],
            ["Guardrails", "wrote the checks and 31 automated tests", "fixed wrong AI tests on evidence, never loosened"],
            ["AI's own ideas", "3 ideas that looked convincing", "cut back or rejected once validated"],
            ["Judgement", "laid out the trade-off", "recommend v1 for pricing (tariff circularity)"]]
    table(fig, [0.05, 0.22, 0.9, 0.58], ["", "What AI did", "What people did"], rows, [0.16, 0.42, 0.42], size=20, height=3.3)
    fig.text(0.5, 0.13, "Every decision is backed by a result file in the repository that a person reviewed", ha="center",
             fontsize=22, color=NAVY, weight="bold")
    return fig


def s_close(n):
    fig = base("Results and remaining uncertainty", n)
    for i, (head, body, col) in enumerate([("Part 1: home market", "rank 43\nprivate score 0.126", GREY),
                                           ("Part 2: new market", "rank 6 of 40\nup 18 places on hidden data\nprivate score 0.137", GREEN)]):
        x0 = 0.06 + i * 0.47
        fig.add_artist(plt.Rectangle((x0, 0.5), 0.41, 0.3, transform=fig.transFigure, facecolor=LIGHT, edgecolor=col, lw=3))
        fig.text(x0 + 0.205, 0.755, head, ha="center", va="top", fontsize=26, weight="bold", color=NAVY)
        fig.text(x0 + 0.205, 0.68, body, ha="center", va="top", fontsize=24, color=col, weight="bold", linespacing=1.5)
    bullets(fig, ["New-market buildings level rests on a few fires (70 claims = 45% of cost)",
                  "Three quarters of 2025 customers are new: structure is borrowed from the home market",
                  "Everything is reproducible from the command list in README.md, with 31 automated tests"],
            0.07, 0.42, size=21, chars=90)
    return fig


SLIDE_FUNCS = [s_title, s_leak, s_exposure, s_part1, s_changed, s_transfer, s_rejected, s_premium, s_ai, s_close]


def render_slides() -> list[Path]:
    SLIDES.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(SLIDE_FUNCS, start=1):
        fig = f(i)
        p = SLIDES / f"{i:02d}.png"
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- narration and encoding
def synthesise(chunks: list[dict]) -> list[Path]:
    """Windows SAPI directly (pyttsx3 hangs when one engine writes several files). Synchronous, one WAV per slide."""
    import win32com.client

    AUDIO.mkdir(parents=True, exist_ok=True)
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    for tok in voice.GetVoices():
        if "Zira" in tok.GetDescription():
            voice.Voice = tok
    voice.Rate = SAPI_RATE
    paths = []
    for i, c in enumerate(chunks, start=1):
        p = AUDIO / f"{i:02d}.wav"
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        fmt = win32com.client.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = 22  # SAFT22kHz16BitMono
        stream.Format = fmt
        stream.Open(str(p), 3)  # SSFMCreateForWrite
        voice.AudioOutputStream = stream
        voice.Speak(c["text"])  # synchronous
        stream.Close()
        paths.append(p)
        print(f"   narration {i:02d}: {len(read_wav(p)[0]) / 22050:.1f}s", flush=True)
    voice.AudioOutputStream = None
    return paths


def read_wav(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, f"{p}: expected 16-bit mono"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate()


def encode(slides: list[Path], wavs: list[Path], path: Path) -> float:
    import av
    from PIL import Image

    audio, rates = zip(*(read_wav(p) for p in wavs))
    sr = rates[0]
    assert all(r == sr for r in rates)
    pad = np.zeros(int(PAD_S * sr), dtype=np.int16)
    audio = [np.concatenate([a, pad]) for a in audio]
    total = sum(len(a) for a in audio) / sr
    assert total <= MAX_S, f"narration {total:.0f}s exceeds {MAX_S}s: raise SAPI_RATE"

    out = av.open(str(path), "w")
    vs = out.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = int(W * DPI), int(H * DPI), "yuv420p"
    vs.options = {"crf": "20", "preset": "medium", "tune": "stillimage"}
    aus = out.add_stream("aac", rate=sr, layout="mono")
    t0, a_pts = 0.0, 0
    for img, a in zip(slides, audio):
        t1 = t0 + len(a) / sr
        frame = av.VideoFrame.from_ndarray(np.asarray(Image.open(img).convert("RGB")), format="rgb24")
        for _ in range(round(t1 * FPS) - round(t0 * FPS)):
            for pkt in vs.encode(frame):
                out.mux(pkt)
        for s in range(0, len(a), 1024):
            chunk = np.ascontiguousarray(a[s:s + 1024]).reshape(1, -1)
            af = av.AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
            af.sample_rate, af.pts = sr, a_pts
            a_pts += chunk.shape[1]
            for pkt in aus.encode(af):
                out.mux(pkt)
        t0 = t1
    for st in (vs, aus):
        for pkt in st.encode(None):
            out.mux(pkt)
    out.close()
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", action="store_true", help="render slides only")
    a = ap.parse_args()
    chunks = parse_script()
    assert len(chunks) == N_SLIDES, f"VIDEO_SCRIPT.md has {len(chunks)} [Screen] cues, expected {N_SLIDES}"
    slides = render_slides()
    words = sum(len(c["text"].split()) for c in chunks)
    print(f"rendered {len(slides)} slides; narration {words} words", flush=True)
    if not a.slides:
        wavs = synthesise(chunks)
        mp4 = OUT / "NazaliBupa_IAA2026.mp4"
        secs = encode(slides, wavs, mp4)
        per = [round(len(read_wav(p)[0]) / read_wav(p)[1] + PAD_S, 1) for p in wavs]
        print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB), {secs:.0f}s; per slide {per}", flush=True)
