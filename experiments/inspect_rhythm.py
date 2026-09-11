"""Inspect cached beats against source percussion; save measured visual evidence.

This experiment fits the prominent snare repetition on 30–110s and inspects
the user-reviewed passages outside that fitting interval. It makes a constant
tempo / two-pulses-per-snare hypothesis explicit. No production caches change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from songviz.beat_grid import prominent_attacks, fit_regular_pulse
from experiments.build_review import fingerprint, run, rms_curve


def inspect(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    song = "Gorillaz - Feel Good Inc (featuring De La Soul)"
    cached = ROOT / "outputs" / song
    analysis_path = cached / "analysis/analysis.json"
    analysis = json.loads(analysis_path.read_text())
    drumsep = cached / "stems/drumsep"
    snare, sr = sf.read(drumsep / "snare.wav", always_2d=True)
    anchors = prominent_attacks(snare, sr)
    # Gate the extrapolated grid to the first prominent hi-hat evidence instead
    # of displaying an invented pulse throughout the unpitched opening.
    hats, hats_sr = sf.read(drumsep / "hh.wav", always_2d=True)
    hat_attacks = prominent_attacks(hats, hats_sr)
    first_evidence = float(hat_attacks[0])
    # Permit the inferred beat immediately before the measured first hat.
    grid = fit_regular_pulse(anchors, duration_s=analysis["meta"]["duration_s"],
                             fit_start_s=30, fit_end_s=110, active_start_s=max(0, first_evidence-.04))
    if grid["status"] != "candidate":
        raise RuntimeError(f"No supported regular-grid candidate: {grid}")
    beats = np.asarray(grid["beat_times_s"])
    old = np.asarray(analysis["beats"]["beat_times_s"])
    report = {"candidate": grid, "baseline": analysis["beats"], "first_percussion_evidence_s": first_evidence,
              "feedback": fingerprint(ROOT / "benchmark/feedback/restart-02.json"),
              "analysis": fingerprint(analysis_path), "snare": fingerprint(drumsep / "snare.wav"),
              "hh": fingerprint(drumsep / "hh.wav"),
              "fit_method": fingerprint(ROOT / "songviz/beat_grid.py"),
              "git_head": run("git", "rev-parse", "HEAD"), "passages": []}
    plt.style.use("dark_background")
    for start, end, slug in [(0,22,"opening"), (130,148,"transition"), (158,178,"return")]:
        old_clip = old[(old >= start)&(old < end)]
        proposed = beats[(beats >= start)&(beats < end)]
        ref = anchors[(anchors >= start)&(anchors < end)]
        intervals = np.diff(old_clip)
        metrics = {"start_s": start, "end_s": end,
                   "old_interval_min_ms": float(intervals.min()*1000),
                   "old_interval_max_ms": float(intervals.max()*1000),
                   "candidate_interval_ms": float(np.median(np.diff(proposed))*1000),
                   "prominent_snare_count": len(ref)}
        if len(ref):
            metrics["old_nearest_snare_median_ms"] = float(np.median(np.min(abs(ref[:,None]-old),axis=1))*1000)
            metrics["candidate_nearest_snare_median_ms"] = float(np.median(np.min(abs(ref[:,None]-beats),axis=1))*1000)
        report["passages"].append(metrics)
        fig, axes = plt.subplots(5,1,figsize=(14,8),sharex=True,gridspec_kw={"height_ratios":[1,1,1,.7,.7]})
        for ax,name,color in zip(axes[:3],["kick","snare","hh"],["#67c6ff","#ff938c","#f5d76c"]):
            y, rate = sf.read(drumsep / f"{name}.wav",always_2d=True)
            lo, hi = round(start*rate), round(end*rate)
            hop=max(1,round(rate*.004)); n=(hi-lo)//hop
            envelope=np.sqrt(np.mean(y[lo:lo+n*hop].reshape(n,hop,-1)**2,axis=(1,2)))
            times=start+np.arange(n)*hop/rate
            ax.plot(times,envelope,color=color,linewidth=.75)
            ax.set_ylabel(name+" RMS")
            for t in proposed: ax.axvline(t,color="white",alpha=.12,linewidth=.6)
            if name=="snare":
                for t in ref: ax.axvline(t,color="#80ffc0",alpha=.9,linewidth=.8)
        axes[3].eventplot(old_clip,colors="#ff6b81",linelengths=.8)
        axes[3].set_ylabel("Cached beats")
        axes[4].eventplot(proposed,colors="#80ffc0",linelengths=.8)
        axes[4].set_ylabel("Proposed pulse")
        for ax in axes:
            ax.set_xlim(start,end); ax.spines[["top","right"]].set_visible(False)
        axes[3].set_yticks([]);axes[4].set_yticks([])
        axes[-1].set_xlabel("Original-song time (seconds)")
        fig.suptitle(f"{slug.title()}: measured component envelopes and two timing hypotheses\n"
                     f"Cached ≈92.29 BPM with variable intervals; proposed {grid['tempo_bpm']:.2f} BPM constant",fontsize=14)
        fig.text(.5,.015,"Green lines on snare = independently detected prominent attacks. Component scales differ; separation can contain bleed.",ha="center",fontsize=10)
        fig.tight_layout(rect=(0,.04,1,.93));fig.savefig(out/f"{slug}-timing.png",dpi=140);plt.close(fig)
    # A close-up of the raw snare waveform plus spectrogram makes it possible
    # to check that inferred pulse markers coincide with actual sharp attacks.
    lo,hi=130.,134.;samples=snare[round(lo*sr):round(hi*sr)].mean(axis=1)
    fig,axes=plt.subplots(3,1,figsize=(14,7),sharex=True,gridspec_kw={"height_ratios":[1,1,.55]})
    axes[0].plot(lo+np.arange(len(samples))/sr,samples,color="#92c9ed",linewidth=.4)
    axes[0].set_ylabel("Snare samples")
    axes[1].specgram(samples,NFFT=512,Fs=sr,noverlap=384,xextent=(lo,hi),cmap="magma",vmin=-110,vmax=-35)
    axes[1].set_ylim(0,8000);axes[1].set_ylabel("Frequency (Hz)")
    for ax in axes[:2]:
        for t in beats[(beats>=lo)&(beats<hi)]:ax.axvline(t,color="#80ffc0",linewidth=1,alpha=.7)
    axes[2].eventplot([old[(old>=lo)&(old<hi)],beats[(beats>=lo)&(beats<hi)]],lineoffsets=[1,0],colors=["#ff6b81","#80ffc0"])
    axes[2].set_yticks([1,0],["Cached","Proposed"])
    axes[2].set_xlim(lo,hi);axes[2].set_xlabel("Original-song seconds")
    fig.suptitle("Transient close-up: source snare waveform + spectrum; green = proposed pulse, red = cached pulse")
    fig.tight_layout();fig.savefig(out/"snare-closeup.png",dpi=140);plt.close(fig)
    (out/"timing.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in grid.items() if not isinstance(v,list)},indent=2))
    print(json.dumps(report["passages"],indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();inspect(args.out.resolve())
