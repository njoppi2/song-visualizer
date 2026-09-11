"""Render a controlled beat-clock comparison with independent percussion lanes.

Both variants have the same audio, percussion, layout and frame rate. Only the
beat schedule changes. Original analysis/reduced caches are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.build_review import cut_audio, fingerprint, run
from songviz.percussion import extract_component_attacks
from songviz.render import RenderConfig, _render_mp4_with_visualizer
from songviz.ingest import sha256_file


class PulseVisualizer:
    """Explicit beat indicator and component lanes on a common absolute clock."""
    COLORS={"beat":"#e3eff7","kick":"#67c6ff","snare":"#ff938c","hh":"#f5d76c"}
    LABELS={"beat":"PULSE","kick":"KICK","snare":"SNARE / CLAP","hh":"HI-HAT"}

    def __init__(self, beats, events, start, end, mode):
        self.start,self.end,self.mode=start,end,mode
        self.series={"beat":[{"t":t,"velocity":1.} for t in beats],
                     **{name:[h for h in events["hits"] if h["component"]==name] for name in ["kick","snare","hh"]}}
        self.times={name:np.array([h["t"] for h in hits]) for name,hits in self.series.items()}
        self.font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
        self.small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
        self.static=Image.new("RGB",(960,540),"#101a24")
        draw=ImageDraw.Draw(self.static)
        draw.text((28,18),"RHYTHM CHECK   /   "+mode,font=self.font,fill="white")
        draw.text((28,45),"Same original audio + same percussion in both versions. Only the pulse clock changes.",font=self.small,fill="#a7bac7")
        for i,name in enumerate(self.series):
            y=111+i*93
            draw.text((100,y-17),self.LABELS[name],font=self.font,fill=self.COLORS[name])
            draw.line((288,y+4,928,y+4),fill="#344957",width=1)
            for hit in self.series[name]:
                if start<=hit["t"]<end:
                    x=self.x(hit["t"])
                    height=20 if name=="beat" else max(4,round(25*hit["velocity"]))
                    draw.line((x,y-height,x,y+height),fill=self.COLORS[name],width=2)
        draw.text((28,485),"Drum lanes: prominent detected attacks, not beat positions. Weaker candidates are omitted.",font=self.small,fill="#a7bac7")
        draw.text((28,510),"Pulse continues through quiet passages; a pulse is not a claim that a drum is playing.",font=self.small,fill="#a7bac7")

    def x(self,t):
        return round(288+(t-self.start)/(self.end-self.start)*640)

    def frame_rgb24(self,t):
        absolute=self.start+t
        img=self.static.copy();draw=ImageDraw.Draw(img)
        draw.text((720,18),f"Song {absolute:06.2f}s",font=self.font,fill="#e3eff7")
        for i,name in enumerate(self.series):
            y=111+i*93
            idx=int(np.searchsorted(self.times[name],absolute,side="right"))-1
            strength=0.
            if idx>=0:
                hit=self.series[name][idx];age=absolute-hit["t"]
                if age<.2:strength=np.exp(-age/.055)*hit["velocity"]
            radius=17+round(15*strength)
            color=self.COLORS[name] if strength>.08 else "#263b4a"
            draw.ellipse((58-radius,y-radius,58+radius,y+radius),fill=color)
            x=self.x(absolute);draw.line((x,y-30,x,y+30),fill="white",width=2)
        return img.tobytes()


def build(evidence: Path,out: Path):
    timing=json.loads((evidence/"timing.json").read_text())
    if timing["candidate"]["status"] != "candidate":
        raise ValueError("Evidence does not contain a supported pulse candidate")
    out.mkdir(parents=True,exist_ok=False)
    inputs=out/"inputs";inputs.mkdir()
    spec=json.loads((ROOT/"benchmark/review_passages.json").read_text())
    song=ROOT/"songs"/spec["passages"][0]["song"]
    cached=ROOT/"outputs"/song.stem
    components={};component_fps=[];component_sr=None
    for name in ["kick","snare","hh"]:
        path=cached/"stems/drumsep"/(name+".wav")
        y,sr=sf.read(path,dtype="float32",always_2d=True)
        if component_sr is not None and sr != component_sr:
            raise ValueError("Component sample rates must match")
        component_sr=sr
        # Preserve stereo energy before the mono detector's RMS operation.
        components[name]=np.sqrt(np.mean(y.astype(float)**2,axis=1)).astype(np.float32)
        component_fps.append(fingerprint(path))
    events=extract_component_attacks(components,sr)
    (out/"percussion-candidates.json").write_text(json.dumps(events,indent=2)+"\n")
    for path in evidence.glob("*.png"):shutil.copy2(path,out/path.name)
    shutil.copy2(evidence/"timing.json",out/"timing.json")
    for rel in ["songviz/percussion.py","songviz/beat_grid.py","experiments/build_rhythm_review.py",
                "experiments/inspect_rhythm.py","experiments/templates/rhythm_review.html",
                "benchmark/feedback/restart-02.json","benchmark/review_passages.json"]:
        dest=inputs/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    manifest={"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),
              "git_head":run("git","rev-parse","HEAD"),"source_audio":fingerprint(song),
              "components":component_fps,"feedback_source":fingerprint(ROOT/"benchmark/feedback/restart-02.json"),
              "comparison":"same audio, layout, extracted percussion, frame rate; only beat times change",
              "timing":fingerprint(evidence/"timing.json"),"percussion":fingerprint(out/"percussion-candidates.json"),
              "frame_rate":60,"maximum_frame_quantization_ms":1000/60,
              "code_and_feedback_snapshots":[fingerprint(p) for p in sorted(inputs.rglob("*")) if p.is_file()],
              "production_pipeline_changed":False,"user_review":"pending","clips":[]}
    cfg=RenderConfig(width=960,height=540,fps=60,audio_codec="aac",audio_bitrate="192k")
    cards=[]
    for passage in spec["passages"]:
        ident=passage["id"];start,end=passage["start_s"],passage["end_s"]
        dest=out/ident;dest.mkdir();mix,mix_sr=cut_audio(song,start,end)
        sf.write(dest/"original.wav",mix,mix_sr,subtype="PCM_24")
        for variant,key,title in [("cached","baseline","CACHED PULSE"),("regular","candidate","REGULAR PULSE CANDIDATE")]:
            print(f"Render {ident}/{variant}",flush=True)
            visualizer=PulseVisualizer(timing[key]["beat_times_s"],events,start,end,title)
            Image.frombytes("RGB",(960,540),visualizer.frame_rgb24(0)).save(dest/(variant+".png"))
            _render_mp4_with_visualizer(audio_path=dest/"original.wav",out_path=dest/(variant+".mp4"),
                                       cfg=cfg,duration_s=end-start,visualizer=visualizer)
        # Same excerpt's source component audio gives the lanes a concrete meaning.
        for name in components:
            y,rate=cut_audio(cached/"stems/drumsep"/(name+".wav"),start,end)
            sf.write(dest/(name+".wav"),y,rate,subtype="PCM_24")
        manifest["clips"].append({**passage,"outputs":[fingerprint(p) for p in sorted(dest.iterdir())]})
        cards.append(f'''<section data-id="{ident}" data-start="{start}"><h2>{html.escape(passage['title'])} · {start}–{end}s</h2>
<div class="choices"><button data-variant="cached" aria-pressed="true">Cached pulse</button><button data-variant="regular" aria-pressed="false">Regular pulse candidate</button><button class="restart">Restart clip</button></div>
<p class="playing">Showing: cached pulse</p><video controls preload="metadata" poster="{ident}/cached.png" src="{ident}/cached.mp4"></video>
<p>Which pulse is easier to clap along with? Are the kick and snare/clap indicators recognizable now? Drum lanes show conservative prominent candidates; missing or misclassified hits remain possible.</p>
<details><summary>Hear the separated parts</summary>Kick<audio controls src="{ident}/kick.wav" preload="none"></audio>Snare/clap<audio controls src="{ident}/snare.wav" preload="none"></audio>Hi-hat<audio controls src="{ident}/hh.wav" preload="none"></audio></details>
<label>Preferred pulse <select class="preference"><option value="unreviewed">Not reviewed</option><option value="regular">Regular candidate</option><option value="cached">Cached</option><option value="neither">Neither</option><option value="unsure">Unsure</option></select></label>
<button class="stamp">Mark current moment</button><input class="time" aria-label="Original-song seconds" type="number" step="0.01" value="{start}">
<textarea aria-label="Observation" placeholder="Describe any remaining timing problem or unclear drum indicator."></textarea></section>''')
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    template=(ROOT/"experiments/templates/rhythm_review.html").read_text()
    (out/"index.html").write_text(template.replace("{{CARDS}}","\n".join(cards)).replace("{{MANIFEST_SHA}}",sha256_file(out/"manifest.json")))
    print(f"Ready: {out/'index.html'}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();build(args.evidence.resolve(),args.out.resolve())
