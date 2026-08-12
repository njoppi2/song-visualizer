import http.server, io, json, socket, subprocess, threading, time
from pathlib import Path
from queue import Empty, Queue
from urllib.parse import quote, unquote

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

OUT = Path("outputs")
SONGS = Path("songs")
SEC = {"A": "#3a7ca5", "B": "#2d9e6b", "C": "#c4732a", "D": "#8b4fa8"}
SEC_CYCLE = ["#3a7ca5", "#2d9e6b", "#c4732a", "#8b4fa8", "#a64f57", "#6f8f3a", "#4f6fa8"]
STEM = {"drums": "#e35cc8", "bass": "#f09840", "vocals": "#f0d860", "other": "#8be0c4"}
STRUCTURE_TOOL_COLORS = [
    "#ffffff", "#00c2c7", "#ffb703", "#fb7185", "#a3e635", "#c084fc", "#60a5fa"
]
PHRASE_CLUSTER_COLORS = [
    "#2d9e6b", "#c4732a", "#8b4fa8", "#3a7ca5", "#a64f57", "#6f8f3a",
    "#d9a441", "#5f8dd3", "#b95ba5", "#00a6a6", "#d46a6a", "#8aa64f",
]
_ML, _MR, _MT, _MB = 58, 18, 20, 35

# ---------------------------------------------------------------------------
# Local audio file server (needed so the HTML player can load audio via URL)
# ---------------------------------------------------------------------------
_audio_port: int | None = None
_wav_cache: dict[str, bytes] = {}  # absolute FLAC path → WAV bytes


def _audio_bytes_as_wav(path: str) -> bytes:
    """Return seekable WAV bytes for any audio file, cached.

    WAV files are returned as-is (already seekable).
    FLAC/MP3 are transcoded to WAV so browsers can seek by byte offset.
    """
    if path not in _wav_cache:
        import soundfile as sf
        p = Path(path)
        if p.suffix.lower() == ".wav":
            _wav_cache[path] = p.read_bytes()
        else:
            data, sr = sf.read(path, dtype="int16", always_2d=True)
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
            _wav_cache[path] = buf.getvalue()
    return _wav_cache[path]


def _ensure_audio_server() -> int:
    global _audio_port
    if _audio_port is not None:
        return _audio_port

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    root_dir = Path(".").absolute()

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root_dir), **kwargs)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def log_message(self, *_args):
            pass

        def do_GET(self):
            # Intercept audio files → serve as seekable WAV with Range support.
            # Browsers can't seek inside FLAC (no byte-offset table), and
            # SimpleHTTPRequestHandler doesn't implement Range requests for WAV,
            # so audio.currentTime = t silently fails during playback.
            rel = unquote(self.path.lstrip("/").split("?")[0])
            fpath = root_dir / rel
            if fpath.suffix.lower() in (".flac", ".mp3", ".wav") and fpath.exists():
                try:
                    self._serve_wav(str(fpath.absolute()))
                    return
                except Exception:
                    pass  # fall through to default handler on error
            super().do_GET()

        def _serve_wav(self, abs_path: str):
            wav = _audio_bytes_as_wav(abs_path)
            total = len(wav)
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                s_str, _, e_str = rng[6:].partition("-")
                start = int(s_str) if s_str else 0
                end   = int(e_str) if e_str else total - 1
                end   = min(end, total - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                chunk = wav[start : end + 1]
            else:
                self.send_response(200)
                start, end, chunk = 0, total - 1, wav
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)

    server = http.server.HTTPServer(("localhost", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _audio_port = port
    return port


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def analysis_dirs():
    if not OUT.exists():
        return []
    paths = [p for p in OUT.iterdir()
             if p.is_dir() and (p / "analysis" / "analysis.json").exists()]
    return sorted(paths, key=lambda p: p.name.casefold())


def sec_color(label, i=0):
    return SEC.get(str(label or "").strip()[:1].upper(), SEC_CYCLE[i % len(SEC_CYCLE)])


def ma(values, n=22):
    arr = np.asarray(values or [], dtype=float)
    if arr.size == 0 or n <= 1:
        return arr
    if arr.size < n:
        return np.full(arr.shape, float(np.nanmean(arr)))
    return np.convolve(arr, np.ones(n) / n, mode="same")


def song_path_for(output_name):
    stems = [output_name]
    stripped = output_name.rstrip("_")
    if stripped and stripped != output_name:
        stems.append(stripped)
    for ext in (".flac", ".mp3", ".wav"):
        for stem in stems:
            for path in SONGS.glob(f"*{ext}"):
                if path.stem == stem:
                    return path
    return None


@st.cache_data(show_spinner=False)
def read_analysis(path_str, mtime):
    del mtime
    with open(path_str, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("analysis.json must contain a JSON object")
    return data


@st.cache_data(show_spinner=False)
def read_optional_json(path_str, mtime):
    del mtime
    with open(path_str, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{Path(path_str).name} must contain a JSON object")
    return data


# ---------------------------------------------------------------------------
# Regenerate helpers
# ---------------------------------------------------------------------------

def start_regen(song_path):
    q = Queue()
    st.session_state.run_queue = q
    st.session_state.run_log = [f"$ uv run songviz analyze {song_path}"]
    st.session_state.running = True
    st.session_state.run_returncode = None

    def worker():
        try:
            proc = subprocess.Popen(
                ["uv", "run", "songviz", "analyze", str(song_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            for line in proc.stdout or []:
                q.put(("line", line.rstrip("\n")))
            q.put(("done", proc.wait()))
        except Exception as exc:
            q.put(("line", f"error: {exc}"))
            q.put(("done", 1))

    t = threading.Thread(target=worker, daemon=True)
    st.session_state.run_thread = t
    t.start()


def stream_log(slot):
    q = st.session_state.get("run_queue")
    logs = st.session_state.get("run_log", [])
    done = None
    while st.session_state.get("running", False):
        if q is not None:
            while True:
                try:
                    kind, payload = q.get_nowait()
                except Empty:
                    break
                if kind == "line":
                    logs.append(str(payload))
                elif kind == "done":
                    done = int(payload)
        st.session_state.run_log = logs
        slot.code("\n".join(logs[-400:]) or "Running...", language="text")
        thread = st.session_state.get("run_thread")
        if done is not None or (thread is not None and not thread.is_alive()):
            st.session_state.running = False
            st.session_state.run_returncode = done
            st.cache_data.clear()
            break
        time.sleep(0.2)
    slot.code("\n".join(logs[-400:]), language="text")


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def _dark(fig, height, extra_bottom=0):
    fig.update_layout(
        height=height,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font={"color": "#cccccc"},
        margin={"l": _ML, "r": _MR, "t": _MT, "b": _MB + extra_bottom},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(gridcolor="#333333", zerolinecolor="#333333")
    fig.update_yaxes(gridcolor="#333333", zerolinecolor="#333333", automargin=False)
    return fig


def _r1_trace(fig, x, y, name, color, width=1.0, visible=True, dash=None, shape=None, customdata=None):
    ln = {"color": color, "width": width}
    if dash:
        ln["dash"] = dash
    fig.add_trace(go.Scatter(x=x, y=y, name=name, line=ln, visible=visible,
                              line_shape=shape, customdata=customdata,
                              mode="lines"), row=1, col=1)


def _add_timeline_row(fig, story, xmax):
    """Add section bars + stem markers to row 2 of a 2-row subplot figure."""
    for i, sec in enumerate(story.get("sections", [])):
        start = float(sec.get("start_s", 0))
        end = float(sec.get("end_s", start))
        label = str(sec.get("label", ""))
        fig.add_trace(go.Bar(
            x=[max(0.0, end - start)], y=[0], base=[start],
            orientation="h", width=[0.6],
            marker_color=sec_color(label, i),
            customdata=[[label, str(sec.get("role", "")), end]],
            hovertemplate=(
                "label=%{customdata[0]}<br>role=%{customdata[1]}<br>"
                "start=%{base:.1f}s<br>end=%{customdata[2]:.1f}s<extra></extra>"
            ),
            showlegend=False,
        ), row=2, col=1)

    transitions = story.get("events", {}).get("stem_transitions", [])
    for kind, symbol, yv in (("enter", "triangle-up", -0.45), ("exit", "triangle-down", 0.45)):
        for stem, color in STEM.items():
            xs = [float(t.get("time_s", 0)) for t in transitions
                  if t.get("kind") == kind and t.get("stem") == stem]
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=[yv] * len(xs), mode="markers",
                    marker={"symbol": symbol, "size": 9, "color": color},
                    hovertemplate=f"{stem} {kind}<br>%{{x:.2f}}s<extra></extra>",
                    showlegend=False,
                ), row=2, col=1)

    fig.update_layout(barmode="overlay")
    fig.update_yaxes(showticklabels=False, ticklen=0, range=[-0.8, 0.8],
                     showgrid=False, zeroline=False, automargin=False, row=2, col=1)
    fig.update_xaxes(range=[0, xmax], title_text="Time (s)", row=2, col=1)


def _structure_tools(diagnostics):
    return list((diagnostics or {}).get("tools", []) or [])


def _tool_color(i):
    return STRUCTURE_TOOL_COLORS[i % len(STRUCTURE_TOOL_COLORS)]


def _tool_display_name(tool):
    return str(tool.get("name") or tool.get("id") or "tool")


def _structure_tool_options(diagnostics):
    return [
        _tool_display_name(tool)
        for tool in _structure_tools(diagnostics)
        if tool.get("available", True) is not False
    ]


def _structure_boundary_rows(diagnostics, active_tools=None):
    active = set(active_tools or [])
    rows = []
    for tool in _structure_tools(diagnostics):
        name = _tool_display_name(tool)
        if active and name not in active:
            continue
        for seg in tool.get("segments", []) or []:
            start = seg.get("start_s")
            end = seg.get("end_s")
            rows.append({
                "Tool": name,
                "Type": "segment",
                "Time": float(start) if start is not None else None,
                "End": float(end) if end is not None else None,
                "Label": seg.get("label", ""),
                "Role": seg.get("role", ""),
                "Boundary conf": seg.get("boundary_confidence", ""),
                "Label conf": seg.get("label_confidence", ""),
                "Beat": seg.get("nearest_beat_idx", ""),
                "Bar": seg.get("nearest_bar_idx", ""),
                "Delta bar": seg.get("delta_to_bar_s", ""),
            })
        for bd in tool.get("boundaries", []) or []:
            t = bd.get("time_s")
            rows.append({
                "Tool": name,
                "Type": bd.get("type", "boundary"),
                "Time": float(t) if t is not None else None,
                "End": "",
                "Label": bd.get("label", ""),
                "Role": bd.get("role", ""),
                "Boundary conf": bd.get("confidence", bd.get("boundary_confidence", "")),
                "Label conf": bd.get("label_confidence", ""),
                "Beat": bd.get("nearest_beat_idx", ""),
                "Bar": bd.get("nearest_bar_idx", ""),
                "Delta bar": bd.get("delta_to_bar_s", ""),
            })
    return sorted(rows, key=lambda r: (
        float("inf") if r["Time"] is None else float(r["Time"]),
        str(r["Tool"]),
        str(r["Type"]),
    ))


def _add_structure_overlay(fig, diagnostics, active_tools=None):
    active = set(active_tools or [])
    for i, tool in enumerate(_structure_tools(diagnostics)):
        name = _tool_display_name(tool)
        if active and name not in active:
            continue
        if tool.get("available", True) is False:
            continue
        color = tool.get("color") or _tool_color(i)

        xs, ys, text = [], [], []
        for bd in tool.get("boundaries", []) or []:
            t = bd.get("time_s")
            if t is None:
                continue
            xs.extend([float(t), float(t), None])
            ys.extend([0.0, 1.0, None])
            label = str(bd.get("label") or bd.get("type") or "boundary")
            conf = bd.get("confidence", bd.get("boundary_confidence"))
            text.extend([
                f"{name}<br>{label}<br>{float(t):.2f}s"
                + (f"<br>conf={float(conf):.2f}" if conf is not None and conf != "" else ""),
                "",
                "",
            ])
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=f"{name} boundaries",
                line={"color": color, "width": 1.2, "dash": "dot"},
                text=text,
                hovertemplate="%{text}<extra></extra>",
                opacity=0.95,
            ), row=1, col=1)

        seg_x, seg_y, seg_text = [], [], []
        for seg in tool.get("segments", []) or []:
            start = seg.get("start_s")
            end = seg.get("end_s")
            if start is None or end is None or float(end) <= float(start):
                continue
            seg_x.extend([float(start), float(end), None])
            seg_y.extend([0.02 + 0.035 * (i % 7), 0.02 + 0.035 * (i % 7), None])
            label = str(seg.get("label") or seg.get("role") or "segment")
            seg_text.extend([
                f"{name}<br>{label}<br>{float(start):.2f}-{float(end):.2f}s",
                "",
                "",
            ])
        if seg_x:
            fig.add_trace(go.Scatter(
                x=seg_x, y=seg_y, mode="lines", name=f"{name} segments",
                line={"color": color, "width": 5},
                text=seg_text,
                hovertemplate="%{text}<extra></extra>",
                opacity=0.85,
            ), row=1, col=1)


def _structure_diagnostics_panel(diagnostics, active_tools=None):
    tools = _structure_tools(diagnostics)
    if not tools:
        st.info("No structure-tool diagnostics found for this analysis.")
        return

    summary_rows = []
    for i, tool in enumerate(tools):
        summary_rows.append({
            "Tool": _tool_display_name(tool),
            "Status": tool.get("status", "available" if tool.get("available", True) else "not run"),
            "Kind": tool.get("kind", ""),
            "Segments": len(tool.get("segments", []) or []),
            "Boundaries": len(tool.get("boundaries", []) or []),
            "Confidence": tool.get("confidence", ""),
            "Notes": tool.get("notes", ""),
        })
    st.dataframe(summary_rows, hide_index=True, use_container_width=True)

    rows = _structure_boundary_rows(diagnostics, active_tools)
    if rows:
        st.markdown("**Boundaries and Segments**")
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _cluster_name(idx: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if idx < len(letters):
        return letters[idx]
    return f"C{idx + 1}"


def _phrase_cluster_labels(sim_matrix, threshold: float) -> list[str]:
    sim = np.asarray(sim_matrix or [], dtype=float)
    if sim.ndim != 2 or sim.shape[0] == 0:
        return []
    n = sim.shape[0]
    clusters: list[list[int]] = []
    labels = [""] * n
    threshold = float(np.clip(threshold, 0.0, 1.0))
    for i in range(n):
        best_idx = None
        best_score = -1.0
        for c_idx, members in enumerate(clusters):
            score = float(np.mean([sim[i, m] for m in members])) if members else -1.0
            if score >= threshold and score > best_score:
                best_idx = c_idx
                best_score = score
        if best_idx is None:
            best_idx = len(clusters)
            clusters.append([])
        clusters[best_idx].append(i)
        labels[i] = _cluster_name(best_idx)
    return labels


def _phrase_cluster_data(phrase_diagnostics, stem: str, phrase_beats: int):
    stem_data = (phrase_diagnostics or {}).get("stems", {}).get(stem, {})
    return stem_data.get("phrase_lengths", {}).get(str(phrase_beats), {})


def _phrase_cluster_options(phrase_diagnostics, stem: str) -> list[int]:
    stem_data = (phrase_diagnostics or {}).get("stems", {}).get(stem, {})
    keys = stem_data.get("phrase_lengths", {}).keys()
    return sorted(int(k) for k in keys if str(k).isdigit())


def _add_phrase_cluster_overlay(fig, phrase_diagnostics, stem: str, phrase_beats: int, threshold: float):
    data = _phrase_cluster_data(phrase_diagnostics, stem, phrase_beats)
    phrases = data.get("phrases", []) or []
    labels = _phrase_cluster_labels(data.get("similarity_matrix", []), threshold)
    if not phrases or not labels:
        return

    for i, phrase in enumerate(phrases):
        start = float(phrase.get("start_s", 0.0))
        end = float(phrase.get("end_s", start))
        if end <= start:
            continue
        label = labels[i]
        color_idx = ord(label[0]) - ord("A") if label and label[0].isalpha() else i
        color = PHRASE_CLUSTER_COLORS[color_idx % len(PHRASE_CLUSTER_COLORS)]
        prev = phrase.get("similarity_to_previous")
        hover = (
            f"{stem} phrase {i}<br>cluster={label}<br>"
            f"{start:.2f}-{end:.2f}s<br>"
            f"bar={phrase.get('bar_index', '')}<br>"
            f"audibility={float(phrase.get('audibility', 0.0)):.2f}"
            + (f"<br>prev sim={float(prev):.3f}" if prev is not None else "")
        )
        fig.add_trace(go.Bar(
            x=[end - start],
            y=[-0.67],
            base=[start],
            orientation="h",
            width=[0.22],
            marker_color=color,
            opacity=0.92,
            text=[label],
            textposition="inside",
            customdata=[[hover]],
            hovertemplate="%{customdata[0]}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)


def _phrase_cluster_panel(phrase_diagnostics, stem: str, phrase_beats: int, threshold: float):
    data = _phrase_cluster_data(phrase_diagnostics, stem, phrase_beats)
    phrases = data.get("phrases", []) or []
    sim = data.get("similarity_matrix", []) or []
    labels = _phrase_cluster_labels(sim, threshold)
    if not phrases or not labels:
        st.info("No phrase cluster diagnostics found for this stem/length.")
        return

    rows = []
    for i, phrase in enumerate(phrases):
        rows.append({
            "Phrase": i,
            "Cluster": labels[i],
            "Start": phrase.get("start_s"),
            "End": phrase.get("end_s"),
            "Bar": phrase.get("bar_index"),
            "Prev sim": phrase.get("similarity_to_previous"),
            "Audibility": phrase.get("audibility"),
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)

    with st.expander("Similarity Matrix", expanded=False):
        header = [f"{i}:{labels[i]}" for i in range(len(labels))]
        matrix = []
        sim_arr = np.asarray(sim, dtype=float)
        for i in range(sim_arr.shape[0]):
            row = {"Phrase": header[i]}
            for j, name in enumerate(header):
                row[name] = round(float(sim_arr[i, j]), 3)
            matrix.append(row)
        st.dataframe(matrix, hide_index=True, use_container_width=True)


def _make_subplot(row_heights=(0.83, 0.17)):
    return make_subplots(
        rows=2, cols=1,
        row_heights=list(row_heights),
        shared_xaxes=True,
        vertical_spacing=0.01,
    )


def combined_fig(analysis, duration=0.0, structure_diagnostics=None, active_structure_tools=None):
    story = analysis["story"]
    env = analysis["envelopes"]
    nov = story["novelties"]
    xmax = duration or float(max(env["times_s"]) if env["times_s"] else 60)

    fig = _make_subplot()

    for i, sec in enumerate(story.get("sections", [])):
        fig.add_shape(
            type="rect", layer="below",
            x0=float(sec.get("start_s", 0)), x1=float(sec.get("end_s", 0)),
            y0=0, y1=1, xref="x", yref="paper",
            fillcolor=sec_color(sec.get("label", ""), i),
            opacity=0.08, line_width=0,
        )
    for drop in story.get("events", {}).get("drop_times_s", []):
        fig.add_shape(type="line", x0=float(drop), x1=float(drop),
                      y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color="red", dash="dash", width=1))

    _r1_trace(fig, env["times_s"], ma(env["loudness"]), "Energy", "#2ca4a4", 1.5)
    _r1_trace(fig, nov["times_s"], nov.get("repetition", []), "Repetition", "#888888", 1.0, "legendonly")
    _r1_trace(fig, nov["times_s"], nov.get("novelty_short", []), "Novelty 4b", "#e35cc8")
    _r1_trace(fig, nov["times_s"], nov.get("novelty_medium", []), "Novelty 16b", "#f09840", 1.1)
    _r1_trace(fig, nov["times_s"], nov.get("novelty_long", []), "Novelty 32b", "#f0d860", 1.3)
    _r1_trace(fig, nov["times_s"], nov.get("section_diff", []), "Section diff", "#ffffff", 2.0, shape="hv")
    _add_structure_overlay(fig, structure_diagnostics, active_structure_tools)

    _add_timeline_row(fig, story, xmax)
    _dark(fig, 560)
    fig.update_yaxes(range=[0, 1], automargin=False, row=1, col=1)
    fig.update_xaxes(range=[0, xmax], showticklabels=False, row=1, col=1)
    return fig


_STEM_METHODS = [
    ("cqt_sim",          "CQT",         ("#5b9cf6", "#3a6fd8", "#2a4fa8")),
    ("phrase_sim",       "Phrase",      ("#c8a2ff", "#9b6eea", "#6f45c8")),
    ("phrase_spec_sim",      "Spec New",  ("#ffd166", "#e0a800", "#a77700")),
    ("phrase_spec_aud_sim",  "Spec Old",  ("#f4a261", "#e76f51", "#9b3522")),
    ("phrase_env_sim",   "Phrase Env",  ("#b7f36b", "#75b843", "#467827")),
    ("phrase_amp_sim",   "Phrase Amp",  ("#7dd3fc", "#0ea5e9", "#075985")),
    ("onset_sim",        "Onset",       ("#ff9966", "#ff5500", "#cc3300")),
]
_DEFAULT_TRACES = {"Energy", "CQT", "Onset"}
_BLOCK_CHOICES = ["8b", "16b", "32b", "4-in-8", "8-in-16", "16-in-32"]
_DEFAULT_BLOCKS = {"16b"}


def _block_curve_key(key_pfx: str, choice: str) -> tuple[str, str, int]:
    if "-in-" not in choice:
        size = int(choice[:-1])
        return f"{key_pfx}_{size}", f"{size}b", size

    child, parent = choice.split("-in-", 1)
    parent_size = int(parent)
    half_pfx = key_pfx[:-4] + "_half_sim" if key_pfx.endswith("_sim") else f"{key_pfx}_half"
    return f"{half_pfx}_{parent_size}", f"{child}b in {parent_size}b", parent_size


def _segment_customdata(x, spans):
    if not x or not spans:
        return None
    ordered = sorted(spans, key=lambda s: float(s.get("start_s", 0.0)))
    out = []
    span_i = 0
    for t_raw in x:
        t = float(t_raw)
        while span_i + 1 < len(ordered) and t >= float(ordered[span_i].get("end_s", 0.0)):
            span_i += 1
        span = ordered[span_i]
        start = float(span.get("start_s", 0.0))
        end = float(span.get("end_s", 0.0))
        out.append([start, end] if start <= t < end and end > start else [None, None])
    return out


def _add_segment_click_targets(fig, x, y, spans, *, color):
    if not x or not y or not spans:
        return
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    n = min(xs.size, ys.size)
    if n == 0:
        return
    xs = xs[:n]
    ys = ys[:n]

    marker_x: list[float] = []
    marker_y: list[float] = []
    marker_cd: list[list[float]] = []
    for span in sorted(spans, key=lambda s: float(s.get("start_s", 0.0))):
        start = float(span.get("start_s", 0.0))
        end = float(span.get("end_s", 0.0))
        if not end > start:
            continue
        mask = (xs >= start) & (xs < end) & np.isfinite(ys)
        if not bool(mask.any()):
            continue
        y_val = float(np.nanmedian(ys[mask]))
        for frac in (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6):
            marker_x.append(start + frac * (end - start))
            marker_y.append(y_val)
            marker_cd.append([start, end])

    if not marker_x:
        return
    fig.add_trace(go.Scatter(
        x=marker_x,
        y=marker_y,
        customdata=marker_cd,
        mode="markers",
        marker={"size": 18, "color": color, "opacity": 0.01, "line": {"width": 0}},
        hoverinfo="skip",
        showlegend=False,
    ), row=1, col=1)


def _add_beat_markers(fig, beat_times, y_max=1.05):
    """Add subtle vertical beat lines to row 1. Every 4th beat is brighter."""
    if not beat_times:
        return
    beats = list(beat_times)
    # Two passes: off-beats (1,2,3) then downbeats (every 4th).
    for phase, color, opacity, width in (
        (None, "#ffffff", 0.10, 0.5),   # all beats (thin, faint)
        (0,    "#ffffff", 0.28, 0.8),   # every 4th beat (brighter)
    ):
        xs, ys = [], []
        for i, t in enumerate(beats):
            if phase is not None and i % 4 != phase:
                continue
            xs += [t, t, None]
            ys += [0.0, y_max, None]
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line={"color": color, "width": width},
            opacity=opacity,
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)


def _phase_votes_text(votes):
    if not isinstance(votes, dict) or not votes:
        return ""
    parts = []
    for key in sorted(votes, key=lambda k: int(k) if str(k).isdigit() else str(k)):
        parts.append(f"{key}: {votes[key]}")
    return "  ".join(parts)


def _alignment_internal_rows(analysis, diagnostics):
    rows = []
    if diagnostics:
        for item in diagnostics.get("internal_analysis", []) or []:
            rows.append({
                "Stem": item.get("stem", ""),
                "Source": item.get("alignment_source_stem", ""),
                "Method": item.get("method", ""),
                "Used phase": item.get("phase", ""),
                "Estimated": item.get("estimated_phase", ""),
                "Accepted": bool(item.get("accepted", False)),
                "Confidence": item.get("confidence", ""),
                "Scores": _phase_votes_text(item.get("scores", {})),
            })
    if rows:
        return rows

    stems = analysis.get("story", {}).get("stem_novelties", {})
    for stem, data in stems.items():
        bg = data.get("bar_grid") or {}
        if not bg:
            continue
        rows.append({
            "Stem": stem,
            "Source": data.get("alignment_source_stem", ""),
            "Method": bg.get("method", ""),
            "Used phase": bg.get("phase", ""),
            "Estimated": bg.get("estimated_phase", ""),
            "Accepted": bool(bg.get("accepted", False)),
            "Confidence": bg.get("confidence", ""),
            "Scores": _phase_votes_text(bg.get("scores", {})),
        })
    return rows


def _alignment_external_rows(diagnostics):
    rows = []
    for src in (diagnostics or {}).get("sources", []) or []:
        if src.get("available") is False:
            rows.append({
                "Library": src.get("name", ""),
                "Selected phase": "",
                "Confidence": "",
                "Votes": "",
                "Downbeats": "",
                "Mean error": "",
                "Status": src.get("warning", "Unavailable"),
            })
            continue
        confidence = src.get("confidence")
        mean_err = src.get("mean_abs_downbeat_error_s")
        rows.append({
            "Library": src.get("name", ""),
            "Selected phase": src.get("selected_phase", ""),
            "Confidence": f"{float(confidence) * 100:.1f}%" if confidence is not None else "",
            "Votes": _phase_votes_text(src.get("phase_votes", {})),
            "Downbeats": src.get("downbeat_count", ""),
            "Mean error": f"{float(mean_err) * 1000:.0f} ms" if mean_err is not None else "",
            "Status": src.get("method", ""),
        })
    return rows


def _bar_alignment_info(analysis, diagnostics):
    """Clickable dashboard summary for bar/downbeat confidence diagnostics."""
    opener = st.popover if hasattr(st, "popover") else st.expander
    with opener("Info: Bar alignment"):
        st.caption(
            "Phase is measured on the current beat grid. Phase 0 means beat indices "
            "0, 4, 8, ... are treated as bar starts."
        )

        if diagnostics and diagnostics.get("summary"):
            st.write(diagnostics["summary"].get("external_consensus", ""))

        external_rows = _alignment_external_rows(diagnostics)
        if external_rows:
            st.markdown("**External libraries**")
            st.dataframe(external_rows, hide_index=True, use_container_width=True)
        else:
            st.info("No external bar-alignment diagnostics found for this analysis.")

        sources = [
            src for src in (diagnostics or {}).get("sources", []) or []
            if src.get("per_beat_preview")
        ]
        if sources:
            st.markdown("**Per-beat preview**")
            src_names = [src.get("name", f"source {i}") for i, src in enumerate(sources)]
            selected = st.selectbox(
                "Source",
                src_names,
                key="bar_alignment_preview_source",
                label_visibility="collapsed",
            )
            source = sources[src_names.index(selected)]
            preview = []
            for row in source.get("per_beat_preview", [])[:48]:
                preview.append({
                    "Beat": row.get("beat_index"),
                    "Time": row.get("time_s"),
                    "Expected pos": row.get("bar_position_from_selected_phase"),
                    "Library pos": row.get("detected_bar_position"),
                    "Error": row.get("nearest_error_s"),
                })
            st.dataframe(preview, hide_index=True, use_container_width=True)


def stem_fig(analysis, stem, duration=0.0,
             active_traces: set | None = None,
             active_blocks: set | None = None,
             phrase_diagnostics=None,
             phrase_beats: int | None = None,
             phrase_threshold: float = 0.84):
    if active_traces is None:
        active_traces = _DEFAULT_TRACES
    if active_blocks is None:
        active_blocks = _DEFAULT_BLOCKS

    story = analysis["story"]
    x = story["novelties"]["times_s"]
    data = story["stem_novelties"][stem]
    xmax = duration or (float(max(x)) if x else 60)

    fig = _make_subplot()
    beat_times = analysis.get("beats", {}).get("beat_times_s", [])
    _add_beat_markers(fig, beat_times)

    if "Energy" in active_traces:
        _r1_trace(fig, x, ma(data.get("energy", [])), "Energy", "#2ca4a4")

    for key_pfx, label, (c8, c16, c32) in _STEM_METHODS:
        if label not in active_traces:
            continue
        colors = {8: c8, 16: c16, 32: c32}
        for choice in _BLOCK_CHOICES:
            if choice not in active_blocks:
                continue
            key, block_label, color_size = _block_curve_key(key_pfx, choice)
            values = data.get(key, [])
            if values:
                spans = data.get("block_spans", {}).get(choice, [])
                customdata = _segment_customdata(x, spans)
                _r1_trace(fig, x, values,
                          f"{label} {block_label}", colors[color_size], visible=True, shape="hv",
                          customdata=customdata)
                _add_segment_click_targets(fig, x, values, spans, color=colors[color_size])

    _add_timeline_row(fig, story, xmax)
    if phrase_diagnostics and phrase_beats:
        _add_phrase_cluster_overlay(fig, phrase_diagnostics, stem, int(phrase_beats), phrase_threshold)
    _dark(fig, 520)
    fig.update_yaxes(range=[0, 1.05], automargin=False, row=1, col=1)
    fig.update_xaxes(range=[0, xmax], showticklabels=False, row=1, col=1)
    return fig


# ---------------------------------------------------------------------------
# Fragment: view selector + chart + audio URL controller
# The fragment re-runs only when its own widgets change (stem / filter).
# The player iframe lives outside the fragment and is never reloaded on stem
# switch, which eliminates canvas flicker and red-line jumps.
# ---------------------------------------------------------------------------

@st.fragment
def _stem_view(analysis, duration, stems_dir, song_path, port,
               bar_diagnostics=None, structure_diagnostics=None,
               phrase_diagnostics=None):
    story = analysis.get("story", {})
    stem_data = story.get("stem_novelties", {})
    present = [s for s in ("bass", "drums", "vocals", "other") if s in stem_data]
    view_options = ["Full Song"] + present

    if st.session_state.get("view_sel") not in view_options:
        st.session_state["view_sel"] = "Full Song"

    view = st.segmented_control("", view_options, key="view_sel")
    if view is None:
        view = "Full Song"

    _bar_alignment_info(analysis, bar_diagnostics)

    has_story = bool(story.get("novelties"))
    if not has_story:
        st.warning("Analysis is outdated — click **Regenerate** in the sidebar to update.")

    elif view == "Full Song":
        structure_options = _structure_tool_options(structure_diagnostics)
        active_structure_tools = []
        if structure_options:
            active_structure_tools = st.pills(
                "Structure tools",
                structure_options,
                default=structure_options,
                key="structure_tools",
                selection_mode="multi",
            ) or []
        try:
            st.plotly_chart(
                combined_fig(analysis, duration, structure_diagnostics, active_structure_tools),
                use_container_width=True, key="full_chart",
            )
            with st.expander("Structure Tool Details", expanded=False):
                _structure_diagnostics_panel(structure_diagnostics, active_structure_tools)
        except Exception as exc:
            st.error(f"Could not render overview: {exc}")

    else:
        all_trace_labels = ["Energy"] + [label for _, label, _ in _STEM_METHODS]
        c_traces, c_blocks, c_phrase, c_thresh = st.columns([4, 1.6, 1.2, 1.2])
        with c_traces:
            sel_traces = st.pills(
                "Traces", all_trace_labels,
                default=sorted(_DEFAULT_TRACES, key=all_trace_labels.index),
                key="vis_traces", selection_mode="multi",
            )
        with c_blocks:
            sel_blocks = st.pills(
                "Block size", _BLOCK_CHOICES,
                default=["16b"],
                key="vis_blocks", selection_mode="multi",
            )
        phrase_options = _phrase_cluster_options(phrase_diagnostics, view)
        phrase_beats = None
        phrase_threshold = float((phrase_diagnostics or {}).get("default_similarity_threshold", 0.84))
        with c_phrase:
            if phrase_options:
                default_phrase = int((phrase_diagnostics or {}).get("default_phrase_beats", 16))
                phrase_beats = st.selectbox(
                    "Phrase",
                    phrase_options,
                    index=phrase_options.index(default_phrase) if default_phrase in phrase_options else 0,
                    format_func=lambda n: f"{n} beats",
                    key=f"phrase_cluster_len_{view}",
                )
        with c_thresh:
            if phrase_options:
                phrase_threshold = st.slider(
                    "Cluster",
                    0.70,
                    0.98,
                    phrase_threshold,
                    0.01,
                    key=f"phrase_cluster_threshold_{view}",
                )
        active_traces = set(sel_traces or [])
        active_blocks = set(sel_blocks or [])
        try:
            st.plotly_chart(
                stem_fig(
                    analysis, view, duration, active_traces, active_blocks,
                    phrase_diagnostics=phrase_diagnostics,
                    phrase_beats=phrase_beats,
                    phrase_threshold=phrase_threshold,
                ),
                use_container_width=True, key="stem_chart",
            )
            if phrase_options and phrase_beats:
                with st.expander("Phrase Clusters", expanded=False):
                    st.caption("Clusters use all-pairs Spec Old similarity for this stem.")
                    _phrase_cluster_panel(phrase_diagnostics, view, int(phrase_beats), phrase_threshold)
            components.html(
                """
                <script>
                (function() {
                try {
                  const pd = window.parent.document;
                  const pw = window.parent;
                  const DBG = false;

                  const numericRange = cd => (
                    cd && cd.length >= 2 && cd[0] != null && cd[1] != null &&
                    Number.isFinite(Number(cd[0])) && Number.isFinite(Number(cd[1]))
                  );
                  const rangePoint = pts => pts && pts.find(pt => numericRange(pt && pt.customdata));

                  const publishRange = pt => {
                    const cd = pt && pt.customdata;
                    if (!cd || cd.length < 2 || cd[0] == null || cd[1] == null) return;
                    const start = Number(cd[0]), end = Number(cd[1]);
                    if (!(end > start)) return;
                    if (DBG) console.log('[SG] publishRange:', start.toFixed(2), '->', end.toFixed(2));
                    pw.__sg_rangeCommand__ = {
                      id: String(Date.now()) + ':' + String(Math.random()),
                      start,
                      end,
                    };
                  };

                  const pointFromClick = (plot, e) => {
                    const layout = plot._fullLayout || {};
                    const xa = layout.xaxis, ya = layout.yaxis;
                    if (!xa || !ya || typeof xa.p2l !== 'function' || typeof ya.p2l !== 'function') {
                      if (DBG) console.log('[SG] pointFromClick: no p2l', xa, ya);
                      return null;
                    }
                    const rect = plot.getBoundingClientRect();
                    const xPx = e.clientX - rect.left - xa._offset;
                    const yPx = e.clientY - rect.top - ya._offset;
                    const xVal = xa.p2l(xPx);
                    const yVal = ya.p2l(yPx);
                    if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) return null;
                    const yRange = Array.isArray(ya.range) ? Math.abs(ya.range[1] - ya.range[0]) : 1.05;
                    const yTol = Math.max(0.10, yRange * 28 / Math.max(1, ya._length || 1));
                    let best = null;
                    let fallback = null;
                    for (let c = 0; c < (plot.data || []).length; c++) {
                      const tr = plot.data[c];
                      if (!tr || !tr.customdata || !tr.x || !tr.y) continue;
                      for (let i = 0; i < tr.customdata.length; i++) {
                        const cd = tr.customdata[i];
                        if (!numericRange(cd)) continue;
                        const s = Number(cd[0]), e2 = Number(cd[1]);
                        if (xVal < s || xVal >= e2) continue;
                        const y = Number(tr.y[i]);
                        if (!Number.isFinite(y)) continue;
                        const dy = Math.abs(y - yVal);
                        if (!fallback || dy < fallback.dy) fallback = {dy, customdata: cd};
                        if (dy <= yTol && (!best || dy < best.dy)) best = {dy, customdata: cd};
                      }
                    }
                    return best || (fallback && fallback.dy <= 0.22 ? fallback : null);
                  };

                  pw.__sgPublishRange__ = publishRange;
                  pw.__sgPointFromClick__ = pointFromClick;
                  if (!pw.__sgDocumentPlotClickInstalled__) {
                    pw.__sgDocumentPlotClickInstalled__ = true;
                    pd.addEventListener('click', e => {
                      let target = e.target;
                      if (target && target.nodeType !== 1) target = target.parentElement;
                      const plot = target && target.closest && target.closest('.js-plotly-plot');
                      if (!plot) return;
                      const resolver = pw.__sgPointFromClick__;
                      const publisher = pw.__sgPublishRange__;
                      if (!resolver || !publisher) return;
                      const clicked = resolver(plot, e);
                      if (clicked) {
                        plot.__sgLastRangePoint = clicked;
                        publisher(clicked);
                      } else if (plot.__sgLastRangePoint) {
                        setTimeout(() => publisher(plot.__sgLastRangePoint), 0);
                      }
                    }, true);
                  }

                  const installHandlers = plot => {
                    if (!plot || typeof plot.on !== 'function') {
                      if (DBG) console.log('[SG] installHandlers: plot not ready', plot);
                      return false;
                    }
                    if (DBG) console.log('[SG] installHandlers: installing on', plot.id || plot);

                    if (plot.__sgHoverHandler) try { plot.removeListener('plotly_hover', plot.__sgHoverHandler); } catch(_) {}
                    if (plot.__sgPlotlyClickHandler) try { plot.removeListener('plotly_click', plot.__sgPlotlyClickHandler); } catch(_) {}
                    plot.__sgHoverHandler = ev => { plot.__sgLastRangePoint = rangePoint(ev && ev.points); };
                    plot.__sgPlotlyClickHandler = ev => {
                      if (DBG) console.log('[SG] plotly_click event', ev && ev.points && ev.points.length);
                      publishRange(rangePoint(ev && ev.points));
                    };
                    plot.on('plotly_hover', plot.__sgHoverHandler);
                    plot.on('plotly_click', plot.__sgPlotlyClickHandler);

                    plot.__sgPointFromClick = e => pointFromClick(plot, e);
                    plot.__sgPublishRange = publishRange;

                    if (!plot.__sgDomClickInstalled) {
                      plot.__sgDomClickInstalled = true;
                      plot.addEventListener('click', e => {
                        if (DBG) console.log('[SG] DOM click fired');
                        const pfc = plot.__sgPointFromClick;
                        const pub = plot.__sgPublishRange;
                        if (!pfc || !pub) { if (DBG) console.log('[SG] DOM click: no handlers'); return; }
                        const clicked = pfc(e);
                        if (clicked) { plot.__sgLastRangePoint = clicked; pub(clicked); return; }
                        setTimeout(() => { if (plot.__sgPublishRange) plot.__sgPublishRange(plot.__sgLastRangePoint); }, 0);
                      }, true);
                    }
                    return true;
                  };

                  const tryAll = () => { for (const p of pd.querySelectorAll('.js-plotly-plot')) installHandlers(p); };
                  tryAll();
                  setTimeout(tryAll, 150);
                  setTimeout(tryAll, 500);
                } catch(err) { console.log('[SG] top-level error', err); }
                })();
                </script>
                """,
                height=0,
            )
        except Exception as exc:
            st.error(f"Could not render stem chart: {exc}")

    # Publish the current audio URL so the stable player iframe can poll it.
    # The controller is a zero-height iframe that just runs one JS assignment.
    if port is not None and song_path is not None:
        view_sel = st.session_state.get("view_sel", "Full Song")
        if view_sel and view_sel != "Full Song":
            stem_wav = stems_dir / f"{view_sel}.wav"
            audio_url = (
                "http://localhost:{}/{}".format(port, quote(str(stem_wav), safe="/"))
                if stem_wav.exists()
                else "http://localhost:{}/{}".format(port, quote(str(song_path), safe="/"))
            )
        else:
            audio_url = "http://localhost:{}/{}".format(port, quote(str(song_path), safe="/"))
        url_js = json.dumps(audio_url)
        components.html(
            f"<script>try{{window.parent.__sg_audioUrl__={url_js};}}catch(e){{}}</script>",
            height=0,
        )


# ---------------------------------------------------------------------------
# Synchronized audio player (custom HTML component)
# ---------------------------------------------------------------------------

def _player_html(analysis, duration: float) -> str:
    """Polling player. Reads window.parent.__sg_audioUrl__ (published by the
    fragment controller) to switch audio sources without reloading this iframe,
    so the canvas and red-line position are preserved across stem changes."""
    story = analysis["story"]
    env = analysis["envelopes"]

    sections_data = [
        {
            "s": float(sec.get("start_s", 0)),
            "e": float(sec.get("end_s", 0)),
            "c": sec_color(sec.get("label", ""), i),
        }
        for i, sec in enumerate(story.get("sections", []))
    ]

    # Downsample energy to ~600 points for canvas drawing
    times_raw = list(env["times_s"])
    loud_raw = ma(env["loudness"], 10).tolist()
    step = max(1, len(times_raw) // 600)
    energy_times = times_raw[::step]
    energy_vals = loud_raw[::step]

    secs_js = json.dumps(sections_data)
    et_js = json.dumps(energy_times)
    ev_js = json.dumps(energy_vals)
    beats_js = json.dumps([
        float(t) for t in analysis.get("beats", {}).get("beat_times_s", [])
        if 0.0 <= float(t) <= float(duration)
    ])
    dur_fmt = f"{int(duration // 60)}:{int(duration % 60):02d}"
    ml, mr = _ML, _MR

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111; color: #ccc; font-family: sans-serif; overflow: hidden; }}
  #wrap {{ padding: 6px 0 4px; }}
  #row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
          padding-left: {ml}px; padding-right: {mr}px; }}
  #btn {{ background: #2ca4a4; border: none; border-radius: 4px;
          color: #fff; width: 32px; height: 32px; font-size: 16px;
          cursor: pointer; flex-shrink: 0; }}
  #btn:hover {{ background: #3ac0c0; }}
  #loopBtn {{ background: #23262d; border: 1px solid #3a3f48; border-radius: 4px;
              color: #aaa; width: 32px; height: 32px; font-size: 17px;
              cursor: pointer; flex-shrink: 0; }}
  #loopBtn.on {{ background: #2ca4a4; border-color: #3ac0c0; color: #fff; }}
  #metroBtn {{ background: #23262d; border: 1px solid #3a3f48; border-radius: 4px;
               color: #aaa; width: 32px; height: 32px; font-size: 13px;
               cursor: pointer; flex-shrink: 0; font-weight: 700; }}
  #metroBtn.on {{ background: #b78f2a; border-color: #d7aa35; color: #111; }}
  #time {{ font-size: 12px; color: #aaa; min-width: 90px; }}
  #cwrap {{ margin-left: {ml}px; margin-right: {mr}px; }}
  #canvas {{ display: block; width: 100%; height: 48px; cursor: pointer; border-radius: 3px; }}
</style>
</head>
<body>
<div id="wrap">
  <audio id="audio" preload="auto"></audio>
  <div id="row">
    <button id="btn" onclick="toggle()">&#9654;</button>
    <button id="loopBtn" title="Cycle selected block" onclick="toggleLoop()">&#8635;</button>
    <button id="metroBtn" title="Metronome" onclick="toggleMetronome()">M</button>
    <span id="time">0:00 / {dur_fmt}</span>
  </div>
  <div id="cwrap"><canvas id="canvas"></canvas></div>
</div>
<script>
const audio = document.getElementById('audio');
const btn   = document.getElementById('btn');
const loopBtn = document.getElementById('loopBtn');
const metroBtn = document.getElementById('metroBtn');
const timeEl = document.getElementById('time');
const canvas = document.getElementById('canvas');
const ctx   = canvas.getContext('2d');
const SECS  = {secs_js};
const ET    = {et_js};
const EV    = {ev_js};
const BEATS = {beats_js};
const DUR   = {duration};

function fmt(t) {{
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return m + ':' + String(s).padStart(2, '0');
}}

// ── parent-page overlay (red line spanning charts above) ──────────────────
function ensureOverlay() {{
  try {{
    const pd = window.parent.document;
    let line = pd.getElementById('__sg_playhead__');
    if (!line) {{
      line = pd.createElement('div');
      line.id = '__sg_playhead__';
      line.style.cssText = [
        'position:fixed', 'width:2px', 'background:rgba(255,68,68,0.80)',
        'z-index:9999', 'pointer-events:none', 'display:none',
        'transition:none',
      ].join(';');
      pd.body.appendChild(line);
    }}
    return line;
  }} catch(e) {{ return null; }}
}}

function updateOverlay(t) {{
  const line = ensureOverlay();
  if (!line || DUR <= 0) return;
  try {{
    const iframeRect = window.frameElement.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    // canvas coords are in iframe viewport; shift to parent viewport
    const cx0 = iframeRect.left + canvasRect.left;
    const cx1 = iframeRect.left + canvasRect.right;
    const frac = Math.max(0, Math.min(1, t / DUR));
    const px = cx0 + frac * (cx1 - cx0);

    const pd = window.parent.document;
    // find the topmost visible Plotly chart in the parent
    let chartTop = iframeRect.top;
    for (const div of pd.querySelectorAll('.js-plotly-plot')) {{
      const r = div.getBoundingClientRect();
      if (r.height > 10 && r.top < chartTop) chartTop = r.top;
    }}
    const lineBottom = iframeRect.top + canvasRect.bottom;

    line.style.left = (px - 1) + 'px';
    line.style.top  = chartTop + 'px';
    line.style.height = Math.max(0, lineBottom - chartTop) + 'px';
    line.style.display = 'block';
  }} catch(e) {{}}
}}

function hideOverlay() {{
  try {{
    const line = window.parent.document.getElementById('__sg_playhead__');
    if (line) line.style.display = 'none';
  }} catch(e) {{}}
}}

// ── canvas draw ───────────────────────────────────────────────────────────
function draw(t) {{
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  canvas.width = W; canvas.height = H;

  for (const sec of SECS) {{
    const x1 = (sec.s / DUR) * W, x2 = (sec.e / DUR) * W;
    ctx.fillStyle = sec.c + '44';
    ctx.fillRect(x1, 0, x2 - x1, H);
    ctx.strokeStyle = sec.c + 'bb';
    ctx.lineWidth = 1;
    ctx.strokeRect(x1 + 0.5, 0.5, x2 - x1 - 1, H - 1);
  }}

  ctx.beginPath();
  for (let i = 0; i < ET.length; i++) {{
    const x = (ET[i] / DUR) * W, y = H - EV[i] * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }}
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = 'rgba(44,164,164,0.18)';
  ctx.fill();
  ctx.beginPath();
  for (let i = 0; i < ET.length; i++) {{
    const x = (ET[i] / DUR) * W, y = H - EV[i] * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }}
  ctx.strokeStyle = '#2ca4a4';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  const px = (t / DUR) * W;
  ctx.strokeStyle = '#ff4444';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();

  timeEl.textContent = fmt(t) + ' / ' + fmt(DUR);
  updateOverlay(t);
}}

let rafId = null;
let loopStart = 0;
let loopEnd = 0;
let loopEnabled = false;
let loopTimerId = null;
let lastRangeCommandId = '';
const LOOP_TIMER_LEAD_S = 0.008;
let webAudioCtx = null;
let webLoopSource = null;
let webLoopActive = false;
let webLoopStartedAtCtx = 0;
let webLoopOffset = 0;
const webAudioBuffers = new Map();
let metronomeEnabled = false;
let metroLastT = 0;
let metroCycle = 0;
const metroScheduled = new Set();
const METRO_LOOKAHEAD_S = 0.11;
const METRO_RESET_BACKSTEP_S = 0.20;

try {{
  lastRangeCommandId = (window.parent.__sg_rangeCommand__ && window.parent.__sg_rangeCommand__.id) || '';
}} catch(e) {{}}

function clearLoopTimer() {{
  if (loopTimerId) {{
    clearTimeout(loopTimerId);
    loopTimerId = null;
  }}
}}

function loopDuration() {{
  return Math.max(0, loopEnd - loopStart);
}}

function playbackTime() {{
  if (webLoopActive && webAudioCtx && loopDuration() > 0) {{
    const elapsed = Math.max(0, webAudioCtx.currentTime - webLoopStartedAtCtx);
    return loopStart + ((webLoopOffset - loopStart + elapsed) % loopDuration());
  }}
  return audio.currentTime;
}}

function isPlaying() {{
  return webLoopActive || !audio.paused;
}}

function scheduleLoopBoundary() {{
  clearLoopTimer();
  if (!loopEnabled || webLoopActive || audio.paused || loopDuration() <= 0) return;
  const remainingS = loopEnd - audio.currentTime - LOOP_TIMER_LEAD_S;
  const delayMs = Math.max(2, remainingS * 1000);
  loopTimerId = setTimeout(wrapLoopBoundary, delayMs);
}}

function wrapLoopBoundary() {{
  if (!loopEnabled || webLoopActive || audio.paused || loopDuration() <= 0) return;
  const now = audio.currentTime;
  if (now < loopEnd) {{
    scheduleLoopBoundary();
    return;
  }}
  const overshoot = Math.max(0, now - loopEnd);
  audio.currentTime = loopStart + (overshoot % loopDuration());
  draw(audio.currentTime);
  scheduleLoopBoundary();
}}

function stopWebLoop() {{
  const t = playbackTime();
  if (webLoopSource) {{
    try {{ webLoopSource.onended = null; webLoopSource.stop(); }} catch(e) {{}}
    try {{ webLoopSource.disconnect(); }} catch(e) {{}}
  }}
  webLoopSource = null;
  webLoopActive = false;
  return t;
}}

async function currentAudioBuffer() {{
  const url = _currentSrc || audio.currentSrc || audio.src;
  if (!url) throw new Error('No current audio URL');
  await ensureAudioContext();
  if (webAudioBuffers.has(url)) return webAudioBuffers.get(url);
  const response = await fetch(url);
  const bytes = await response.arrayBuffer();
  const buffer = await webAudioCtx.decodeAudioData(bytes.slice(0));
  webAudioBuffers.set(url, buffer);
  return buffer;
}}

async function ensureAudioContext() {{
  if (!webAudioCtx) {{
    webAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }}
  if (webAudioCtx.state === 'suspended') await webAudioCtx.resume();
  return webAudioCtx;
}}

function resetMetronome(t) {{
  metroLastT = Number.isFinite(Number(t)) ? Number(t) : playbackTime();
  metroCycle = 0;
  metroScheduled.clear();
}}

function updateMetronomeButton() {{
  metroBtn.classList.toggle('on', metronomeEnabled);
}}

async function toggleMetronome() {{
  metronomeEnabled = !metronomeEnabled;
  updateMetronomeButton();
  resetMetronome(playbackTime());
  if (metronomeEnabled) {{
    try {{ await ensureAudioContext(); }} catch(e) {{}}
  }}
}}

function playMetroClick(ctxTime, isBar) {{
  if (!webAudioCtx) return;
  const osc = webAudioCtx.createOscillator();
  const gain = webAudioCtx.createGain();
  const dur = isBar ? 0.055 : 0.035;
  osc.type = 'square';
  osc.frequency.setValueAtTime(isBar ? 1500 : 950, ctxTime);
  gain.gain.setValueAtTime(0.0001, ctxTime);
  gain.gain.exponentialRampToValueAtTime(isBar ? 0.22 : 0.10, ctxTime + 0.002);
  gain.gain.exponentialRampToValueAtTime(0.0001, ctxTime + dur);
  osc.connect(gain);
  gain.connect(webAudioCtx.destination);
  osc.start(ctxTime);
  osc.stop(ctxTime + dur + 0.01);
}}

function scheduleMetronome(t) {{
  if (!metronomeEnabled || !isPlaying() || !BEATS.length) return;
  if (!webAudioCtx) return;
  if (t < metroLastT - METRO_RESET_BACKSTEP_S) {{
    metroCycle += 1;
    metroScheduled.clear();
  }}
  metroLastT = t;

  const windows = [];
  const end = t + METRO_LOOKAHEAD_S;
  if (loopEnabled && loopDuration() > 0) {{
    if (end < loopEnd) {{
      windows.push([Math.max(t, loopStart), end, metroCycle]);
    }} else {{
      windows.push([Math.max(t, loopStart), loopEnd, metroCycle]);
      windows.push([loopStart, loopStart + (end - loopEnd), metroCycle + 1]);
    }}
  }} else {{
    windows.push([t, end, metroCycle]);
  }}

  for (const [a, b, cyc] of windows) {{
    for (let i = 0; i < BEATS.length; i++) {{
      const beatT = BEATS[i];
      if (beatT < a || beatT >= b) continue;
      const key = cyc + ':' + i;
      if (metroScheduled.has(key)) continue;
      metroScheduled.add(key);
      const delay = Math.max(0, beatT - t);
      playMetroClick(webAudioCtx.currentTime + delay, i % 4 === 0);
    }}
  }}
}}

async function startWebLoop(offset) {{
  if (!loopEnabled || loopDuration() <= 0) return false;
  const buffer = await currentAudioBuffer();
  const source = webAudioCtx.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  source.loopStart = Math.max(0, Math.min(buffer.duration, loopStart));
  source.loopEnd = Math.max(source.loopStart, Math.min(buffer.duration, loopEnd));
  if (!(source.loopEnd > source.loopStart)) return false;
  const startAt = Math.max(source.loopStart, Math.min(source.loopEnd - 0.001, Number(offset) || source.loopStart));
  stopWebLoop();
  clearLoopTimer();
  audio.pause();
  source.connect(webAudioCtx.destination);
  webLoopSource = source;
  webLoopActive = true;
  webLoopStartedAtCtx = webAudioCtx.currentTime;
  webLoopOffset = startAt;
  source.start(0, startAt);
  return true;
}}

function updateLoopButton() {{
  loopBtn.classList.toggle('on', loopEnabled);
}}

function setLoopEnabled(enabled) {{
  loopEnabled = !!enabled && loopEnd > loopStart;
  updateLoopButton();
  loopEnabled ? scheduleLoopBoundary() : clearLoopTimer();
  try {{ window.parent.__sg_loopEnabled__ = loopEnabled; }} catch(e) {{}}
}}

async function toggleLoop() {{
  if (loopEnabled) {{
    const t = stopWebLoop();
    setLoopEnabled(false);
    if (!audio.paused || webLoopActive) return;
    audio.currentTime = Math.min(DUR, t);
    resetMetronome(audio.currentTime);
    audio.play().catch(() => {{}});
    btn.innerHTML = '&#9646;&#9646;';
    rafId = requestAnimationFrame(loop);
  }} else {{
    setLoopEnabled(true);
    if (!audio.paused) {{
      try {{ await startWebLoop(audio.currentTime || loopStart); resetMetronome(playbackTime()); }} catch(e) {{ scheduleLoopBoundary(); }}
    }}
  }}
}}

function loop() {{
  if (loopEnabled && loopDuration() > 0 && !webLoopActive && audio.currentTime >= loopEnd) {{
    wrapLoopBoundary();
  }}
  const t = playbackTime();
  draw(t);
  scheduleMetronome(t);
  // Persist position for page-reload restoration.
  try {{ window.parent.__sg_playTime__ = t;
         window.parent.__sg_wasPlaying__ = true; }} catch(e) {{}}
  if (isPlaying()) rafId = requestAnimationFrame(loop);
}}

async function toggle() {{
  if (!isPlaying()) {{
    if (loopEnabled && loopDuration() > 0) {{
      try {{
        if (await startWebLoop(audio.currentTime || loopStart)) {{
          btn.innerHTML = '&#9646;&#9646;';
          resetMetronome(playbackTime());
          rafId = requestAnimationFrame(loop);
          return;
        }}
      }} catch(e) {{}}
    }}
    if (metronomeEnabled) {{
      try {{ await ensureAudioContext(); }} catch(e) {{}}
    }}
    audio.play().catch(() => {{}});
    btn.innerHTML = '&#9646;&#9646;';
    resetMetronome(audio.currentTime);
    scheduleLoopBoundary();
    rafId = requestAnimationFrame(loop);
  }} else {{
    const t = stopWebLoop();
    audio.pause();
    audio.currentTime = Math.min(DUR, t);
    btn.innerHTML = '&#9654;';
    clearLoopTimer();
    if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
    try {{ window.parent.__sg_wasPlaying__ = false; }} catch(e) {{}}
  }}
}}

function isSpaceKey(e) {{
  return e.code === 'Space' || e.key === ' ' || e.key === 'Spacebar';
}}

function isEditableTarget(target) {{
  if (!target) return false;
  const tag = String(target.tagName || '').toLowerCase();
  return tag === 'input' || tag === 'textarea' || target.isContentEditable;
}}

function handleSpaceToggle(e) {{
  if (!isSpaceKey(e) || e.repeat || isEditableTarget(e.target)) return;
  e.preventDefault();
  e.stopPropagation();
  toggle();
}}

window.addEventListener('keydown', handleSpaceToggle, true);

try {{
  window.parent.__sg_togglePlayback__ = () => toggle();
  if (!window.parent.__sg_spaceToggleInstalled__) {{
    window.parent.__sg_spaceToggleInstalled__ = true;
    window.parent.document.addEventListener('keydown', e => {{
      if (!isSpaceKey(e) || e.repeat || isEditableTarget(e.target)) return;
      e.preventDefault();
      e.stopPropagation();
      try {{
        if (typeof window.parent.__sg_togglePlayback__ === 'function') {{
          window.parent.__sg_togglePlayback__();
        }}
      }} catch(err) {{}}
    }}, true);
  }}
}} catch(e) {{}}

audio.addEventListener('ended', () => {{
  btn.innerHTML = '&#9654;';
  clearLoopTimer();
  if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
  draw(DUR);
  try {{ window.parent.__sg_wasPlaying__ = false; }} catch(e) {{}}
  hideOverlay();
}});

// ── seeking ───────────────────────────────────────────────────────────────
async function _seekToTime(t) {{
  const wasPlaying = isPlaying();
  stopWebLoop();
  setLoopEnabled(false);
  if (!audio.paused) {{
    audio.pause();
    clearLoopTimer();
    if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
  }}
  audio.currentTime = t;
  draw(t);
  resetMetronome(t);
  if (wasPlaying) {{
    if (metronomeEnabled) {{
      try {{ await ensureAudioContext(); }} catch(e) {{}}
    }}
    audio.play().catch(() => {{}});
    scheduleLoopBoundary();
    rafId = requestAnimationFrame(loop);
  }}
}}

async function playRange(start, end) {{
  start = Math.max(0, Math.min(DUR, Number(start) || 0));
  end = Math.max(start, Math.min(DUR, Number(end) || start));
  if (end <= start) return;
  loopStart = start;
  loopEnd = end;
  setLoopEnabled(true);
  if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
  stopWebLoop();
  audio.currentTime = loopStart;
  draw(loopStart);
  resetMetronome(loopStart);
  if (metronomeEnabled) {{
    try {{ await ensureAudioContext(); }} catch(e) {{}}
  }}
  audio.play().catch(() => {{}});
  btn.innerHTML = '&#9646;&#9646;';
  scheduleLoopBoundary();
  rafId = requestAnimationFrame(loop);

  try {{
    if (await startWebLoop(loopStart)) {{
      draw(loopStart);
      btn.innerHTML = '&#9646;&#9646;';
      resetMetronome(loopStart);
      if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
      rafId = requestAnimationFrame(loop);
      return;
    }}
  }} catch(e) {{}}
}}

// Track last reliable mouse X (pointerdown fires before synthetic click events
// that Streamlit injects at offsetX=0 to manage iframe focus).
let _lastMouseX = 0;
let _fromPointer = false;
canvas.addEventListener('pointermove', e => {{ _lastMouseX = e.offsetX; }});
canvas.addEventListener('pointerdown', e => {{
  _lastMouseX = e.offsetX;
  _fromPointer = true;
  const frac = Math.max(0, Math.min(1, e.offsetX / canvas.offsetWidth));
  _seekToTime(frac * DUR);
}});
canvas.addEventListener('click', e => {{
  if (_fromPointer) {{ _fromPointer = false; return; }}  // already handled
  const ox = (e.offsetX > 0) ? e.offsetX : _lastMouseX;
  const frac = Math.max(0, Math.min(1, ox / canvas.offsetWidth));
  _seekToTime(frac * DUR);
}});

window.addEventListener('resize', () => draw(playbackTime()));

// ── seamless source switching ─────────────────────────────────────────────
// The fragment controller iframe updates window.parent.__sg_audioUrl__ on each
// stem change. We poll for it and call switchSource when it differs — no iframe
// reload, no canvas reset, no red-line jump.
let _currentSrc = '';
let _isFirstLoad = true;

function switchSource(url) {{
  _currentSrc = url;
  let t0, play0;
  if (_isFirstLoad) {{
    // Restore position from a previous session (e.g. Streamlit hot-reload).
    try {{ t0 = +window.parent.__sg_playTime__ || 0; }} catch(e) {{ t0 = 0; }}
    try {{ play0 = !!window.parent.__sg_wasPlaying__; }} catch(e) {{ play0 = false; }}
    _isFirstLoad = false;
  }} else {{
    // Stem switch: preserve current position and playing state.
    t0 = playbackTime();
    play0 = isPlaying();
  }}
  stopWebLoop();
  if (!audio.paused) {{
    audio.pause();
    clearLoopTimer();
    if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
  }}
  audio.src = url;
  audio.load();
  audio.addEventListener('loadedmetadata', () => {{
    if (t0 > 0.1) audio.currentTime = Math.min(t0, DUR);
    draw(audio.currentTime);
    if (play0) {{
      if (loopEnabled && loopDuration() > 0) {{
        startWebLoop(Math.max(loopStart, Math.min(loopEnd - 0.001, t0))).then(ok => {{
          if (!ok) {{
            audio.play().catch(() => {{}});
            scheduleLoopBoundary();
          }}
          btn.innerHTML = '&#9646;&#9646;';
          rafId = requestAnimationFrame(loop);
        }}).catch(() => {{
          audio.play().catch(() => {{}});
          btn.innerHTML = '&#9646;&#9646;';
          scheduleLoopBoundary();
          rafId = requestAnimationFrame(loop);
        }});
      }} else {{
        audio.play().catch(() => {{}});
        btn.innerHTML = '&#9646;&#9646;';
        scheduleLoopBoundary();
        rafId = requestAnimationFrame(loop);
      }}
    }}
  }}, {{ once: true }});
}}

function pollUrl() {{
  let url = '';
  try {{ url = window.parent.__sg_audioUrl__ || ''; }} catch(e) {{}}
  if (url && url !== _currentSrc) switchSource(url);
  try {{
    const cmd = window.parent.__sg_rangeCommand__;
    if (cmd && cmd.id && cmd.id !== lastRangeCommandId) {{
      lastRangeCommandId = cmd.id;
      playRange(cmd.start, cmd.end);
    }}
  }} catch(e) {{}}
  setTimeout(pollUrl, 400);
}}

draw(0);
pollUrl();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Song Visualizer", layout="wide")
    st.title("Song Visualizer")
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("run_log", [])

    dirs = analysis_dirs()
    if not dirs:
        st.error("No analyses found under outputs/*/analysis/analysis.json")
        st.stop()

    names = [p.name for p in dirs]
    default_idx = next(
        (i for i, n in enumerate(names) if "arctic monkeys" in n.lower() or "do i wanna know" in n.lower()),
        0,
    )
    selected = st.sidebar.selectbox("Song", names, index=default_idx)
    selected_dir = OUT / selected
    song_path = song_path_for(selected)

    if st.sidebar.button("Regenerate", disabled=st.session_state.running):
        if song_path is None:
            st.sidebar.error(f"No matching audio file found in songs/ for {selected}")
        else:
            start_regen(song_path)

    log_slot = st.sidebar.empty()
    if st.session_state.running:
        stream_log(log_slot)
        code = st.session_state.get("run_returncode")
        (st.sidebar.success if code == 0 else st.sidebar.error)(
            f"Regenerate finished with exit code {code}"
        )
    elif st.session_state.get("run_log"):
        log_slot.code("\n".join(st.session_state.run_log[-80:]), language="text")

    json_path = selected_dir / "analysis" / "analysis.json"
    if not json_path.exists():
        st.error(f"Missing analysis.json: {json_path}")
        st.stop()
    try:
        analysis = read_analysis(str(json_path), json_path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not load analysis.json: {exc}")
        st.stop()

    bar_diagnostics = None
    bar_diag_path = selected_dir / "analysis" / "bar_alignment_diagnostics.json"
    if bar_diag_path.exists():
        try:
            bar_diagnostics = read_optional_json(str(bar_diag_path), bar_diag_path.stat().st_mtime)
        except Exception as exc:
            st.sidebar.warning(f"Could not load bar alignment diagnostics: {exc}")

    structure_diagnostics = None
    structure_diag_path = selected_dir / "analysis" / "structure_tool_diagnostics.json"
    if structure_diag_path.exists():
        try:
            structure_diagnostics = read_optional_json(
                str(structure_diag_path), structure_diag_path.stat().st_mtime
            )
        except Exception as exc:
            st.sidebar.warning(f"Could not load structure tool diagnostics: {exc}")

    phrase_diagnostics = None
    phrase_diag_path = selected_dir / "analysis" / "phrase_cluster_diagnostics.json"
    if phrase_diag_path.exists():
        try:
            phrase_diagnostics = read_optional_json(str(phrase_diag_path), phrase_diag_path.stat().st_mtime)
        except Exception as exc:
            st.sidebar.warning(f"Could not load phrase cluster diagnostics: {exc}")

    duration = float(analysis.get("meta", {}).get("duration_s", 0.0))

    # Reset view when song changes; preserve filter prefs (they span all songs).
    if st.session_state.get("_current_song") != selected:
        st.session_state["_current_song"] = selected
        st.session_state.pop("view_sel", None)

    port = _ensure_audio_server() if song_path and song_path.exists() and duration > 0 else None
    stems_dir = selected_dir / "stems"

    # Fragment: view selector + chart. Re-runs in isolation on stem/filter changes.
    _stem_view(analysis, duration, stems_dir, song_path, port,
               bar_diagnostics, structure_diagnostics, phrase_diagnostics)

    # ── Video (collapsed by default) ─────────────────────────────────────────
    video_path = selected_dir / "overview_video.mp4"
    if video_path.exists():
        with st.expander("Video"):
            st.video(str(video_path))

    # ── Audio player (outside fragment — never reloads on stem switch) ────────
    if port is not None and duration > 0:
        st.divider()
        components.html(_player_html(analysis, duration), height=120)
    elif song_path is None:
        st.info("No matching audio file found in songs/ — player unavailable.")


if __name__ == "__main__":
    main()
