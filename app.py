import html
import json
import os
import tempfile
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from main import (
    DEFAULT_FILLERS,
    FILLER_PATTERNS,
    FILLER_PROMPT,
    METRIC_GROUPS,
    METRIC_INFO,
    YAKE_OK,
    compute_sections,
    fmt_time,
    load_model,
    make_srt,
    metric_insight,
    summary_row,
    transcribe_audio,
)

st.set_page_config(page_title="Audio Transcriber", page_icon="🎙️", layout="wide")


# ---------------------------------------------------------------- UI helpers

@st.cache_resource(show_spinner="Loading Whisper model…")
def get_model(path: str, device: str, compute_type: str):
    return load_model(path, device, compute_type)


def copy_button(text: str, label: str = "📋 Copy"):
    """Clipboard button rendered in a tiny HTML component."""
    payload = json.dumps(text).replace("</", "<\\/")

    components.html(
        f"""
        <button id="b" style="padding:6px 14px;border-radius:8px;border:1px solid #888;
                background:transparent;color:inherit;cursor:pointer;font-size:14px;">
            {label}
        </button>
        <script>
        const text = {payload};
        const btn = document.getElementById('b');
        btn.onclick = async () => {{
            try {{ await navigator.clipboard.writeText(text); }}
            catch (e) {{
                const t = document.createElement('textarea');
                t.value = text; document.body.appendChild(t); t.select();
                document.execCommand('copy'); document.body.removeChild(t);
            }}
            const old = btn.innerText; btn.innerText = '✅ Copied!';
            setTimeout(() => btn.innerText = old, 1500);
        }};
        </script>
        """,
        height=45,
    )


# ---------------------------------------------------------------- stats UI

NERD_CSS = """<style>
.ns-wrap{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:28px 40px;align-items:start;margin-top:.5rem}
.ns-group-title{font-size:1.05rem;font-weight:600;margin:0 0 .35rem;display:flex;align-items:center;gap:.5rem}
.ns-group-title::before{content:"";width:.5rem;height:.5rem;border-radius:50%;background:#14b8a6}
.ns-row{border-bottom:1px solid rgba(128,128,128,.22)}
.ns-row summary{list-style:none;display:flex;align-items:baseline;gap:1rem;padding:.7rem .25rem;cursor:pointer;border-radius:6px}
.ns-row summary::-webkit-details-marker{display:none}
.ns-row summary:hover{background:rgba(20,184,166,.08)}
.ns-row summary:focus-visible{outline:2px solid #14b8a6;outline-offset:2px}
.ns-label{flex:1;opacity:.72}
.ns-value{font-weight:600;font-variant-numeric:tabular-nums;text-align:right;word-break:break-word;max-width:55%}
.ns-mark{width:.9rem;flex:none;text-align:center;opacity:.55;transition:transform .2s ease}
.ns-mark::before{content:"+"}
.ns-row[open] .ns-mark{transform:rotate(45deg);opacity:1;color:#14b8a6}
.ns-body{padding:.15rem .25rem 1rem;line-height:1.55;font-size:.92rem;opacity:.92}
.ns-row[open] .ns-body{animation:ns-open .22s ease}
.ns-body p{margin:0 0 .6rem}
.ns-note{border-left:3px solid #14b8a6;padding:.45rem .75rem;background:rgba(20,184,166,.09);border-radius:0 8px 8px 0}
.ns-note b{display:block;font-size:.78rem;opacity:.75;font-weight:600;margin-bottom:.15rem}
@keyframes ns-open{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.ns-row[open] .ns-body{animation:none}.ns-mark{transition:none}}
</style>"""


def render_nerd_stats(idx, res, sec):
    st.caption("Click any row to see what it means and how to read your number.")
    expand_all = st.toggle("Expand all explanations", key=f"expand_{idx}")

    nerd = sec["nerd"]
    groups = dict(METRIC_GROUPS)
    known = {m for ms in groups.values() for m in ms}
    leftover = [k for k in nerd if k not in known]

    if leftover:
        groups["Other"] = leftover

    blocks = []

    for title, labels in groups.items():
        rows = []

        for label in labels:
            if label not in nerd:
                continue

            desc = METRIC_INFO.get(label, "")
            note = metric_insight(label, res, sec)
            note_html = (
                f'<div class="ns-note"><b>In this recording</b>{html.escape(note)}</div>'
                if note
                else ""
            )

            rows.append(
                f'<details class="ns-row"{" open" if expand_all else ""}><summary>'
                f'<span class="ns-label">{html.escape(label)}</span>'
                f'<span class="ns-value">{html.escape(str(nerd[label]))}</span>'
                f'<span class="ns-mark" aria-hidden="true"></span></summary>'
                f'<div class="ns-body"><p>{html.escape(desc)}</p>{note_html}</div></details>'
            )

        if rows:
            blocks.append(
                f'<section><div class="ns-group-title">{html.escape(title)}</div>'
                f'{"".join(rows)}</section>'
            )

    st.markdown(
        f'<div class="ns-wrap">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


def render_result(idx, res, sec, options):
    info = res["info"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", fmt_time(info.duration))
    c2.metric("Words", sec["nerd"]["Words"])
    c3.metric("Processing time", f"{res['elapsed']:.1f}s")
    c4.metric("Speaking rate", sec["nerd"]["Speaking rate"])

    names = ["📝 Transcription", "📊 Analysis", "🤓 Stats for nerds"]

    if "kw" in sec:
        names.append("🔑 Keywords")
    if "filler" in sec:
        names.append("🗣️ Fillers & pauses")

    tabs = st.tabs(names)

    with tabs[0]:
        st.text_area(
            "Transcript",
            sec["transcript"],
            height=280,
            label_visibility="collapsed",
            key=f"ta_{idx}",
        )

        b1, b2, b3, _ = st.columns([1.3, 1, 1, 4])

        with b1:
            copy_button(sec["transcript"], "📋 Copy transcript")

        with b2:
            st.download_button(
                "⬇️ .txt",
                sec["transcript"],
                f"{res['name'].split('.')[0]}_transcript.txt",
                key=f"txt_{idx}",
            )

        with b3:
            st.download_button(
                "⬇️ .srt",
                make_srt(res["segments"]),
                f"{res['name'].split('.')[0]}_transcript.srt",
                key=f"srt_{idx}",
            )

    with tabs[1]:
        st.text(sec["analysis"])
        copy_button(sec["analysis"], "📋 Copy analysis")

        if sec["freq"]:
            df = pd.DataFrame(
                sec["freq"].most_common(options["top_n"]),
                columns=["word", "count"],
            ).set_index("word")
            st.bar_chart(df)

    with tabs[2]:
        render_nerd_stats(idx, res, sec)
        copy_button(sec["nerd_text"], "📋 Copy stats")

    tab_index = 3

    if "kw" in sec:
        with tabs[tab_index]:
            if sec["kw"]:
                kdf = pd.DataFrame(
                    sec["kw"],
                    columns=["Keyword / phrase", "Score (lower = more relevant)"],
                )
                st.dataframe(kdf, use_container_width=True, hide_index=True)
                copy_button(sec["kw_text"], "📋 Copy keywords")
            else:
                st.info("Not enough text to extract keywords.")

        tab_index += 1

    if "filler" in sec:
        f = sec["filler"]

        with tabs[tab_index]:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filler words", f["total"])
            m2.metric("Fillers / min", f"{f['per_min']:.1f}")
            m3.metric("Pauses", len(f["pauses"]))
            m4.metric("Silence in pauses", f"{f['pause_pct']:.1%}")

            left, right = st.columns(2)

            with left:
                st.markdown("**Filler words**")
                if f["counts"]:
                    fdf = pd.DataFrame(
                        f["counts"].most_common(),
                        columns=["filler", "count"],
                    ).set_index("filler")
                    st.bar_chart(fdf)
                else:
                    st.success("No filler words found 🎉")

            with right:
                st.markdown(
                    f"**Pauses ≥ {options['pause_thr']:.1f}s** "
                    f"(longest {f['longest']:.1f}s)"
                )

                if f["pauses"]:
                    pdf = pd.DataFrame(
                        [(fmt_time(ts), round(g, 2)) for ts, g in f["pauses"]],
                        columns=["At", "Length (s)"],
                    )
                    st.dataframe(
                        pdf,
                        use_container_width=True,
                        hide_index=True,
                        height=250,
                    )
                else:
                    st.success("No long pauses found")

            if f["pause_source"] == "segment" or not res["word_ts"]:
                st.caption(
                    "ℹ️ Pauses were measured between segments. Re-run **Transcribe** "
                    "with this feature enabled for word-level accuracy and "
                    "filler-friendly transcription."
                )

            st.caption(
                "Words like “like”, “so”, “right” can be genuine uses, "
                "so counts are an upper bound."
            )
            copy_button(sec["filler_text"], "📋 Copy fillers & pauses")

    st.markdown("")
    copy_button(sec["report"], "📋 Copy everything (this file)")


# ---------------------------------------------------------------- Streamlit app

st.markdown(NERD_CSS, unsafe_allow_html=True)
st.title("🎙️ Audio Transcriber")
st.caption("Local, private speech-to-text powered by faster-whisper.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_path = st.text_input("Model path", "./models/small.en")
    device = st.selectbox("Device", ["cpu", "cuda"])
    compute_type = st.selectbox("Compute type", ["int8", "float16", "float32"])

    st.divider()
    st.subheader("Display")
    top_n = st.slider("Top words to show", 5, 30, 7)
    drop_stop = st.checkbox("Ignore common words (the, and, is…)", value=False)
    timestamps = st.checkbox("Show timestamps in transcript", value=False)

    st.divider()
    st.subheader("✨ Optional features")
    use_batch = st.checkbox("📦 Batch mode (multiple files)", value=False)
    use_mic = st.checkbox("🎤 Microphone recording", value=False)
    use_kw = st.checkbox(
        "🔑 Keyword extraction",
        value=False,
        disabled=not YAKE_OK,
        help=None if YAKE_OK else "Install with: pip install yake",
    )
    use_fill = st.checkbox("🗣️ Filler words & pause detection", value=False)

    kw_ngram, kw_top = 2, 10
    if use_kw:
        kw_ngram = st.slider("Max keyword length (words)", 1, 3, 2)
        kw_top = st.slider("Number of keywords", 5, 30, 10)

    filler_sel, pause_thr, keep_fillers = DEFAULT_FILLERS, 1.0, True

    if use_fill:
        filler_sel = st.multiselect(
            "Words to count as fillers",
            list(FILLER_PATTERNS),
            DEFAULT_FILLERS,
        )
        pause_thr = st.slider(
            "Pause threshold (seconds)",
            0.3,
            3.0,
            1.0,
            0.1,
        )
        keep_fillers = st.checkbox(
            "Nudge Whisper to keep “um/uh”",
            value=True,
            help="Whisper usually drops disfluencies. A prompt with some in it makes it keep more.",
        )

options = {
    "top_n": top_n,
    "drop_stop": drop_stop,
    "timestamps": timestamps,
    "keywords": use_kw,
    "kw_ngram": kw_ngram,
    "kw_top": kw_top,
    "fillers": use_fill,
    "filler_sel": filler_sel,
    "pause_thr": pause_thr,
}

# ---- inputs
inputs = []

uploads = st.file_uploader(
    "Upload audio" + (" files" if use_batch else " file"),
    type=["m4a", "mp3", "wav", "flac", "ogg", "aac", "wma", "webm", "mp4"],
    accept_multiple_files=use_batch,
)

if uploads:
    uploads = uploads if isinstance(uploads, list) else [uploads]
    inputs += [(u.name, u.getvalue()) for u in uploads]

if use_mic:
    if hasattr(st, "audio_input"):
        rec = st.audio_input("Record from your microphone")
        if rec is not None:
            inputs.append(
                (
                    f"mic_recording_{time.strftime('%H%M%S')}.wav",
                    rec.getvalue(),
                )
            )
    else:
        st.warning(
            "Microphone recording needs a newer Streamlit: "
            "`pip install -U streamlit`"
        )

if len(inputs) == 1:
    st.audio(inputs[0][1])
elif len(inputs) > 1:
    st.info(f"{len(inputs)} audio inputs queued. Everything above will be transcribed.")


# ---- transcription
if inputs and st.button("🚀 Transcribe", type="primary"):
    cfg = {
        "model_path": model_path,
        "device": device,
        "compute_type": compute_type,
    }
    prompt = FILLER_PROMPT if (use_fill and keep_fillers) else None
    results = []

    try:
        model = get_model(model_path, device, compute_type)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    overall = st.progress(0.0, text="Queued…") if len(inputs) > 1 else None
    st.caption("Live preview")
    live = st.empty()

    for i, (name, data) in enumerate(inputs):
        label = f"[{i + 1}/{len(inputs)}] {name}" if len(inputs) > 1 else name
        file_bar = st.progress(0.0, text=f"{label}: starting…")
        tmp_path = None

        def update_progress(pct, text):
            file_bar.progress(pct, text=text)

        def update_live(text):
            live.markdown(text)

        try:
            suffix = os.path.splitext(name)[1] or ".m4a"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            segments, info, elapsed = transcribe_audio(
                tmp_path,
                model,
                progress_callback=update_progress,
                live_callback=update_live,
                label=label,
                word_ts=use_fill,
                prompt=prompt,
            )

            results.append(
                {
                    "name": name,
                    "size": len(data),
                    "segments": segments,
                    "info": info,
                    "elapsed": elapsed,
                    "cfg": cfg,
                    "word_ts": use_fill,
                }
            )

        except Exception as e:
            st.error(f"{name}: {e}")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        if overall:
            overall.progress(
                (i + 1) / len(inputs),
                text=f"{i + 1}/{len(inputs)} files done",
            )

    live.empty()
    st.session_state["results"] = results


# ---- results
results = st.session_state.get("results")

if results:
    sections = [compute_sections(r, options) for r in results]

    if len(results) > 1:
        st.subheader("📦 Batch summary")

        sdf = pd.DataFrame(
            [summary_row(r, s) for r, s in zip(results, sections)]
        )
        st.dataframe(sdf, use_container_width=True, hide_index=True)

        total_dur = sum((r["info"].duration or 0) for r in results)
        total_words = sum(s["n_words"] for s in sections)
        total_proc = sum(r["elapsed"] for r in results)

        t1, t2, t3 = st.columns(3)
        t1.metric("Total audio", fmt_time(total_dur))
        t2.metric("Total words", total_words)
        t3.metric("Total processing", f"{total_proc:.1f}s")

        d1, d2, _ = st.columns([1.3, 1.3, 4])

        with d1:
            st.download_button(
                "⬇️ Summary CSV",
                sdf.to_csv(index=False),
                "batch_summary.csv",
            )

        with d2:
            all_report = "\n\n".join(
                f"##### {r['name']} #####\n{s['report']}"
                for r, s in zip(results, sections)
            )
            copy_button(all_report, "📋 Copy all files")

        st.divider()

    for i, (res, sec) in enumerate(zip(results, sections)):
        if len(results) > 1:
            with st.expander(
                f"🎧 {res['name']}",
                expanded=(i == 0),
            ):
                render_result(i, res, sec, options)
        else:
            render_result(i, res, sec, options)
