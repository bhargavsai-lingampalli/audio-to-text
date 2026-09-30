import html
import json
import os
import re
import tempfile
import time
from collections import Counter
import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel
import streamlit.components.v1 as components

try:
    import yake
    YAKE_OK = True
except ImportError:
    YAKE_OK = False

st.set_page_config(page_title="Audio Transcriber", page_icon="🎙️", layout="wide")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "i", "you", "he", "she", "we", "they", "me", "my", "your", "our", "so", "as",
    "do", "does", "did", "not", "no", "yes", "have", "has", "had", "will", "would",
    "can", "could", "just", "from", "by", "about", "there", "then", "than", "them",
}

# label -> regex. Ambiguous ones (like, so, right, actually) can be legit words,
# so counts are an upper bound.
FILLER_PATTERNS = {
    "um / umm": r"\b(?:um+|uhm+)\b",
    "uh / uhh": r"\buh+\b",
    "er / erm": r"\ber+m*\b",
    "ah": r"\bah+\b",
    "hmm": r"\bh+m+\b",
    "like": r"\blike\b",
    "you know": r"\byou know\b",
    "i mean": r"\bi mean\b",
    "basically": r"\bbasically\b",
    "actually": r"\bactually\b",
    "literally": r"\bliterally\b",
    "kind of": r"\bkind of\b",
    "sort of": r"\bsort of\b",
    "right": r"\bright\b",
    "so": r"\bso\b",
}
DEFAULT_FILLERS = [k for k in FILLER_PATTERNS if k not in ("right", "so")]

# Whisper tends to "clean up" disfluencies; a prompt containing them nudges it to keep them.
FILLER_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."

# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading Whisper model…")
def load_model(path: str, device: str, compute_type: str) -> WhisperModel:
    try:
        return WhisperModel(path, device=device, compute_type=compute_type)
    except:
        return WhisperModel("small.en",device=device, compute_type=compute_type)


def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


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


# ---------------------------------------------------------------- core logic
def transcribe_audio(file_path, model, progress_bar, live_box, label, word_ts, prompt):
    """Transcribe with a real progress bar (segment end time / total duration)."""
    started = time.perf_counter()
    segments_iter, info = model.transcribe(
        file_path, word_timestamps=word_ts, initial_prompt=prompt
    )

    segments = []
    for seg in segments_iter:
        segments.append(seg)
        pct = min(seg.end / info.duration, 1.0) if info.duration else 0.0
        progress_bar.progress(pct, text=f"{label}: transcribing… {pct:.0%}")
        live_box.markdown("".join(s.text for s in segments)[-600:])  # live preview

    progress_bar.progress(1.0, text=f"{label}: done ✅")
    return segments, info, time.perf_counter() - started


def analyze_audio(text: str, top_n: int, drop_stopwords: bool):
    words = tokenize(text)
    freq_words = [w for w in words if w not in STOPWORDS] if drop_stopwords else words
    freq = Counter(freq_words)
    summary = (
        f"The transcribed text contains {len(words)} words, "
        f"with {len(set(words))} unique words.\n"
        f"Top {top_n} words{' (stopwords removed)' if drop_stopwords else ''}:\n"
    )
    summary += "\n".join(f"  {w}: {c}" for w, c in freq.most_common(top_n))
    return summary, freq


def build_nerd_stats(res, text):
    segments, info, elapsed, cfg = res["segments"], res["info"], res["elapsed"], res["cfg"]
    words = tokenize(text)
    duration = info.duration or 0.0
    n_words, unique = len(words), len(set(words))
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    seg_lengths = [s.end - s.start for s in segments] or [0]

    return {
        "Audio duration": f"{fmt_time(duration)} ({duration:.2f} s)",
        "Processing time": f"{elapsed:.2f} s",
        "Real-time factor (RTF)": f"{elapsed / duration:.3f}x" if duration else "n/a",
        "Speed": f"{duration / elapsed:.1f}x faster than real time" if elapsed else "n/a",
        "File name": res["name"],
        "File size": f"{res['size'] / 1024 / 1024:.2f} MB",
        "Detected language": f"{info.language} ({info.language_probability:.1%} confidence)",
        "Model path": cfg["model_path"],
        "Device / compute type": f"{cfg['device']} / {cfg['compute_type']}",
        "Word timestamps": "yes" if res["word_ts"] else "no",
        "Segments": len(segments),
        "Avg segment length": f"{sum(seg_lengths) / len(seg_lengths):.2f} s",
        "Longest segment": f"{max(seg_lengths):.2f} s",
        "Words": n_words,
        "Unique words": unique,
        "Lexical diversity": f"{unique / n_words:.1%}" if n_words else "n/a",
        "Characters": len(text),
        "Sentences (approx.)": len(sentences),
        "Avg words / sentence": f"{n_words / len(sentences):.1f}" if sentences else "n/a",
        "Avg word length": f"{sum(map(len, words)) / n_words:.2f} chars" if n_words else "n/a",
        "Longest word": max(words, key=len) if words else "n/a",
        "Speaking rate": f"{n_words / (duration / 60):.0f} words/min" if duration else "n/a",
    }


def stats_to_text(stats: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in stats.items())


def make_transcript(segments, with_timestamps: bool) -> str:
    if with_timestamps:
        return "\n".join(
            f"[{fmt_time(s.start)} → {fmt_time(s.end)}] {s.text.strip()}" for s in segments
        )
    return "\n".join(s.text.strip() for s in segments)


def make_srt(segments) -> str:
    return "\n".join(
        f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{s.text.strip()}\n"
        for i, s in enumerate(segments, 1)
    )


# ---- feature: keyword extraction (YAKE)
def extract_keywords(text: str, max_ngram: int, top: int):
    if not text.strip():
        return []
    extractor = yake.KeywordExtractor(lan="en", n=max_ngram, top=top, dedupLim=0.8)
    return extractor.extract_keywords(text)  # [(keyword, score)], lower score = better


# ---- feature: filler words
def analyze_fillers(text: str, selected: list[str]):
    lower = text.lower()
    counts = Counter()
    for label in selected:
        n = len(re.findall(FILLER_PATTERNS[label], lower))
        if n:
            counts[label] = n
    return counts


# ---- feature: pause detection
def detect_pauses(segments, threshold: float):
    """Gaps between consecutive words (or segments if no word timestamps)."""
    spans = []
    for s in segments:
        if getattr(s, "words", None):
            spans.extend((w.start, w.end) for w in s.words)
    source = "word"
    if not spans:
        spans = [(s.start, s.end) for s in segments]
        source = "segment"

    pauses = []
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        gap = next_start - prev_end
        if gap >= threshold:
            pauses.append((prev_end, gap))
    return pauses, source


# ---------------------------------------------------------------- stats for nerds UI
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

METRIC_GROUPS = {
    "Speed": ["Audio duration", "Processing time", "Real-time factor (RTF)", "Speed"],
    "File and model": [
        "File name", "File size", "Detected language", "Model path",
        "Device / compute type", "Word timestamps",
    ],
    "How the audio was split": ["Segments", "Avg segment length", "Longest segment"],
    "The text itself": [
        "Words", "Unique words", "Lexical diversity", "Characters", "Sentences (approx.)",
        "Avg words / sentence", "Avg word length", "Longest word", "Speaking rate",
    ],
}

METRIC_INFO = {
    "Audio duration": "How long the recording is. Every rate and ratio on this page is measured against this number.",
    "Processing time": "How long your computer spent turning the audio into text. The timer starts after the model is loaded, so it covers transcription only.",
    "Real-time factor (RTF)": "Processing time divided by audio duration. Below 1 means faster than real time, and 0.5 means it needed half the recording's length. Lower is better. It depends on your CPU, the model size and the compute type.",
    "Speed": "The same idea flipped: audio duration divided by processing time. A bigger number means a faster run.",
    "File name": "The name of the file you uploaded, or an auto-generated name if you recorded from the microphone.",
    "File size": "Size on disk. Formats like m4a and mp3 are compressed, so size says little about speed. Duration is what drives processing time.",
    "Detected language": "Whisper listens to the first ~30 seconds and guesses the language. Confidence is how sure it is. With an English-only model such as small.en this is mostly a sanity check.",
    "Model path": "The folder holding the Whisper model weights. Bigger models (medium, large) are more accurate but slower. Models ending in .en are English-only and a bit better at English.",
    "Device / compute type": "Where the math runs (cpu or a cuda GPU) and how precise the numbers are. int8 is compact and fast on CPU with a small accuracy cost. float16 suits GPUs. float32 is the most precise and the slowest.",
    "Word timestamps": "Whether Whisper also recorded a start and end time for every word. It's needed for accurate pause detection and adds a little processing time.",
    "Segments": "Whisper cuts the audio into chunks at natural pauses or sentence ends. Each segment becomes one line in the .srt subtitle export.",
    "Avg segment length": "The average length of those chunks in seconds. Short segments suggest choppy or fast speech with many breaks. Long ones suggest flowing speech.",
    "Longest segment": "The longest uninterrupted chunk. A very long one can mean a long monologue, or a stretch where Whisper found no natural place to break.",
    "Words": "Total words in the transcript, counted after stripping punctuation.",
    "Unique words": "How many different words appear, ignoring upper and lower case. \"The\" said ten times counts once.",
    "Lexical diversity": "Unique words divided by total words. Higher means a more varied vocabulary. Short clips score high naturally, so only compare recordings of similar length.",
    "Characters": "Total characters in the transcript, including spaces and punctuation.",
    "Sentences (approx.)": "Counted by splitting on . ! and ?, so abbreviations like \"Dr.\" or numbers like 3.5 can inflate it slightly.",
    "Avg words / sentence": "A rough feel for sentence length. Shorter is punchy and conversational. Longer is formal or complex. Speech is usually shorter than writing.",
    "Avg word length": "Average letters per word. Everyday speech sits around 4 to 5. Technical talks run higher.",
    "Longest word": "The longest single word in the transcript. It's a handy way to spot glitches such as two words merged together.",
    "Speaking rate": "Words divided by minutes of audio. Conversational English is typically 120 to 160 words per minute. It includes pauses, so it's lower than the speed at which someone actually articulates words.",
}


def metric_insight(label, res, sec):
    """A short, recording-specific reading of a metric (empty if nothing useful to say)."""
    dur = res["info"].duration or 0.0
    elapsed = res["elapsed"]
    n = sec["n_words"]
    if not dur:
        return ""
    if label == "Real-time factor (RTF)":
        rtf = elapsed / dur
        verdict = "faster than real time" if rtf < 1 else "slower than real time"
        return (f"{elapsed:.1f}s of work for {dur:.1f}s of audio, so {verdict}. "
                f"At this pace a 1-hour recording would take about {rtf * 60:.0f} minutes.")
    if label == "Speed":
        return f"One minute of audio takes about {elapsed / dur * 60:.0f} seconds to transcribe."
    if label == "Lexical diversity" and n < 200:
        return f"Only {n} words, so a high score is expected. It will drop as recordings get longer."
    if label == "Speaking rate":
        wpm = n / (dur / 60)
        if wpm < 110:
            pace = "slower than typical conversation"
        elif wpm <= 160:
            pace = "in the typical conversational range"
        else:
            pace = "faster than typical conversation"
        return f"{wpm:.0f} words per minute is {pace}."
    return ""


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
                f'<div class="ns-note"><b>In this recording</b>{html.escape(note)}</div>' if note else ""
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
                f'<section><div class="ns-group-title">{html.escape(title)}</div>{"".join(rows)}</section>'
            )
    st.markdown(f'<div class="ns-wrap">{"".join(blocks)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- per-file sections
def compute_sections(res, o):
    segments, info = res["segments"], res["info"]
    duration = info.duration or 0.0
    plain = " ".join(s.text.strip() for s in segments)
    n_words = len(tokenize(plain))

    sec = {"plain": plain, "n_words": n_words}
    sec["transcript"] = make_transcript(segments, o["timestamps"])
    sec["analysis"], sec["freq"] = analyze_audio(plain, o["top_n"], o["drop_stop"])
    sec["nerd"] = build_nerd_stats(res, plain)
    sec["nerd_text"] = stats_to_text(sec["nerd"])
    sec["extra_text"] = ""

    if o["keywords"] and YAKE_OK:
        kws = extract_keywords(plain, o["kw_ngram"], o["kw_top"])
        sec["kw"] = kws
        sec["kw_text"] = "\n".join(k for k, _ in kws)
        sec["extra_text"] += f"\n\n=== KEYWORDS ===\n{sec['kw_text']}"

    if o["fillers"]:
        counts = analyze_fillers(plain, o["filler_sel"])
        total = sum(counts.values())
        pauses, source = detect_pauses(segments, o["pause_thr"])
        total_pause = sum(g for _, g in pauses)

        sec["filler"] = {
            "counts": counts,
            "total": total,
            "per_min": total / (duration / 60) if duration else 0.0,
            "pct": total / n_words if n_words else 0.0,
            "pauses": pauses,
            "pause_source": source,
            "pause_total": total_pause,
            "pause_pct": total_pause / duration if duration else 0.0,
            "longest": max((g for _, g in pauses), default=0.0),
        }
        f = sec["filler"]
        lines = [
            f"Filler words: {total} ({f['pct']:.1%} of words, {f['per_min']:.1f}/min)",
            *[f"  {k}: {v}" for k, v in counts.most_common()],
            "",
            f"Pauses >= {o['pause_thr']:.1f}s: {len(pauses)} "
            f"(total {total_pause:.1f}s, {f['pause_pct']:.1%} of audio, longest {f['longest']:.1f}s)",
            *[f"  at {fmt_time(t)}: {g:.1f}s" for t, g in pauses],
        ]
        sec["filler_text"] = "\n".join(lines)
        sec["extra_text"] += f"\n\n=== FILLERS & PAUSES ===\n{sec['filler_text']}"

    sec["report"] = (
        f"=== TRANSCRIPTION ===\n{sec['transcript']}\n\n"
        f"=== ANALYSIS ===\n{sec['analysis']}\n\n"
        f"=== STATS ===\n{sec['nerd_text']}{sec['extra_text']}"
    )
    return sec


def summary_row(res, sec, o):
    info = res["info"]
    duration = info.duration or 0.0
    row = {
        "File": res["name"],
        "Duration (s)": round(duration, 1),
        "Words": sec["n_words"],
        "Unique words": len(set(tokenize(sec["plain"]))),
        "Words/min": round(sec["n_words"] / (duration / 60), 0) if duration else None,
        "Processing (s)": round(res["elapsed"], 1),
        "RTF": round(res["elapsed"] / duration, 3) if duration else None,
        "Language": info.language,
    }
    if "filler" in sec:
        f = sec["filler"]
        row["Fillers"] = f["total"]
        row["Fillers/min"] = round(f["per_min"], 1)
        row["Pauses"] = len(f["pauses"])
        row["Silence %"] = round(f["pause_pct"] * 100, 1)
    return row


def render_result(idx, res, sec, o):
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
            "Transcript", sec["transcript"], height=280,
            label_visibility="collapsed", key=f"ta_{idx}",
        )
        b1, b2, b3, _ = st.columns([1.3, 1, 1, 4])
        with b1:
            copy_button(sec["transcript"], "📋 Copy transcript")
        with b2:
            st.download_button("⬇️ .txt", sec["transcript"], f"{res['name'].split('.')[0]}_transcript.txt", key=f"txt_{idx}")
        with b3:
            st.download_button("⬇️ .srt", make_srt(res["segments"]), f"{res['name'].split('.')[0]}_transcript.srt", key=f"srt_{idx}")

    with tabs[1]:
        st.text(sec["analysis"])
        copy_button(sec["analysis"], "📋 Copy analysis")
        if sec["freq"]:
            df = pd.DataFrame(sec["freq"].most_common(o["top_n"]), columns=["word", "count"]).set_index("word")
            st.bar_chart(df)

    with tabs[2]:
        render_nerd_stats(idx, res, sec)
        copy_button(sec["nerd_text"], "📋 Copy stats")

    t = 3
    if "kw" in sec:
        with tabs[t]:
            if sec["kw"]:
                kdf = pd.DataFrame(sec["kw"], columns=["Keyword / phrase", "Score (lower = more relevant)"])
                st.dataframe(kdf, use_container_width=True, hide_index=True)
                copy_button(sec["kw_text"], "📋 Copy keywords")
            else:
                st.info("Not enough text to extract keywords.")
        t += 1

    if "filler" in sec:
        f = sec["filler"]
        with tabs[t]:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filler words", f["total"])
            m2.metric("Fillers / min", f"{f['per_min']:.1f}")
            m3.metric("Pauses", len(f["pauses"]))
            m4.metric("Silence in pauses", f"{f['pause_pct']:.1%}")

            left, right = st.columns(2)
            with left:
                st.markdown("**Filler words**")
                if f["counts"]:
                    fdf = pd.DataFrame(f["counts"].most_common(), columns=["filler", "count"]).set_index("filler")
                    st.bar_chart(fdf)
                else:
                    st.success("No filler words found 🎉")
            with right:
                st.markdown(f"**Pauses ≥ {o['pause_thr']:.1f}s** (longest {f['longest']:.1f}s)")
                if f["pauses"]:
                    pdf = pd.DataFrame(
                        [(fmt_time(ts), round(g, 2)) for ts, g in f["pauses"]],
                        columns=["At", "Length (s)"],
                    )
                    st.dataframe(pdf, use_container_width=True, hide_index=True, height=250)
                else:
                    st.success("No long pauses found")

            if f["pause_source"] == "segment" or not res["word_ts"]:
                st.caption(
                    "ℹ️ Pauses were measured between segments. Re-run **Transcribe** with this "
                    "feature enabled for word-level accuracy and filler-friendly transcription."
                )
            st.caption("Words like “like”, “so”, “right” can be genuine uses, so counts are an upper bound.")
            copy_button(sec["filler_text"], "📋 Copy fillers & pauses")

    st.markdown("")
    copy_button(sec["report"], "📋 Copy everything (this file)")


# ---------------------------------------------------------------- UI
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
        "🔑 Keyword extraction", value=False, disabled=not YAKE_OK,
        help=None if YAKE_OK else "Install with: pip install yake",
    )
    use_fill = st.checkbox("🗣️ Filler words & pause detection", value=False)

    kw_ngram, kw_top = 2, 10
    if use_kw:
        kw_ngram = st.slider("Max keyword length (words)", 1, 3, 2)
        kw_top = st.slider("Number of keywords", 5, 30, 10)

    filler_sel, pause_thr, keep_fillers = DEFAULT_FILLERS, 1.0, True
    if use_fill:
        filler_sel = st.multiselect("Words to count as fillers", list(FILLER_PATTERNS), DEFAULT_FILLERS)
        pause_thr = st.slider("Pause threshold (seconds)", 0.3, 3.0, 1.0, 0.1)
        keep_fillers = st.checkbox(
            "Nudge Whisper to keep “um/uh”", value=True,
            help="Whisper usually drops disfluencies. A prompt with some in it makes it keep more.",
        )

opts = {
    "top_n": top_n, "drop_stop": drop_stop, "timestamps": timestamps,
    "keywords": use_kw, "kw_ngram": kw_ngram, "kw_top": kw_top,
    "fillers": use_fill, "filler_sel": filler_sel, "pause_thr": pause_thr,
}

# ---- inputs
inputs = []  # list of (name, bytes)

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
            inputs.append((f"mic_recording_{time.strftime('%H%M%S')}.wav", rec.getvalue()))
    else:
        st.warning("Microphone recording needs a newer Streamlit: `pip install -U streamlit`")

if len(inputs) == 1:
    st.audio(inputs[0][1])
elif len(inputs) > 1:
    st.info(f"{len(inputs)} audio inputs queued. Everything above will be transcribed.")

# ---- run
if inputs and st.button("🚀 Transcribe", type="primary"):
    cfg = {"model_path": model_path, "device": device, "compute_type": compute_type}
    prompt = FILLER_PROMPT if (use_fill and keep_fillers) else None
    results = []
    try:
        model = load_model(model_path, device, compute_type)
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
        try:
            suffix = os.path.splitext(name)[1] or ".m4a"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            segments, info, elapsed = transcribe_audio(
                tmp_path, model, file_bar, live, label, use_fill, prompt
            )
            results.append({
                "name": name, "size": len(data), "segments": segments, "info": info,
                "elapsed": elapsed, "cfg": cfg, "word_ts": use_fill,
            })
        except Exception as e:
            st.error(f"{name}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        if overall:
            overall.progress((i + 1) / len(inputs), text=f"{i + 1}/{len(inputs)} files done")

    live.empty()
    st.session_state["results"] = results

results = st.session_state.get("results")
if results:
    sections = [compute_sections(r, opts) for r in results]

    if len(results) > 1:
        st.subheader("📦 Batch summary")
        sdf = pd.DataFrame([summary_row(r, s, opts) for r, s in zip(results, sections)])
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
            st.download_button("⬇️ Summary CSV", sdf.to_csv(index=False), "batch_summary.csv")
        with d2:
            all_report = "\n\n".join(
                f"##### {r['name']} #####\n{s['report']}" for r, s in zip(results, sections)
            )
            copy_button(all_report, "📋 Copy all files")
        st.divider()

    for i, (res, sec) in enumerate(zip(results, sections)):
        if len(results) > 1:
            # st.write("Debug:", "hello", res['name'], "world")
            with st.expander(f"🎧 {res['name']}", expanded=(i == 0)):
                render_result(i, res, sec, opts)
        else:
            render_result(i, res, sec, opts)