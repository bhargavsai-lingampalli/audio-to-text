import re
import time
from collections import Counter

import pandas as pd
from faster_whisper import WhisperModel

try:
    import yake
    YAKE_OK = True
except ImportError:
    YAKE_OK = False


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


def load_model(path: str, device: str, compute_type: str) -> WhisperModel:
    """Load the requested Whisper model, falling back to small.en if needed."""
    try:
        return WhisperModel(path, device=device, compute_type=compute_type)
    except Exception:
        return WhisperModel("small.en", device=device, compute_type=compute_type)


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


def parse_ignore_words(raw: str) -> set[str]:
    """Turn 'okay, yeah\nbasically' into {'okay', 'yeah', 'basically'}."""
    return set(tokenize(raw or ""))


def transcribe_audio(
    file_path,
    model,
    progress_callback=None,
    live_callback=None,
    label="",
    word_ts=False,
    prompt=None,
):
    """Transcribe audio.

    UI-specific progress/live-preview behavior is provided through callbacks.
    """
    started = time.perf_counter()

    segments_iter, info = model.transcribe(
        file_path,
        word_timestamps=word_ts,
        initial_prompt=prompt,
    )

    segments = []
    for seg in segments_iter:
        segments.append(seg)

        pct = min(seg.end / info.duration, 1.0) if info.duration else 0.0

        if progress_callback:
            progress_callback(pct, f"{label}: transcribing… {pct:.0%}")

        if live_callback:
            live_callback("".join(s.text for s in segments)[-600:])

    if progress_callback:
        progress_callback(1.0, f"{label}: done ✅")

    return segments, info, time.perf_counter() - started


def analyze_audio(text: str, top_n: int, drop_stopwords: bool, ignore_words=None):
    words = tokenize(text)
    ignore = set(ignore_words or ())

    freq_words = [w for w in words if w not in STOPWORDS] if drop_stopwords else list(words)
    if ignore:
        freq_words = [w for w in freq_words if w not in ignore]
    freq = Counter(freq_words)

    notes = []
    if drop_stopwords:
        notes.append("stopwords removed")
    if ignore:
        notes.append("ignoring: " + ", ".join(sorted(ignore)))
    note = f" ({'; '.join(notes)})" if notes else ""

    summary = (
        f"The transcribed text contains {len(words)} words, "
        f"with {len(set(words))} unique words.\n"
        f"Top {top_n} words{note}:\n"
    )
    summary += "\n".join(f"  {w}: {c}" for w, c in freq.most_common(top_n))

    return summary, freq


def build_nerd_stats(res, text):
    segments, info, elapsed, cfg = (
        res["segments"],
        res["info"],
        res["elapsed"],
        res["cfg"],
    )

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
            f"[{fmt_time(s.start)} → {fmt_time(s.end)}] {s.text.strip()}"
            for s in segments
        )
    return "\n".join(s.text.strip() for s in segments)


def make_srt(segments) -> str:
    return "\n".join(
        f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{s.text.strip()}\n"
        for i, s in enumerate(segments, 1)
    )


def extract_keywords(text: str, max_ngram: int, top: int):
    if not text.strip() or not YAKE_OK:
        return []

    extractor = yake.KeywordExtractor(
        lan="en",
        n=max_ngram,
        top=top,
        dedupLim=0.8,
    )
    return extractor.extract_keywords(text)


def analyze_fillers(text: str, selected: list[str]):
    lower = text.lower()
    counts = Counter()

    for label in selected:
        n = len(re.findall(FILLER_PATTERNS[label], lower))
        if n:
            counts[label] = n

    return counts


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


METRIC_GROUPS = {
    "Speed": ["Audio duration", "Processing time", "Real-time factor (RTF)", "Speed"],
    "File and model": [
        "File name",
        "File size",
        "Detected language",
        "Model path",
        "Device / compute type",
        "Word timestamps",
    ],
    "How the audio was split": ["Segments", "Avg segment length", "Longest segment"],
    "The text itself": [
        "Words",
        "Unique words",
        "Lexical diversity",
        "Characters",
        "Sentences (approx.)",
        "Avg words / sentence",
        "Avg word length",
        "Longest word",
        "Speaking rate",
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
    """A short, recording-specific reading of a metric."""
    dur = res["info"].duration or 0.0
    elapsed = res["elapsed"]
    n = sec["n_words"]

    if not dur:
        return ""

    if label == "Real-time factor (RTF)":
        rtf = elapsed / dur
        verdict = "faster than real time" if rtf < 1 else "slower than real time"
        return (
            f"{elapsed:.1f}s of work for {dur:.1f}s of audio, so {verdict}. "
            f"At this pace a 1-hour recording would take about {rtf * 60:.0f} minutes."
        )

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


def build_stats_csv(nerd: dict, filler: dict | None = None, pause_thr: float | None = None) -> str:
    """Stats as CSV text with Group, Metric, Value columns."""
    rows = []
    written = set()

    for group, labels in METRIC_GROUPS.items():
        for label in labels:
            if label in nerd:
                rows.append((group, label, nerd[label]))
                written.add(label)

    rows += [("Other", k, v) for k, v in nerd.items() if k not in written]

    if filler:
        g = "Fillers and pauses"
        rows += [
            (g, "Filler words", filler["total"]),
            (g, "Fillers per minute", round(filler["per_min"], 2)),
            (g, "Fillers as % of words", f"{filler['pct']:.1%}"),
            *[(g, f"Filler: {k}", n) for k, n in filler["counts"].most_common()],
        ]
        if pause_thr is not None:
            rows.append((g, "Pause threshold (s)", pause_thr))
        rows += [
            (g, "Pauses", len(filler["pauses"])),
            (g, "Total pause time (s)", round(filler["pause_total"], 2)),
            (g, "Silence in pauses", f"{filler['pause_pct']:.1%}"),
            (g, "Longest pause (s)", round(filler["longest"], 2)),
        ]

    return pd.DataFrame(rows, columns=["Group", "Metric", "Value"]).to_csv(index=False)


def compute_sections(res, options):
    segments, info = res["segments"], res["info"]
    duration = info.duration or 0.0
    plain = " ".join(s.text.strip() for s in segments)
    n_words = len(tokenize(plain))

    sec = {"plain": plain, "n_words": n_words}
    sec["transcript"] = make_transcript(segments, options["timestamps"])
    sec["analysis"], sec["freq"] = analyze_audio(
        plain,
        options["top_n"],
        options["drop_stop"],
        options.get("ignore_words"),
    )
    sec["nerd"] = build_nerd_stats(res, plain)
    sec["nerd_text"] = stats_to_text(sec["nerd"])
    sec["extra_text"] = ""

    if options["keywords"] and YAKE_OK:
        kws = extract_keywords(plain, options["kw_ngram"], options["kw_top"])
        sec["kw"] = kws
        sec["kw_text"] = "\n".join(k for k, _ in kws)
        sec["extra_text"] += f"\n\n=== KEYWORDS ===\n{sec['kw_text']}"

    if options["fillers"]:
        counts = analyze_fillers(plain, options["filler_sel"])
        total = sum(counts.values())
        pauses, source = detect_pauses(segments, options["pause_thr"])
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
            f"Pauses >= {options['pause_thr']:.1f}s: {len(pauses)} "
            f"(total {total_pause:.1f}s, {f['pause_pct']:.1%} of audio, longest {f['longest']:.1f}s)",
            *[f"  at {fmt_time(t)}: {g:.1f}s" for t, g in pauses],
        ]

        sec["filler_text"] = "\n".join(lines)
        sec["extra_text"] += f"\n\n=== FILLERS & PAUSES ===\n{sec['filler_text']}"

    sec["stats_csv"] = build_stats_csv(
        sec["nerd"], sec.get("filler"), options.get("pause_thr")
    )

    sec["report"] = (
        f"=== TRANSCRIPTION ===\n{sec['transcript']}\n\n"
        f"=== ANALYSIS ===\n{sec['analysis']}\n\n"
        f"=== STATS ===\n{sec['nerd_text']}{sec['extra_text']}"
    )

    return sec


def summary_row(res, sec):
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