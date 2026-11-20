import numpy as np

from clipper import captions, highlight, refine, segment
from clipper.audio import interest_curve


def W(text, start, end):
    return {"w": text, "start": start, "end": end, "p": 0.9}


def sample_words():
    return [W("Halo", 0.0, 0.4), W("semua.", 0.5, 1.0),
            W("Hari", 1.2, 1.5), W("ini", 1.6, 1.8), W("seru", 1.9, 2.3),
            W("banget", 3.5, 4.0), W("kan?", 4.1, 4.4)]


def test_segment_by_punct_and_gap():
    s = segment.build_sentences(sample_words(), gap=0.6)
    assert [x["text"] for x in s] == ["Halo semua.", "Hari ini seru", "banget kan?"]
    assert [x["id"] for x in s] == [1, 2, 3]


def test_windows_cover_all():
    sents = [{"id": i + 1, "start": i * 10.0, "end": i * 10.0 + 8, "text": "x"} for i in range(60)]
    wins = segment.make_windows(sents, 240, 45)
    covered = {i for w in wins for i in w}
    assert covered == set(range(60))


def test_nms_removes_overlap():
    c = [{"start": 0, "end": 30, "score": 0.9}, {"start": 20, "end": 50, "score": 0.8},
         {"start": 60, "end": 90, "score": 0.5}]
    kept = highlight.nms(c)
    assert [k["score"] for k in kept] == [0.9, 0.5]


def test_final_score_renormalizes_missing_heatmap():
    w = {"llm": 0.5, "audio_energy": 0.5, "heatmap": 0.5}
    s = highlight.final_score(1.0, {"audio_energy": 0.0, "heatmap": None}, w)
    assert abs(s - 0.5) < 1e-9


def test_fit_and_ending():
    sents = [{"id": i + 1, "start": i * 5.0, "end": i * 5.0 + 4.0,
              "text": "kalimat." if i % 2 else "kalimat"} for i in range(30)]
    fit = highlight.fit_duration(sents, 1, 30, 20, 40)
    assert fit and 20 <= sents[fit[1] - 1]["end"] - sents[fit[0] - 1]["start"] <= 40
    e = highlight.adjust_ending(sents, 1, 8, 20, 60, 3)
    assert sents[e - 1]["text"].endswith(".")


def test_cosine_dedup():
    assert highlight.cosine("aku suka kopi hitam", "aku suka kopi hitam") > 0.99
    assert highlight.cosine("aku suka kopi", "dia benci teh manis") == 0.0


def test_fuzzy_glossary_keeps_count_and_punct():
    words = [W("Halo", 0, 1), W("Jokowy,", 1, 2), W("apa", 2, 3)]
    out = refine.fuzzy_fix(words, ["Jokowi"])
    assert len(out) == 3 and out[1]["w"] == "Jokowi," and out[1]["start"] == 1


def test_ass_generation_and_mapping():
    words = sample_words()
    mapped = captions.map_words(words, [(1.2, 2.3), (0.0, 1.0)])
    assert mapped[0]["start"] == 0 and abs(mapped[3]["start"] - 1.1) < 1e-6
    ass = captions.build_ass(words, {"preset": "tiktok", "words_per_line": 2})
    assert "[Events]" in ass and ass.count("Dialogue:") == len(words)
    assert "&H00" in ass


def test_interest_curve_drops_missing():
    sig = {"n": 4, "audio_energy": [1, 1, 0, 0], "laughter": [0, 0, 0, 0],
           "speech_rate": [0, 0, 0, 0], "heatmap": None}
    c = interest_curve(sig, {"audio_energy": 0.5, "heatmap": 0.5})
    assert list(np.round(c, 2)) == [1, 1, 0, 0]
