"""The steps-1-6 reuse cache in generate_sofa.master_sofa must be invisible:
mastering the uneq then dfeq variant through a shared cache has to give
exactly the same files as two independent master_sofa calls."""
import numpy as np
import pytest
import sofar as sf

from core.scripts.generate_sofa import master_sofa

CASES = {
    "plain": {},
    "lf_harmonize_ild": dict(lf_extend=True, harmonize_radius=True,
                             target_radius=1.2, ild_fix=True),
}


def _read(path):
    s = sf.read_sofa(str(path), verbose=False)
    return np.asarray(s.Data_IR), np.asarray(s.SourcePosition), float(s.Data_SamplingRate)


@pytest.mark.parametrize("name", list(CASES))
def test_cached_variants_equal_independent_calls(name, synthetic_sofa, sofa_args,
                                                 tmp_path, capsys):
    synthetic_sofa = synthetic_sofa()
    args = sofa_args(**CASES[name])
    ref, cached = {}, {}
    for dfeq in (False, True):
        p = tmp_path / f"ref_{dfeq}.sofa"
        master_sofa(synthetic_sofa, 44100, dfeq, str(p), args)
        ref[dfeq] = _read(p)
    ref_out = capsys.readouterr().out

    cache = {}
    for dfeq in (False, True):
        p = tmp_path / f"cached_{dfeq}.sofa"
        master_sofa(synthetic_sofa, 44100, dfeq, str(p), args, _prep_cache=cache)
        cached[dfeq] = _read(p)
    cached_out = capsys.readouterr().out

    for dfeq in (False, True):
        for a, b in zip(ref[dfeq], cached[dfeq]):
            assert np.array_equal(a, b)
    # reuse must not change the printed step lines either (bar the file names)
    norm = lambda s: s.replace("cached_", "ref_")
    assert norm(cached_out) == ref_out

    # a different samplerate invalidates the entry; the source is never mutated
    before = np.array(synthetic_sofa.Data_IR)
    p = tmp_path / "fs48.sofa"
    master_sofa(synthetic_sofa, 48000, True, str(p), args, _prep_cache=cache)
    assert cache["fs"] == 48000
    assert np.array_equal(before, synthetic_sofa.Data_IR)
