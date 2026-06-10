from agentAPI.tools import run_bash, BashResult, _normalize, REGISTRY


# --- run_bash: real subprocess, kept to fast deterministic commands ---------

def test_run_bash_captures_stdout_and_clean_exit():
    result = run_bash("echo hi")
    assert result.returncode == 0, f"got {result.returncode!r}"
    assert result.stdout == "hi\n", f"got {result.stdout!r}"
    assert result.stderr == "", f"got {result.stderr!r}"
    assert result.timeout is False, f"got {result.timeout!r}"

def test_run_bash_reports_nonzero_exit():
    result = run_bash("exit 3")
    assert result.returncode == 3, f"got {result.returncode!r}"

def test_run_bash_captures_stderr_separately():
    result = run_bash("echo oops >&2")
    assert result.stdout == "", f"got {result.stdout!r}"
    assert result.stderr == "oops\n", f"got {result.stderr!r}"

def test_run_bash_uses_bash_not_sh():
    # [[ ]] is a bash builtin; would error under /bin/sh (dash). Proves the
    # deliberate ["bash", "-c", ...] + shell=False choice.
    result = run_bash("[[ 1 == 1 ]] && echo yes")
    assert result.returncode == 0, f"got {result.returncode!r}"
    assert result.stdout == "yes\n", f"got {result.stdout!r}"

def test_run_bash_closes_stdin_so_readers_do_not_hang():
    # `cat` with no args reads stdin; stdin=DEVNULL gives it immediate EOF.
    # Without it a stdin-reader would block until the 10s timeout fires.
    result = run_bash("cat")
    assert result.returncode == 0, f"got {result.returncode!r}"
    assert result.stdout == "", f"got {result.stdout!r}"
    assert result.timeout is False, f"got {result.timeout!r}"


# --- _normalize: the bytes|str|None handling the timeout branch leans on -----

def test_normalize_coerces_each_input_kind_to_str():
    assert _normalize(None) == "", "None should normalize to empty string"
    assert _normalize("already text") == "already text"
    assert _normalize(b"raw bytes") == "raw bytes"


# --- BashResult.render: the model-facing projection --------------------------

def test_render_shows_status_and_stdout_no_truncation_under_limit():
    rendered = BashResult(0, "hi\n", "").render()
    assert rendered == "exit 0\nstdout:\nhi\n", f"got {rendered!r}"
    assert "truncated" not in rendered

def test_render_marks_empty_stdout_explicitly():
    # A blank string is ambiguous (silent success vs broken tool); the marker
    # disambiguates for the model.
    rendered = BashResult(0, "", "").render()
    assert rendered == "exit 0\nstdout:\n(no output)", f"got {rendered!r}"

def test_render_includes_stderr_section_only_when_present():
    rendered = BashResult(1, "", "boom\n").render()
    assert rendered == "exit 1\nstdout:\n(no output)\nstderr:\nboom\n", \
        f"got {rendered!r}"

def test_render_notes_timeout():
    rendered = BashResult(-9, "partial", "", timeout=True).render()
    assert rendered == \
        "exit -9\ntimed out, output may be incomplete\nstdout:\npartial", \
        f"got {rendered!r}"

def test_render_truncates_long_output_with_a_labeled_marker():
    # 300 chars, cap 200 -> keep 100 head + 100 tail, drop 100, say so.
    rendered = BashResult(0, "A" * 300, "").render(max_length=200)
    assert "…[truncated 100 chars]…" in rendered, f"got {rendered!r}"
    assert rendered.count("A") == 200, \
        f"expected 100 head + 100 tail kept, got {rendered.count('A')}"


# --- REGISTRY: the anti-corruption boundary (func returns str, not BashResult)

def test_registry_run_bash_func_returns_rendered_string():
    out = REGISTRY["run_bash"].func("echo hi")
    assert isinstance(out, str), f"expected str, got {type(out)!r}"
    assert "exit 0" in out and "hi" in out, f"got {out!r}"
