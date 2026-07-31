from __future__ import annotations

import io
import time

from sub_align.timing import Timings


def test_timings_prints_steps_and_total():
    buf = io.StringIO()
    timings = Timings(enabled=True, stream=buf)
    with timings.step("transcribe"):
        time.sleep(0.01)
    with timings.step("force_align"):
        time.sleep(0.01)
    total = timings.finish()

    text = buf.getvalue()
    assert "[timing] transcribe:" in text
    assert "[timing] force_align:" in text
    assert "[timing] total:" in text
    assert len(timings.records) == 3
    assert timings.records[0].name == "transcribe"
    assert timings.records[0].seconds >= 0.01
    assert total >= 0.02


def test_timings_disabled_is_silent():
    buf = io.StringIO()
    timings = Timings(enabled=False, stream=buf)
    with timings.step("transcribe"):
        pass
    timings.finish()

    assert buf.getvalue() == ""
    assert timings.records == []
