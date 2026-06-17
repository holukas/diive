"""
TEST_EVENTS: events domain model + flag/overlay helpers
=======================================================

Run: pytest tests/test_events.py -v
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from diive.events import (
    Event,
    event_to_flag,
    make_event_flag_name,
    overlay_events,
)


@pytest.fixture
def index():
    return pd.date_range("2021-01-01", periods=48 * 30, freq="30min")


def test_instant_event_is_not_range():
    ev = Event("Fert", "2021-01-05")
    assert not ev.is_range
    assert ev.end is None
    assert ev.duration == pd.Timedelta(0)


def test_period_event_duration_and_validation():
    ev = Event("Graze", "2021-01-05", "2021-01-07")
    assert ev.is_range
    assert ev.duration == pd.Timedelta(days=2)
    with pytest.raises(ValueError):
        Event("Bad", "2021-01-07", "2021-01-05")  # end before start


def test_flag_name_sanitization():
    assert make_event_flag_name("Fertilization 1") == "EVENT_Fertilization_1"
    assert Event("Harvest A", "2021-01-01").flag_name == "EVENT_Harvest_A"


def test_instant_flag_marks_one_record(index):
    ev = Event("Fert", "2021-01-05 00:10")  # falls between two 30-min records
    flag = event_to_flag(ev, index)
    assert flag.sum() == 1
    assert flag.dtype == "int64"
    assert flag.name == "EVENT_Fert"
    # The single marked record is the nearest one to the instant.
    marked = flag[flag == 1].index[0]
    assert abs(marked - pd.Timestamp("2021-01-05 00:10")) <= pd.Timedelta("15min")


def test_period_flag_is_inclusive(index):
    ev = Event("Graze", "2021-01-05", "2021-01-06")  # 24h inclusive
    flag = event_to_flag(ev, index)
    assert set(flag.unique()) <= {0, 1}
    # 48 half-hours in a day, plus the inclusive end record at 00:00 => 49.
    assert flag.sum() == 49
    assert flag.loc["2021-01-05 00:00"] == 1
    assert flag.loc["2021-01-06 00:00"] == 1
    assert flag.loc["2021-01-06 00:30"] == 0


def test_empty_index_returns_empty_flag():
    flag = event_to_flag(Event("X", "2021-01-01"), pd.DatetimeIndex([]))
    assert len(flag) == 0


def test_color_resolution():
    assert Event("A", "2021-01-01", category="category1").resolved_color() == "#66BB6A"
    assert Event("B", "2021-01-01", color="#123456").resolved_color() == "#123456"
    # Unknown category cycles through the fallback palette by index.
    c0 = Event("C", "2021-01-01").resolved_color(0)
    c1 = Event("D", "2021-01-01").resolved_color(1)
    assert c0 != c1


def test_roundtrip_serialization():
    ev = Event("Graze", "2021-01-05 06:00", "2021-01-07", category="category3",
               description="cows", color="#abcdef")
    assert Event.from_dict(ev.to_dict()).to_dict() == ev.to_dict()


def test_overlay_adds_artists_without_rescaling(index):
    fig, ax = plt.subplots()
    ax.plot(index, range(len(index)))
    xlim_before = ax.get_xlim()
    n_lines_before = len(ax.lines)
    overlay_events(ax, [Event("Fert", "2021-01-05"),
                        Event("Graze", "2021-01-10", "2021-01-12")], axis="x")
    assert len(ax.lines) > n_lines_before  # added the event lines
    assert ax.get_xlim() == xlim_before    # didn't rescale the data axes
    plt.close(fig)


def test_overlay_y_axis_for_heatmap(index):
    fig, ax = plt.subplots()
    overlay_events(ax, [Event("Fert", "2021-01-05")], axis="y", show_labels=False)
    assert len(ax.lines) == 1
    plt.close(fig)
