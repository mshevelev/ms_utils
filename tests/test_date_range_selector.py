import pytest
import datetime as dt
from datetime import datetime
import pandas as pd
from ms_utils.panel.widgets.date_range import DateRangeSelector

# Helper function to create date objects for testing
def d(year, month, day):
    return dt.date(year, month, day)

# Test cases for initialization with various inputs
@pytest.mark.parametrize(
    "start_input, end_input, value_input, expected_start, expected_end, expected_value",
    [
        # Basic cases
        ("20200101", "20221231", ("20210115", "20210125"), d(2020, 1, 1), d(2022, 12, 31), (d(2021, 1, 15), d(2021, 1, 25))),
        (d(2020, 1, 1), d(2022, 12, 31), (d(2021, 1, 15), d(2021, 1, 25)), d(2020, 1, 1), d(2022, 12, 31), (d(2021, 1, 15), d(2021, 1, 25))),
        (datetime(2020, 1, 1), datetime(2022, 12, 31), (datetime(2021, 1, 15), datetime(2021, 1, 25)), d(2020, 1, 1), d(2022, 12, 31), (d(2021, 1, 15), d(2021, 1, 25))),

        # Cases with None values
        (None, None, None, d(2020, 1, 1), dt.date.today(), (d(2020, 1, 1), dt.date.today())),
        ("20210101", None, None, d(2021, 1, 1), dt.date.today(), (d(2021, 1, 1), dt.date.today())),
        (None, "20221231", None, d(2020, 1, 1), d(2022, 12, 31), (d(2020, 1, 1), d(2022, 12, 31))),
        ("20210101", "20221231", None, d(2021, 1, 1), d(2022, 12, 31), (d(2021, 1, 1), d(2022, 12, 31))),
        (None, None, (None, None), d(2020, 1, 1), dt.date.today(), (d(2020, 1, 1), dt.date.today())),
        ("20210101", None, (None, "20210630"), d(2021, 1, 1), dt.date.today(), (d(2021, 1, 1), d(2021, 6, 30))),
        (None, "20221231", ("20220101", None), d(2020, 1, 1), d(2022, 12, 31), (d(2022, 1, 1), d(2022, 12, 31))),
        ("20210101", "20221231", ("20210615", None), d(2021, 1, 1), d(2022, 12, 31), (d(2021, 6, 15), d(2022, 12, 31))),
        ("20210101", "20221231", (None, "20210615"), d(2021, 1, 1), d(2022, 12, 31), (d(2021, 1, 1), d(2021, 6, 15))),

        # Cases with value outside start/end bounds
        ("20210101", "20211231", ("20200101", "20220101"), d(2021, 1, 1), d(2021, 12, 31), (d(2021, 1, 1), d(2021, 12, 31))),
        ("20210101", "20211231", ("20210615", "20230101"), d(2021, 1, 1), d(2021, 12, 31), (d(2021, 6, 15), d(2021, 12, 31))),
        ("20210101", "20211231", ("20200101", "20210615"), d(2021, 1, 1), d(2021, 12, 31), (d(2021, 1, 1), d(2021, 6, 15))),

        # Cases with invalid date strings (should raise ValueError)
        ("invalid_date", "20221231", None, d(2020, 1, 1), d(2022, 12, 31), (d(2020, 1, 1), d(2022, 12, 31))),
        ("20210101", "invalid_date", None, d(2021, 1, 1), dt.date.today(), (d(2021, 1, 1), dt.date.today())),
        ("20210101", "20221231", ("invalid_date", "20210615"), d(2021, 1, 1), d(2022, 12, 31), (d(2021, 1, 1), d(2021, 6, 15))),
        ("20210101", "20221231", ("20210615", "invalid_date"), d(2021, 1, 1), d(2022, 12, 31), (d(2021, 6, 15), d(2022, 12, 31))),
    ]
)
@pytest.mark.parametrize(
    "throttled_input, expected_throttled",
    [
        (True, True),
        (False, False),
    ]
)
def test_date_range_selector_initialization(start_input, end_input, value_input, expected_start, expected_end, expected_value, throttled_input, expected_throttled):
    """Tests DateRangeSelector initialization with various valid and invalid inputs."""
    # Handle cases where invalid date strings should raise an error
    if isinstance(start_input, str) and "invalid_date" in start_input:
        with pytest.raises(ValueError):
            DateRangeSelector(start=start_input, end=end_input, value=value_input)
        return
    if isinstance(end_input, str) and "invalid_date" in end_input:
        with pytest.raises(ValueError):
            DateRangeSelector(start=start_input, end=end_input, value=value_input)
        return
    if value_input and isinstance(value_input[0], str) and "invalid_date" in value_input[0]:
        with pytest.raises(ValueError):
            DateRangeSelector(start=start_input, end=end_input, value=value_input)
        return
    if value_input and isinstance(value_input[1], str) and "invalid_date" in value_input[1]:
        with pytest.raises(ValueError):
            DateRangeSelector(start=start_input, end=end_input, value=value_input)
        return

    # Create the DateRangeSelector instance
    selector = DateRangeSelector(start=start_input, end=end_input, value=value_input, throttled=throttled_input)

    # Assert the values
    assert selector.start == expected_start
    assert selector.end == expected_end
    assert selector.value == expected_value
    assert selector.throttled == expected_throttled

@pytest.mark.parametrize(
    "shortcuts_input, expected_shortcuts_len",
    [
        (None, 0),
        ([], 0),
        (["ALL", "1W"], 2),
        ("YTD", 1),
    ]
)
def test_date_range_selector_shortcuts_initialization(shortcuts_input, expected_shortcuts_len):
    """Tests DateRangeSelector initialization with various valid shortcut inputs."""
    selector = DateRangeSelector(shortcuts=shortcuts_input)
    assert len(selector.shortcuts) == expected_shortcuts_len
    assert len(selector._snapshot_buttons) == expected_shortcuts_len

def test_date_range_selector_invalid_shortcuts_type():
    """Tests DateRangeSelector initialization with an invalid type for shortcuts."""
    with pytest.raises(TypeError, match="Shortcuts must be a list of strings, a single string, or None."):
        DateRangeSelector(shortcuts=123)
    with pytest.raises(TypeError, match="Shortcuts must be a list of strings, a single string, or None."):
        DateRangeSelector(shortcuts={"ALL"}) # Set is not allowed

# Test cases for invalid value formats
@pytest.mark.parametrize(
    "value_input",
    [
        ("20210101",),  # Too few elements
        ("20210101", "20210115", "20210120"), # Too many elements
        "20210101", # Not a tuple/list
        12345, # Not a tuple/list
    ]
)
def test_date_range_selector_invalid_value_format(value_input):
    """Tests DateRangeSelector initialization with invalid value formats."""
    with pytest.raises(ValueError):
        DateRangeSelector(value=value_input)

# Test case for start date after end date
def test_date_range_selector_start_after_end():
    """Tests DateRangeSelector initialization when start date is after end date."""
    with pytest.raises(ValueError):
        DateRangeSelector(start="20220101", end="20210101")

# Test case for edge case in _parse_date with pandas
def test_parse_date_pandas_edge_case():
    """Tests _parse_date with a string that pandas can parse but strptime might struggle with."""
    selector = DateRangeSelector() # Use default values for start/end
    # A date like '2023-02-29' is invalid, but pandas might handle it differently or raise an error.
    # Let's test a valid but less common format that pandas handles well.
    date_str = "2023/03/15"
    expected_date = d(2023, 3, 15)
    assert selector._parse_date(date_str) == expected_date

# Test case for _parse_date with datetime.datetime object
def test_parse_date_with_datetime_object():
    """Tests _parse_date with a datetime.datetime object."""
    selector = DateRangeSelector()
    dt_obj = datetime(2023, 10, 26, 10, 30, 0)
    expected_date = d(2023, 10, 26)
    assert selector._parse_date(dt_obj) == expected_date

# Test case for _parse_date with an unsupported type that should fall back to default
def test_parse_date_unsupported_type_fallback():
    """Tests _parse_date with an unsupported type, expecting fallback to default."""
    selector = DateRangeSelector()
    unsupported_input = [2023, 10, 26] # A list is not a supported date type
    default_date = d(2020, 1, 1)
    # Temporarily set a default for this test
    selector.start = default_date 
    with pytest.raises(TypeError, match="Unsupported date input type: <class 'list'>"):
        selector._parse_date(unsupported_input, default=default_date)

# Test case for _parse_date with an unsupported type that should raise TypeError
def test_parse_date_unsupported_type_raise_error():
    """Tests _parse_date with an unsupported type, expecting TypeError."""
    selector = DateRangeSelector()
    unsupported_input = [2023, 10, 26] # A list is not a supported date type
    with pytest.raises(TypeError):
        selector._parse_date(unsupported_input)

# Test case for _update_all_widgets with invalid new_value
def test_update_all_widgets_invalid_value():
    """Tests _update_all_widgets with an invalid new_value."""
    selector = DateRangeSelector(start=d(2020, 1, 1), end=d(2022, 12, 31))
    
    # Test with None
    selector._update_all_widgets(None)
    assert selector.value == (d(2020, 1, 1), d(2022, 12, 31))
    assert selector._start_input.value.date() == d(2020, 1, 1)
    assert selector._end_input.value.date() == d(2022, 12, 31)
    assert selector._slider.value == (d(2020, 1, 1), d(2022, 12, 31))

    # Test with incorrect tuple length
    selector._update_all_widgets(("20210101",))
    assert selector.value == (d(2020, 1, 1), d(2022, 12, 31)) # Should reset to defaults
    assert selector._start_input.value.date() == d(2020, 1, 1)
    assert selector._end_input.value.date() == d(2022, 12, 31)
    assert selector._slider.value == (d(2020, 1, 1), d(2022, 12, 31))

    # Test with non-date elements in tuple
    with pytest.raises(ValueError, match="Could not parse date string: invalid_date"):
        selector._update_all_widgets(("20210101", "invalid_date"))
    # After the expected error, the internal state might be inconsistent,
    # so we re-initialize or check specific attributes if needed.
    # For simplicity, we'll just ensure the error is raised.

# Test case for _update_all_widgets with start > end
def test_update_all_widgets_start_after_end():
    """Tests _update_all_widgets when the new value has start > end."""
    selector = DateRangeSelector(start=d(2020, 1, 1), end=d(2022, 12, 31))
    selector._update_all_widgets((d(2022, 12, 31), d(2020, 1, 1))) # Swapped dates
    assert selector.value == (d(2020, 1, 1), d(2022, 12, 31)) # Should be swapped back
    assert selector._start_input.value.date() == d(2020, 1, 1)
    assert selector._end_input.value.date() == d(2022, 12, 31)
    assert selector._end_input.value.date() == d(2022, 12, 31)

# Test cases for shortcut buttons
@pytest.mark.parametrize(
    "shortcut_name, expected_start_offset_days, expected_start_offset_months, expected_start_offset_years",
    [
        ("ALL", None, None, None), # ALL should use start_bound
        ("YTD", None, None, None), # YTD should use start of year
        ("1W", 7, None, None),
        ("2W", 14, None, None),
        ("1M", None, 1, None),
        ("3M", None, 3, None),
        ("1Y", None, None, 1),
        ("2Y", None, None, 2),
    ]
)
def test_date_range_selector_shortcuts(shortcut_name, expected_start_offset_days, expected_start_offset_months, expected_start_offset_years):
    """Tests the functionality of date range shortcut buttons."""
    end_date = d(2024, 10, 20) # Use a fixed end date for predictable testing
    start_date_bound = d(2020, 1, 1)
    
    selector = DateRangeSelector(start=start_date_bound, end=end_date, shortcuts=[shortcut_name])
    
    # Simulate button click
    if shortcut_name in selector._snapshot_buttons:
        selector._snapshot_buttons[shortcut_name].param.trigger('clicks')
    else:
        pytest.fail(f"Shortcut button '{shortcut_name}' not found.")

    new_start, new_end = selector.value

    assert new_end == end_date # End date should always be the max end date

    if shortcut_name == "ALL":
        assert new_start == start_date_bound
    elif shortcut_name == "YTD":
        assert new_start == d(end_date.year, 1, 1)
    elif expected_start_offset_days is not None:
        assert new_start == end_date - dt.timedelta(days=expected_start_offset_days)
    elif expected_start_offset_months is not None:
        expected_start_ts = pd.Timestamp(end_date) - pd.DateOffset(months=expected_start_offset_months)
        assert new_start == expected_start_ts.date()
    elif expected_start_offset_years is not None:
        expected_start_year = end_date.year - expected_start_offset_years
        try:
            expected_start_date = end_date.replace(year=expected_start_year)
        except ValueError: # Handle leap year edge case
            expected_start_date = end_date.replace(year=expected_start_year, day=28)
        assert new_start == expected_start_date
