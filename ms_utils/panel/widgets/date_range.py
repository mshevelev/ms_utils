import panel as pn
import param
import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import re

pn.extension()


class DateRangeSelector(pn.viewable.Viewer):
    """
    A custom date range selector widget that provides multiple ways to select dates:
    1. Manual input via date pickers.
    2. Date range slider.
    3. Dynamic shortcut buttons based on the `shortcuts` parameter.

    The `shortcuts` parameter accepts a list of strings, where each string defines a
    predefined date range. Supported formats include:
    - "ALL": Selects the entire range from `start` to `end`.
    - "YTD": Selects the year-to-date range (from January 1st of the current year to `end`).
    - "1W", "2W", etc.: Selects the most recent week(s) up to `end`.
    - "1M", "2M", etc.: Selects the most recent month(s) up to `end`.
    - "1Y", "2Y", etc.: Selects the most recent year(s) up to `end`.
    If `shortcuts` is None or an empty list, no shortcut buttons will be displayed.

    The `custom_shortcuts` parameter accepts a dictionary where keys are names for custom
    periods and values are tuples `(start_date, end_date)`. Each element in the tuple
    can be a string, date, datetime, or None, and will be interpreted similarly to the
    `value` parameter. These custom ranges should ideally fall within the `start` and `end`
    bounds of the selector. For each key, a red button will be created and displayed
    in a row below the standard shortcut buttons. If `custom_shortcuts` is None or an
    empty dictionary, no such buttons will be created, and the row will not be displayed.
    """

    # Define the value parameter as a tuple of dates
    # Allow None or tuples containing None or date-like objects
    value = param.Tuple(default=None, length=2, doc="Selected date range (start_date, end_date)")

    # Min and max bounds for the date range
    # Allow None and datetime.date objects
    start = param.Date(default=None, doc="Minimum selectable date")
    end = param.Date(default=None, doc="Maximum selectable date")
    shortcuts = param.List(
        default=["ALL", "YTD", "1W", "1M", "1Y"], allow_None=True, doc="List of shortcut buttons to display"
    )
    custom_shortcuts = param.Dict(default=None, allow_None=True, doc="Dictionary of custom date range shortcuts")
    name = param.String(default="Date Range", doc="Name of the date range selector, used as the card title.")
    throttled = param.Boolean(default=True, doc="Whether the DateRangeSlider should be throttled.")

    def __init__(
        self,
        start=None,
        end=None,
        value=None,
        shortcuts=None,
        custom_shortcuts=None,
        name=None,
        throttled=False,
        **params,
    ):
        # Parse and set default start date
        parsed_start = self._parse_date(start, default=dt.date(2020, 1, 1))
        # Parse and set default end date
        parsed_end = self._parse_date(end, default=dt.date.today())

        # Set shortcuts, ensuring it's a list or None
        if shortcuts is None:
            self.shortcuts = []
        elif isinstance(shortcuts, str):
            self.shortcuts = [shortcuts]
        elif isinstance(shortcuts, (list, tuple)):
            self.shortcuts = list(shortcuts)
        else:
            raise TypeError("Shortcuts must be a list of strings, a single string, or None.")

        # Set custom shortcuts, ensuring it's a dict or None
        if custom_shortcuts is None:
            self.custom_shortcuts = {}
        elif isinstance(custom_shortcuts, dict):
            self.custom_shortcuts = custom_shortcuts
        else:
            raise TypeError("Custom shortcuts must be a dictionary or None.")

        # Parse and set initial value
        if value is None:
            # If value is None, use the parsed start and end dates
            parsed_value = (parsed_start, parsed_end)
        else:
            # Ensure value is a tuple of two elements
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("Value must be a tuple or list of two elements.")

            parsed_value_start = self._parse_date(value[0])
            parsed_value_end = self._parse_date(value[1])

            # Handle None values in the parsed tuple by using start/end defaults
            if parsed_value_start is None:
                parsed_value_start = parsed_start
            if parsed_value_end is None:
                parsed_value_end = parsed_end

            parsed_value = (parsed_value_start, parsed_value_end)

        # Ensure start <= end for the initial setup
        if parsed_start and parsed_end and parsed_start > parsed_end:
            raise ValueError("Start date cannot be after end date.")

        # Ensure value dates are within start and end bounds if provided
        if parsed_value and parsed_start and parsed_value[0] is not None and parsed_value[0] < parsed_start:
            parsed_value = (parsed_start, parsed_value[1])
        if parsed_value and parsed_end and parsed_value[1] is not None and parsed_value[1] > parsed_end:
            parsed_value = (parsed_value[0], parsed_end)

        # If after parsing, value is still None, set it to the range
        if parsed_value is None or (parsed_value[0] is None and parsed_value[1] is None):
            parsed_value = (parsed_start, parsed_end)

        super().__init__(
            start=parsed_start,
            end=parsed_end,
            value=parsed_value,
            shortcuts=self.shortcuts,
            custom_shortcuts=self.custom_shortcuts,
            name=name if name is not None else self.name,
            throttled=throttled,
            **params,
        )

        # Create the date input widgets
        self._start_input = pn.widgets.DatetimeInput(
            name="From:",
            value=datetime.combine(self.value[0], datetime.min.time()) if self.value[0] else None,
            start=datetime.combine(self.start, datetime.min.time()) if self.start else None,
            end=datetime.combine(self.end, datetime.min.time()) if self.end else None,
            format="%Y-%m-%d",
            placeholder="YYYY-MM-DD",
            width=100,
        )

        self._end_input = pn.widgets.DatetimeInput(
            name="To:",
            value=datetime.combine(self.value[1], datetime.min.time()) if self.value[1] else None,
            start=datetime.combine(self.start, datetime.min.time()) if self.start else None,
            end=datetime.combine(self.end, datetime.min.time()) if self.end else None,
            format="%Y-%m-%d",
            placeholder="YYYY-MM-DD",
            width=100,
        )

        # Create the date range slider
        self._slider = pn.widgets.DateRangeSlider(
            name="",
            start=self.start,
            end=self.end,
            value=self.value,
            step=1,
            width=250,
        )

        self._snapshot_buttons = {}
        for shortcut_name in self.shortcuts:
            self._snapshot_buttons[shortcut_name] = pn.widgets.Button(
                name=shortcut_name, button_type="primary", width=50, height=35, margin=(5, 2)
            )

        self._custom_shortcut_buttons = {}
        for name, (start_val, end_val) in self.custom_shortcuts.items():
            self._custom_shortcut_buttons[name] = pn.widgets.Button(
                name=name, button_type="danger", min_width=50, height=35, margin=(5, 2)
            )

        self._setup_callbacks()

    def _parse_date(self, date_input, default=None):
        """
        Parses a date input, which can be a string, datetime.date, datetime.datetime, or None.
        Returns a datetime.date object or None if input is invalid or None.
        """
        if date_input is None:
            return default

        if isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, dt.date):
            return date_input
        elif isinstance(date_input, str):
            try:
                # Try parsing with the expected format first
                return datetime.strptime(date_input, "%Y%m%d").date()
            except ValueError:
                try:
                    # Fallback to pandas for more flexible parsing
                    return pd.to_datetime(date_input).date()
                except ValueError:
                    # If still not parsable, raise error
                    raise ValueError(f"Could not parse date string: {date_input}")
        else:
            # If it's not None, datetime, date, or string, try pandas
            try:
                return pd.to_datetime(date_input).date()
            except (ValueError, TypeError):
                # If still not parsable, raise error
                raise TypeError(f"Unsupported date input type: {type(date_input)}")

    def _setup_callbacks(self):
        """Set up all the callbacks for syncing between widgets"""
        self._start_input.param.watch(self._on_input_change, "value")
        self._end_input.param.watch(self._on_input_change, "value")
        self._slider.param.watch(self._on_slider_change, "value_throttled" if self.throttled else "value")

        for name, button in self._snapshot_buttons.items():
            button.on_click(self._get_date_range_callback(name))

        for name, button in self._custom_shortcut_buttons.items():
            button.on_click(self._get_custom_date_range_callback(name))

    def _get_date_range_callback(self, shortcut_name):
        """
        Generates a callback function for date range shortcut buttons.
        """

        def callback(event):
            today = dt.date.today()
            # Use the current selected end date for relative calculations, or the max end date for 'ALL'
            current_selected_end = self.value[1] if self.value and self.value[1] else self.end

            start_bound = self.start  # already initialized
            end_bound = self.end  # already initialized

            new_start = None
            new_end = end_bound  # Always use the maximum selectable end date for relative calculations

            if shortcut_name == "ALL":
                new_start = start_bound
                new_end = end_bound
            elif shortcut_name == "YTD":
                new_start = dt.date(end_bound.year, 1, 1)
            else:
                # Handle N-week, N-month, N-year shortcuts
                import re

                match = re.match(r"(\d+)([WMY])", shortcut_name)
                if match:
                    num = int(match.group(1))
                    unit = match.group(2)

                    if unit == "W":
                        new_start = end_bound - timedelta(weeks=num)
                    elif unit == "M":
                        end_ts = pd.Timestamp(end_bound)
                        new_start = (end_ts - pd.DateOffset(months=num)).date()
                    elif unit == "Y":
                        try:
                            new_start = end_bound.replace(year=end_bound.year - num)
                        except ValueError:
                            # Handle leap year edge case (Feb 29)
                            new_start = end_bound.replace(year=end_bound.year - num, day=28)
                else:
                    # Fallback for unrecognized shortcuts, though ideally all should be handled
                    print(f"Warning: Unrecognized shortcut '{shortcut_name}'")

            if new_start:
                new_start = max(new_start, start_bound) if start_bound else new_start

            if new_start and new_end:
                self._update_all_widgets((new_start, new_end))

        return callback

    def _get_custom_date_range_callback(self, custom_shortcut_name):
        """
        Generates a callback function for custom date range shortcut buttons.
        """

        def callback(event):
            start_val, end_val = self.custom_shortcuts[custom_shortcut_name]

            parsed_start = self._parse_date(start_val, default=self.start)
            parsed_end = self._parse_date(end_val, default=self.end)

            # Ensure custom range is within overall start/end bounds
            start_bound = self.start
            end_bound = self.end

            if parsed_start and start_bound:
                parsed_start = max(parsed_start, start_bound)
            if parsed_end and end_bound:
                parsed_end = min(parsed_end, end_bound)

            if parsed_start and parsed_end:
                self._update_all_widgets((parsed_start, parsed_end))
            elif parsed_start:  # If only start is provided, set end to max
                self._update_all_widgets((parsed_start, end_bound))
            elif parsed_end:  # If only end is provided, set start to min
                self._update_all_widgets((start_bound, parsed_end))

        return callback

    def _on_input_change(self, event):
        """Handle changes from date input widgets"""
        if self._start_input.value and self._end_input.value:
            # Convert datetime.datetime from DatetimeInput to datetime.date for internal value and slider
            new_start_date = self._start_input.value.date()
            new_end_date = self._end_input.value.date()
            new_value = (new_start_date, new_end_date)
            if new_value[0] <= new_value[1]:
                self.value = new_value
                self._updating = True
                self._slider.value = new_value
                self._updating = False

    def _on_slider_change(self, event):
        """Handle changes from the slider"""
        if hasattr(self, "_updating") and self._updating:
            return
        if event.new:
            self.value = event.new
            # Convert datetime.date from DateRangeSlider to datetime.datetime for DatetimeInput
            self._start_input.value = datetime.combine(event.new[0], datetime.min.time())
            self._end_input.value = datetime.combine(event.new[1], datetime.min.time())

    def _update_all_widgets(self, new_value):
        """Update all widgets with new value"""
        # Ensure new_value is a tuple of two dates, handling potential None values
        if new_value is None or not isinstance(new_value, (tuple, list)) or len(new_value) != 2:
            # If new_value is invalid, reset to current valid range or defaults
            current_start = self.start if self.start else dt.date(2020, 1, 1)
            current_end = self.end if self.end else dt.date.today()
            new_value = (current_start, current_end)

        # Ensure dates within new_value are valid datetime.date objects
        valid_start = self._parse_date(new_value[0], default=self.start if self.start else dt.date(2020, 1, 1))
        valid_end = self._parse_date(new_value[1], default=self.end if self.end else dt.date.today())

        # Ensure start <= end
        if valid_start and valid_end and valid_start > valid_end:
            valid_start, valid_end = valid_end, valid_start  # Swap if out of order

        # Update internal value and widgets
        self.param.update(value=(valid_start, valid_end))
        # Convert datetime.date to datetime.datetime for DatetimeInput widgets
        self._start_input.value = datetime.combine(valid_start, datetime.min.time())
        self._end_input.value = datetime.combine(valid_end, datetime.min.time())
        self._slider.value = (valid_start, valid_end)

    def __panel__(self):
        """Create the panel layout"""
        date_inputs = pn.Row(
            self._start_input,
            pn.pane.Markdown("**to**", margin=(10, 10)),
            self._end_input,
            align="center",
            margin=(0, 0, 10, 0),
        )

        slider_row = pn.Row(self._slider, margin=(0, 0, 10, 0))

        buttons_row = pn.Row(*self._snapshot_buttons.values(), margin=(0, 0, 0, 0))

        custom_buttons_row = (
            pn.Row(*self._custom_shortcut_buttons.values(), margin=(0, 0, 0, 0))
            if self.custom_shortcuts
            else pn.Column()
        )  # Only display if custom_shortcuts exist

        return pn.Card(
            pn.Column(
                date_inputs,
                slider_row,
                buttons_row,
                custom_buttons_row,  # Add custom buttons row
                width=350,
            ),
            title=self.name,
            width=380,  # Adjust card width to accommodate inner content and padding
            collapsible=False,
        )


# Example usage
if __name__ == "__main__":
    # Create an instance of the custom widget
    date_selector = DateRangeSelector(
        start=dt.date(2020, 1, 1),
        end=dt.date.today(),
        value=(dt.date(2024, 1, 1), dt.date.today()),
        shortcuts=["ALL", "YTD", "1W", "2W", "1M", "3M", "1Y"],
        custom_shortcuts={
            "Last 6 Months": ((dt.date.today() - pd.DateOffset(months=6)).date(), dt.date.today()),
            "Last Year": (dt.date.today().replace(year=dt.date.today().year - 1), dt.date.today()),
            "Custom Period": ("20230101", "20230630"),
            "Start to Today": (dt.date(2020, 1, 1), None),
            "Today to End": (None, dt.date.today()),
        },
    )

    # Create a display that shows the selected value
    def display_selection(value):
        if value:
            return f"**Selected Range:** {value[0]} to {value[1]}"
        return "No selection"

    # Create the dashboard
    dashboard = pn.template.FastListTemplate(
        title="Date Range Selector Demo",
        sidebar=[
            "## Date Selection",
            date_selector,
            pn.pane.Markdown("---"),
            "### Current Selection:",
            pn.bind(display_selection, date_selector.param.value),
        ],
        main=[],
    )

    # Display in notebook or serve
    # For Jupyter notebook, just display the widget:
    display(
        pn.Column(
            "# Custom Date Range Selector Widget",
            date_selector,
            pn.pane.Markdown("---"),
            "### Selected Value:",
            pn.bind(display_selection, date_selector.param.value),
        )
    )
