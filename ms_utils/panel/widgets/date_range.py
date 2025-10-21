import panel as pn
import param
import datetime as dt
from datetime import datetime, timedelta
import pandas as pd

pn.extension()

class DateRangeSelector(pn.viewable.Viewer):
    """
    A custom date range selector widget that provides three ways to select dates:
    1. Manual input via date pickers
    2. Date range slider
    3. Shortcut buttons (ALL, YTD, 1W, 1M, 1Y)
    """

    # Define the value parameter as a tuple of dates
    # Allow None or tuples containing None or date-like objects
    value = param.Tuple(default=None, length=2, doc="Selected date range (start_date, end_date)")

    # Min and max bounds for the date range
    # Allow None and datetime.date objects
    start = param.Date(default=None, doc="Minimum selectable date")
    end = param.Date(default=None, doc="Maximum selectable date")

    def __init__(self, start=None, end=None, value=None, **params):
        # Parse and set default start date
        parsed_start = self._parse_date(start, default=dt.date(2020, 1, 1))
        # Parse and set default end date
        parsed_end = self._parse_date(end, default=dt.date.today())

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

        super().__init__(start=parsed_start, end=parsed_end, value=parsed_value, **params)

        # Create the date input widgets
        self._start_input = pn.widgets.DatePicker(
            name='From:',
            value=self.value[0],
            start=self.start, # Use the parsed start date
            end=self.end,     # Use the parsed end date
            width=150
        )

        self._end_input = pn.widgets.DatePicker(
            name='To:',
            value=self.value[1],
            start=self.start, # Use the parsed start date
            end=self.end,     # Use the parsed end date
            width=150
        )

        # Create the date range slider
        self._slider = pn.widgets.DateRangeSlider(
            name='',
            start=self.start,
            end=self.end,
            value=self.value,
            step=1,
            width=320
        )

        self._snapshot_buttons = {
            'ALL': pn.widgets.Button(name='ALL', button_type='primary', width=60, height=35, margin=(5, 2)),
            'YTD': pn.widgets.Button(name='YTD', button_type='primary', width=60, height=35, margin=(5, 2)),
            '1W': pn.widgets.Button(name='1W', button_type='primary', width=60, height=35, margin=(5, 2)),
            '1M': pn.widgets.Button(name='1M', button_type='primary', width=60, height=35, margin=(5, 2)),
            '1Y': pn.widgets.Button(name='1Y', button_type='primary', width=60, height=35, margin=(5, 2)),
        }

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
        self._start_input.param.watch(self._on_input_change, 'value')
        self._end_input.param.watch(self._on_input_change, 'value')
        self._slider.param.watch(self._on_slider_change, 'value')

        for name, button in self._snapshot_buttons.items():
            button.on_click(self._get_date_range_callback(name))

    def _get_date_range_callback(self, shortcut_name):
        """
        Generates a callback function for date range shortcut buttons.
        """
        def callback(event):
            today = dt.date.today()
            # Use the current selected end date for relative calculations, or the max end date for 'ALL'
            current_selected_end = self.value[1] if self.value and self.value[1] else self.end
            
            start_bound = self.start # already initialized
            end_bound = self.end # already initialized

            new_start = None
            new_end = end_bound # Always use the maximum selectable end date for relative calculations

            if shortcut_name == 'ALL':
                new_start = start_bound
                new_end = end_bound
            elif shortcut_name == 'YTD':
                new_start = dt.date(end_bound.year, 1, 1)
                new_start = max(new_start, start_bound) if start_bound else new_start
            elif shortcut_name == '1W':
                new_start = end_bound - timedelta(days=7)
                new_start = max(new_start, start_bound) if start_bound else new_start
            elif shortcut_name == '1M':
                end_ts = pd.Timestamp(end_bound)
                one_month_ago_ts = end_ts - pd.DateOffset(months=1)
                new_start = one_month_ago_ts.date()
                new_start = max(new_start, start_bound) if start_bound else new_start
            elif shortcut_name == '1Y':
                try:
                    new_start = end_bound.replace(year=end_bound.year - 1)
                except ValueError:
                    # Handle leap year edge case (Feb 29)
                    new_start = end_bound.replace(year=end_bound.year - 1, day=28)
                new_start = max(new_start, start_bound) if start_bound else new_start
            
            if new_start and new_end:
                self._update_all_widgets((new_start, new_end))

        return callback

    def _on_input_change(self, event):
        """Handle changes from date input widgets"""
        if self._start_input.value and self._end_input.value:
            new_value = (self._start_input.value, self._end_input.value)
            if new_value[0] <= new_value[1]:
                self.value = new_value
                self._updating = True
                self._slider.value = new_value
                self._updating = False

    def _on_slider_change(self, event):
        """Handle changes from the slider"""
        if hasattr(self, '_updating') and self._updating:
            return
        if event.new:
            self.value = event.new
            self._start_input.value = event.new[0]
            self._end_input.value = event.new[1]

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
            valid_start, valid_end = valid_end, valid_start # Swap if out of order

        # Update internal value and widgets
        self.param.update(value=(valid_start, valid_end))
        self._start_input.value = valid_start
        self._end_input.value = valid_end
        self._slider.value = (valid_start, valid_end)


    def __panel__(self):
        """Create the panel layout"""
        date_inputs = pn.Row(
            self._start_input,
            pn.pane.Markdown("**to**", margin=(10, 10)),
            self._end_input,
            align='center',
            margin=(0, 0, 10, 0)
        )
        
        slider_row = pn.Row(
            self._slider,
            margin=(0, 0, 10, 0)
        )
        
        buttons_row = pn.Row(
            *self._snapshot_buttons.values(),
            margin=(0, 0, 0, 0)
        )
        
        return pn.Column(
            date_inputs,
            slider_row,
            buttons_row,
            width=350
        )

# Example usage
if __name__ == "__main__":
    # Create an instance of the custom widget
    date_selector = DateRangeSelector(
        start=dt.date(2020, 1, 1),
        end=dt.date.today(),
        value=(dt.date(2024, 1, 1), dt.date.today())
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
            pn.bind(display_selection, date_selector.param.value)
        ],
        main=[]
    )
    
    # Display in notebook or serve
    # For Jupyter notebook, just display the widget:
    display(pn.Column(
        "# Custom Date Range Selector Widget",
        date_selector,
        pn.pane.Markdown("---"),
        "### Selected Value:",
        pn.bind(display_selection, date_selector.param.value)
    ))