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
    value = param.Tuple(default=None, length=2, doc="Selected date range (start_date, end_date)")
    
    # Min and max bounds for the date range
    start = param.Date(default=None, doc="Minimum selectable date")
    end = param.Date(default=None, doc="Maximum selectable date")
    
    def __init__(self, start=None, end=None, value=None, **params):
        # Set default date range if not provided
        if start is None:
            start = dt.date(2020, 1, 1)
        if end is None:
            end = dt.date.today()
            
        # Set initial value to full range if not provided
        if value is None:
            value = (start, end)
            
        super().__init__(start=start, end=end, value=value, **params)
        
        # Create the date input widgets
        self._start_input = pn.widgets.DatePicker(
            name='From:', 
            value=self.value[0],
            start=self.start,
            end=self.end,
            width=150
        )
        
        self._end_input = pn.widgets.DatePicker(
            name='To:', 
            value=self.value[1],
            start=self.start,
            end=self.end,
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
        
        # Create shortcut buttons
        self._button_all = pn.widgets.Button(
            name='ALL',
            button_type='primary',
            width=60,
            height=35,
            margin=(5, 2)
        )
        
        self._button_ytd = pn.widgets.Button(
            name='YTD',
            button_type='primary',
            width=60,
            height=35,
            margin=(5, 2)
        )
        
        self._button_1w = pn.widgets.Button(
            name='1W',
            button_type='primary',
            width=60,
            height=35,
            margin=(5, 2)
        )
        
        self._button_1m = pn.widgets.Button(
            name='1M',
            button_type='primary',
            width=60,
            height=35,
            margin=(5, 2)
        )
        
        self._button_1y = pn.widgets.Button(
            name='1Y',
            button_type='primary',
            width=60,
            height=35,
            margin=(5, 2)
        )
        
        # Set up event handlers
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        """Set up all the callbacks for syncing between widgets"""
        
        # Watch for changes in date inputs
        self._start_input.param.watch(self._on_input_change, 'value')
        self._end_input.param.watch(self._on_input_change, 'value')
        
        # Watch for changes in slider
        self._slider.param.watch(self._on_slider_change, 'value')
        
        # Set up button callbacks
        self._button_all.on_click(self._on_all_click)
        self._button_ytd.on_click(self._on_ytd_click)
        self._button_1w.on_click(self._on_1w_click)
        self._button_1m.on_click(self._on_1m_click)
        self._button_1y.on_click(self._on_1y_click)
        
    def _on_input_change(self, event):
        """Handle changes from date input widgets"""
        if self._start_input.value and self._end_input.value:
            new_value = (self._start_input.value, self._end_input.value)
            
            # Ensure start <= end
            if new_value[0] <= new_value[1]:
                # Update internal value
                self.value = new_value
                
                # Update slider without triggering its callback
                self._updating = True
                self._slider.value = new_value
                self._updating = False
    
    def _on_slider_change(self, event):
        """Handle changes from the slider"""
        if hasattr(self, '_updating') and self._updating:
            return
            
        if event.new:
            # Update internal value
            self.value = event.new
            
            # Update input fields
            self._start_input.value = event.new[0]
            self._end_input.value = event.new[1]
    
    def _on_all_click(self, event):
        """Select entire date range"""
        new_value = (self.start, self.end)
        self._update_all_widgets(new_value)
    
    def _on_ytd_click(self, event):
        """Select year-to-date"""
        year_start = dt.date(self.end.year, 1, 1)
        # Make sure year_start is not before self.start
        year_start = max(year_start, self.start)
        new_value = (year_start, self.end)
        self._update_all_widgets(new_value)
    
    def _on_1w_click(self, event):
        """Select last 1 week"""
        week_ago = self.end - timedelta(days=7)
        # Make sure week_ago is not before self.start
        week_ago = max(week_ago, self.start)
        new_value = (week_ago, self.end)
        self._update_all_widgets(new_value)
    
    def _on_1m_click(self, event):
        """Select last 1 month"""
        # Calculate 1 month ago (approximate)
        if self.end.month == 1:
            month_ago = self.end.replace(year=self.end.year - 1, month=12)
        else:
            month_ago = self.end.replace(month=self.end.month - 1)
        
        # Handle day overflow (e.g., March 31 -> Feb 31 doesn't exist)
        try:
            month_ago = month_ago
        except ValueError:
            # Set to last day of previous month
            if self.end.month == 1:
                month_ago = dt.date(self.end.year - 1, 12, 31)
            else:
                month_ago = dt.date(self.end.year, self.end.month - 1, 
                                   pd.Timestamp(self.end.year, self.end.month - 1, 1).days_in_month)
        
        # Make sure month_ago is not before self.start
        month_ago = max(month_ago, self.start)
        new_value = (month_ago, self.end)
        self._update_all_widgets(new_value)
    
    def _on_1y_click(self, event):
        """Select last 1 year"""
        try:
            year_ago = self.end.replace(year=self.end.year - 1)
        except ValueError:
            # Handle leap year edge case (Feb 29)
            year_ago = self.end.replace(year=self.end.year - 1, day=28)
        
        # Make sure year_ago is not before self.start
        year_ago = max(year_ago, self.start)
        new_value = (year_ago, self.end)
        self._update_all_widgets(new_value)
    
    def _update_all_widgets(self, new_value):
        """Update all widgets with new value"""
        self.value = new_value
        self._start_input.value = new_value[0]
        self._end_input.value = new_value[1]
        self._slider.value = new_value
    
    def __panel__(self):
        """Create the panel layout"""
        # Create horizontal layout for date inputs with label
        date_inputs = pn.Row(
            self._start_input,
            pn.pane.Markdown("**to**", margin=(10, 10)),
            self._end_input,
            align='center',
            margin=(0, 0, 10, 0)
        )
        
        # Slider row
        slider_row = pn.Row(
            self._slider,
            margin=(0, 0, 10, 0)
        )
        
        # Buttons row
        buttons_row = pn.Row(
            self._button_all,
            self._button_ytd,
            self._button_1w,
            self._button_1m,
            self._button_1y,
            margin=(0, 0, 0, 0)
        )
        
        # Combine all components
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



# # Basic usage
# date_selector = DateRangeSelector(
#     start=dt.date(2020, 1, 1),
#     end=dt.date.today()
# )

# # Access the selected value
# selected_range = date_selector.value  # Returns (start_date, end_date)

# # Use with pn.depends
# @pn.depends(date_selector.param.value)
# def process_data(date_range):
#     start, end = date_range
#     # Your processing logic here
#     return f"Processing from {start} to {end}"

# # Use with pn.bind
# output = pn.bind(process_data, date_selector.param.value)