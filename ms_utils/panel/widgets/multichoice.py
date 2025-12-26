import panel as pn
import param
from typing import List, Union, Optional, Any


class MultiChoiceEnhanced(pn.widgets.MultiChoice):
    """
    An enhanced version of Panel's MultiChoice widget that adds a limit feature.

    When the limit is reached, additional options become grayed out and cannot be selected
    until one of the selected options is removed.

    Parameters
    ----------
    limit : int or None, optional
        Maximum number of choices that can be selected. If None (default),
        behaves exactly like the standard MultiChoice widget.
    """

    limit = param.Integer(
        default=None,
        bounds=(1, None),
        doc="""
        Maximum number of choices that can be selected.
        If None, no limit is applied (default behavior).""",
    )

    # Prevent 'limit' from being passed to Bokeh model
    _rename = {"limit": None, **pn.widgets.MultiChoice._rename}

    def __init__(self, **params):
        super().__init__(**params)
        # Watch for changes to value and limit to update disabled options
        self.param.watch(self._update_disabled_options, ["value", "limit", "options"])
        # Initial update of disabled options
        self._update_disabled_options()

    def _update_disabled_options(self, *events):
        """Update which options are disabled based on the current selection and limit"""
        # If no limit is set, enable all options
        if self.limit is None:
            if hasattr(self, "disabled_options"):
                self.disabled_options = []
            return

        # Get current value and options
        current_value = self.value or []
        current_options = self.options

        # Convert options to list format for consistency
        if isinstance(current_options, dict):
            all_options = list(current_options.keys())
        else:
            all_options = list(current_options)

        # If we've reached the limit, disable all options that are not currently selected
        if len(current_value) >= self.limit:
            # Disable all options that are not currently selected
            self.disabled_options = [opt for opt in all_options if opt not in current_value]
        else:
            # Enable all options when below the limit
            if hasattr(self, "disabled_options"):
                self.disabled_options = []

    def _process_param_change(self, msg):
        # Handle the case where value is being set
        if "value" in msg:
            # Ensure we don't exceed the limit when setting value programmatically
            if self.limit is not None and msg["value"] is not None:
                if len(msg["value"]) > self.limit:
                    # Truncate to the limit
                    msg["value"] = msg["value"][: self.limit]

        return super()._process_param_change(msg)


# Add to module exports
__all__ = ["MultiChoiceEnhanced"]
