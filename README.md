# ms_utils

To install package locally run:

```
python setup.py install
```

## Usage and Configuration

`ms_utils` provides a suite of extension methods for pandas, xarray, and HoloViews.

### Automatic Registration
Extension methods are automatically registered on their respective classes when you import the corresponding subpackage:

```python
import ms_utils.pandas as mspd
import pandas as pd

# Extension methods are now available on pd.DataFrame/pd.Series via the .ms namespace
df = pd.DataFrame({'a': [1, 2, 3]})
df.ms.ecdf_transform()

# Methods are also available directly from the subpackage
mspd.ecdf_transform(df)
```

### Configuring the Namespace
By default, methods are registered under the `.ms` accessor. You can customize this by calling `set_namespace` **before** importing any extension subpackages:

```python
import ms_utils
ms_utils.set_namespace("custom_ns")

import ms_utils.pandas
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3]})
df.custom_ns.ecdf_transform()
```

### Registration Conflict Handling
You can control how `ms_utils` handles cases where a method with the same name is already registered in a namespace:

```python
import ms_utils
ms_utils.set_conflict_mode("ignore") # Valid modes: "raise", "override", "ignore"
```

- **`raise`** (default): Raises a `ValueError` if a conflict is detected.
- **`override`**: Overwrites the existing method with the new one.
- **`ignore`**: Skips registration of the new method and logs a warning.
