# Snippet Manager

The `SnippetManager` class provides a user interface for browsing, searching, and filtering code snippets stored in a directory. It is built using [Panel](https://panel.holoviz.org/) and supports filtering by tags (defined in the file header) and searching within the file content.

## Features

*   **Tag Filtering**: Filter files based on tags specified in the first line of the file (e.g., `@tags: python, pandas`).
*   **Content Search**: Search for text within the snippet files.
*   **File Preview**: View the content of the selected file.

## Usage

### 1. Using in a Jupyter Notebook

You can use `SnippetManager` directly within a Jupyter Notebook to interactively browse your snippets.

See [snippet_manager_demo.ipynb](snippet_manager_demo.ipynb) for a complete example.

```python
from ms_utils.panel.snippet_manager.snippet_manager import SnippetManager

# Initialize with the path to your snippets folder
manager = SnippetManager(root_folder="/path/to/your/snippets")

# Display the interface
manager.display()
```

### 2. Running as a Standalone App

You can also run the Snippet Manager as a standalone web application using the provided `app.py` script.

```bash
# Run with default settings (looks for snippets in /home/mshevelev/git/snippets)
python ms_utils/panel/snippet_manager/app.py

# Specify a custom snippets folder and port
python ms_utils/panel/snippet_manager/app.py --snippet_folder /path/to/my/snippets --port 5006
```

## File Format

To make your snippets compatible with the tag filter, add a tag line as the first line of your file:

```text
@tags: tag1, tag2, tag3
... rest of the file ...
```
