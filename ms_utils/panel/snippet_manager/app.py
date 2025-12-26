import os
import argparse
import panel as pn
from ms_utils.panel.snippet_manager.snippet_manager import SnippetManager

# Argument parsing
parser = argparse.ArgumentParser(description="Standalone SnippetManager app")
parser.add_argument("--port", default=0, type=int, help="Port to listen on (default: 0 => auto assigned)")
parser.add_argument(
    "--snippet_folder",
    type=str,
    default="/home/mshevelev/git/snippets",
    help="Folder containing snippets (default: '/home/mshevelev/git/snippets')",
)
args = parser.parse_args()

# Determine the root folder
root_folder = args.snippet_folder

# Initialize SnippetManager
snippet_manager = SnippetManager(root_folder)

# Get the Panel layout
app = snippet_manager.display()


print(args.port)

# Serve the app
pn.serve(app, port=args.port, allow_websocket_origin=["*"], show=True)
