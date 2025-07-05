import re
import os
import pandas as pd
import panel as pn

pn.extension("codeeditor")


FILES_TO_IGNORE = ('LICENSE',)


class SnippetManager:
  def __init__(self, root_folder: str):
    self._root_folder = os.path.normpath(root_folder)
    self.reload()

  @property
  def root_folder(self):
    return self._root_folder

  def reload(self):
    """Reload snippets from same folder"""
    self._file_list_with_tags = list(self._get_all_items(self.root_folder))

    self._tag_list = set()
    for item in self._file_list_with_tags:
      for tag in item[1]:
        self._tag_list.add(tag)
    return self

  @staticmethod
  def _read_tags(fname):
    pattern = r"@tags:\s*([a-z0-9_,\s]+)"
    with open(fname, "r") as f:
      first_line = f.readline().strip()

    match = re.search(pattern, first_line)
    if match:
      # Extract the matched group and remove any whitespace around commas
      tags = re.sub('\s*,\s*', ',', match.group(1).strip())
      return tags.split(',')
    else:
      if "@tags:" in first_line:
        raise ValueError("first line contains '@tags:', but I cannot parse this string...")
      return []

  def _get_all_items(self, folder):
    #for item in root_dir.iterdir():
    for item in os.listdir(folder):
      if item.startswith(".") or (item in FILES_TO_IGNORE):
        continue
      item = os.path.join(folder, item)
      if os.path.isfile(item):
        yield (item, self._read_tags(item))
      elif os.path.isdir(item):
        yield from self._get_all_items(item)
      else:
        raise OSError(f"{item=} is neither file or directory")

  def _file_select_callback(self, event):
    fpath = os.path.join(self.root_folder, event.new)
    if os.path.isfile(fpath):
      with open(fpath, "r") as f:
        self._editor.value = f.read()
        self._editor.filename = event.new
        self._file_path.value = fpath
    else:
      self._editor.value = "No such file found"
      self._editor.filename = "Not_found.txt"
      self._file_path.value = ""

  def _tag_select_callback(self, event):
    if len(event.new) == 0:
      self._file_select.options = [os.path.relpath(item[0], self.root_folder) for item in self._file_list_with_tags]
      _tag_list = set()
      for item in self._file_list_with_tags:
        for tag in item[1]:
          _tag_list.add(tag)
      self._tag_select.options = list(_tag_list)
    else:
      items_to_add = []
      for item in self._file_list_with_tags:
        if(all(x in item[1] for x in event.new)):
          items_to_add.append(item)
      self._file_select.options = [os.path.relpath(item[0], self.root_folder) for item in items_to_add]

      _tag_list = set()
      for item in items_to_add:
        for tag in item[1]:
          _tag_list.add(tag)
      self._tag_select.options = list(_tag_list)

  def get_full_file_list_with_tags(self):
    return self._file_list_with_tags

  def get_df_with_files_and_tags(self) -> pd.DataFrame:
    """Return dataframe with one row per (filepath, tag)"""
    res = pd.DataFrame([
      (filepath, tag) for filepath, tags in self.get_full_file_list_with_tags() for tag in tags],
      columns=["filepath", "tag"]
    )
    return res

  def panel_display(self):
    self._file_select = pn.widgets.Select(
      name='File Select', 
      options=[os.path.relpath(item[0], self.root_folder) for item in self._file_list_with_tags], 
      size=len(self._file_list_with_tags),
      height=300
      )

    self._editor = pn.widgets.CodeEditor(value="", sizing_mode='stretch_width', readonly=True,  height=400)
    self._tag_select = pn.widgets.MultiChoice(
      options=list(self._tag_list),
      value=[],
      width=300,
      name="Tags"
    )
    self._file_path = pn.widgets.StaticText(name='File Path', value="")
    self._file_select.param.watch(self._file_select_callback, ["value"], onlychanged=False)
    self._tag_select.param.watch(self._tag_select_callback, ["value"], onlychanged=False)
    root_folder_display = pn.widgets.StaticText(name='Root Folder', value=self.root_folder)
    layout = pn.Column(
      root_folder_display, 
      pn.Row(
        pn.Column(
          self._tag_select,
          self._file_select,
        ),
        pn.Column(
          self._file_path,
          self._editor
        )
      )
    ) 
    return layout

  def display(self, renderer="panel"):
    if renderer == "panel":
      return self.panel_display()
    else:
      raise ValueError(f"{renderer=} not supported")

