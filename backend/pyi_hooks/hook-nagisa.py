"""Keep Nagisa modules on disk for its legacy absolute imports.

Nagisa appends its package directory to ``sys.path`` and then imports sibling
modules such as ``prepro`` as top-level modules. Those imports work from a
normal installation but fail when the package is stored only in PyInstaller's
embedded PYZ archive.
"""

module_collection_mode = "py"
