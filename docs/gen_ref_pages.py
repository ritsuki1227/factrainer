"""API reference pages automatic generation script"""

import importlib
import os
from pathlib import Path

import mkdocs_gen_files

# Project root directory
root = Path(__file__).parent.parent

# Source code directories and their corresponding modules
modules = [
    ("factrainer-core", "factrainer.core"),
    ("factrainer-base", "factrainer.base"),
    ("factrainer-lightgbm", "factrainer.lightgbm"),
    ("factrainer-sklearn", "factrainer.sklearn"),
    ("factrainer-xgboost", "factrainer.xgboost"),
    ("factrainer-catboost", "factrainer.catboost"),
]

# Generate documentation for each module
for dir_name, module_name in modules:
    # Skip if the module directory doesn't exist
    if not (root / dir_name).exists():
        continue
    
    # Try to import the module to get its __all__ list
    try:
        module = importlib.import_module(module_name)
        exported_objects = getattr(module, "__all__", [])
    except (ImportError, ModuleNotFoundError):
        # If the module can't be imported, skip it
        continue
    
    # Get the module parts for path construction
    module_parts = module_name.split(".")
    
    # Documentation directory path
    doc_path = Path("reference") / module_parts[-1]
    
    # Create index page
    index_path = doc_path / "index.md"
    with mkdocs_gen_files.open(index_path, "w") as f:
        f.write(f"# {module_name}\n\n")
        f.write(f"Overview of the {module_name} module\n\n")
        
        if exported_objects:
            f.write("## Public API\n\n")
            for obj in exported_objects:
                f.write(f"- [{obj}]({obj.lower()}.md)\n")
        else:
            f.write("This module is currently a placeholder and not yet implemented.\n")
            f.write("Functionality will be added in a future release.\n")
    
    # Generate documentation for each exported object
    for obj_name in exported_objects:
        # Create a documentation file for each exported object
        doc_file = doc_path / f"{obj_name.lower()}.md"
        
        with mkdocs_gen_files.open(doc_file, "w") as f:
            f.write(f"# {obj_name}\n\n")
            f.write(f"::: {module_name}.{obj_name}\n")
            f.write("    options:\n")
            f.write("      show_root_heading: true\n")
            f.write("      show_root_toc_entry: true\n")
            f.write("      show_signature_annotations: true\n")
            f.write("      members: true\n")
            f.write("      show_source: false\n")
