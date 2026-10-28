"""Convert notebooks/eda.py (cells split on '# %%') into notebooks/eda.ipynb."""
import re, nbformat as nbf
src = open("notebooks/eda.py").read()
nb = nbf.v4.new_notebook(); cells = []
for blk in re.split(r"^# %%", src, flags=re.M)[1:]:
    if blk.startswith(" [markdown]"):
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in blk.split("\n")[1:])
        cells.append(nbf.v4.new_markdown_cell(body.strip()))
    else:
        cells.append(nbf.v4.new_code_cell(blk.strip("\n")))
nb["cells"] = cells; nbf.write(nb, "notebooks/eda.ipynb")
