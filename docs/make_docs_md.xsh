import os
import string
env = os.environ.copy()
env["COLUMNS"] = "50"
from subprocess import run
docs = $(cat docs-template.md)
format_dict = {}
for _, field_name, _, _ in string.Formatter().parse(docs):
    if field_name is None:
        continue
    cmd = ["micromamba", "run", "-n", "template-env"] + field_name.split("_") + ["--help"]
    out = run(cmd, capture_output=True, env=env).stdout.decode()
    format_dict[field_name] = f"```\n{out}```"
formated = docs.format(**format_dict)
echo @(formated) > build/docs.md


