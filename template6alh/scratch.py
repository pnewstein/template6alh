from pathlib import Path
import json
import shutil

json_path = Path("eg.json")
eg_ob = json.loads(json_path.read_text())
out_dict: dict[str, str] = {}
for name, eg_path in eg_ob.items():
    exe = Path(eg_path).name
    real_path = shutil.which(exe)
    if real_path is None:
        print(f"please install {exe}, {name}")
        continue
    out_dict[name] = real_path
Path("myconfig.py").write_text(json.dumps(out_dict, indent="\t"))
