from pathlib import Path
import nrrd
import numpy as np

def convert(path: Path):
    data, md = nrrd.read(str(path))
    spacings = np.diag(md["space directions"])
    axs_to_flip = tuple(np.where(spacings < 0)[0])
    out_data = np.flip(data, axs_to_flip)
    md["space directions"] = np.abs(md["space directions"])
    nrrd.write(str(path), out_data, header=md, compression_level=3, detached_header=True)

for path in Path().glob("*.nhdr"):
    convert(path)
