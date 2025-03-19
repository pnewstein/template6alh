# Template 6 Alh

**WORK IN PROGRESS**

A tool for aligning images of <i>Drosophila</i> brains from 6 hour old larvea

## Quick start
Three commands are necessary to align to the template brain

### Import data
 ```bash
t6alh init-and-segment --root-dir /storage/data --fasii-chan 1 --neuropil-chan 2 example1.czi example2.czi
 ```
- Reads all scenes from zeiss czi files and caches the raw data
- Downloads templates
- Lowers the resolution of the file and classifies pixels as belonging to the neuropil

### Choose landmarks
```bash
t6alh make-landmarks
```
- A custom GUI designed for use over ssh forwarding is used to select the landmarks
- landmarks are saved to the cache

### Align
```bash
t6alh align
```
- First register to the landmarks
- then register and warp to the mask template
- then register and warp to the FasII template 


## Installation

### Prerequisites
- [CMTK](https://www.nitrc.org/projects/cmtk)
- [conda](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
- Template images: I will make these publicly available soon

### Create environment
```
git clone https://github.com/pnewstein/template6alh
cd template6alh
conda env create -f environment.yml
```
### Activate environment
```
conda activate template-env
```



# Documentation
Full documentation can be found at [pnewstein.github.io/t6alh-doc](https://pnewstein.github.io/t6alh-doc)

