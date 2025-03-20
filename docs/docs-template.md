## Considerations for all commands
- t6alh organizes images in to folders with names like `001` where each folder
  corresponds to a confocal acquisition, such as a czi scene
- Most commands have optional arguments for a variable number folder names. Or
  an --image-folders-file argument which is the path to a file with this
  information
- Global option ```--verbose``` can be used to get more output. eg `t6alh
  --verbose get-paths`
- Global option ```--database``` can be used to use a custom database location.
  eg `t6alh --database ~/t6alh.db get-paths`

## Align commands
These commands are used to align your images to the template image

### init-and-segment
<p class=accent>Run the below commands to initialize a database and segment the neuropil</p>
{t6alh_init-and-segment}

#### init
<p class=accent>Initialize the database caching raw files</p>
{t6alh_init}

#### add-more-raw
<p class=accent>Add more raw files to existing database</p>
{t6alh_add-more-raw}

#### segment-neuropil
<p class=accent>Finds which pixels are neuropil</p>
{t6alh_segment-neuropil}

### make-landmarks
<p class=accent>Select brain, SEZ, and neuropil tip</p>
{t6alh_make-landmarks}

### align
<p class=accent>These commands are responsible for aligning the image first
based on its neuropil mask, then based on its FasII stain<\p>
{t6alh_align}

#### select-neuropil-fasii
<p class=accent>Find the FasII image that is within the neuropil</p>
{t6alh_select-neuropil-fasii}

#### landmark-register
<p class=accent>Does a rigid registration using landmarks</p>
{t6alh_landmark-register}

#### mask-register
<p class=accent>Aligns the image to the neuropil mask template</p>
{t6alh_mask-register}

#### fasii-align
<p class=accent>Aligns the image to the fasii template</p>
{t6alh_fasii-align}

## Template creation commands
These commands are necessary to make a template image. They depend on some
[Align commands](#align-commands) being run.

### landmark-align
<p class=accent>Align based on landmarks</p>
{t6alh_template_landmark-align}

### make-mask-template
<p class=accent>Aligns and warps together landmark aligned neuropil masks into
a template</p>
{t6alh_template_make-mask-template}

### fasii-template
<p class=accent>Aligns and warps together</p>
{t6alh_template_fasii-template}


## Miscellaneous commands
These commands are not necessary to make or align to a template, but they still
may be helpful

### get-paths
<p class=accent>Informs user about file locations</p>
{t6alh_get-paths}

### clean
<p class=accent>Resets database to which images exist on disk</p>
{t6alh_clean}

### view
<p class=accent>View the image</p>
{t6alh_view}
