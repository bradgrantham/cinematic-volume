Goal is to view my own calcium score raw CT data, and maybe be flexible enough to help other people view their volume data in a DICOM file if you go through some backflips.

"Cinematic" is an aspiration and depends on how much time I can find to dedicate to this.

Here's a sloppy recipe for viewing DICOM volume data.

* Unpack your DICOM data.  There may be a file called `DICOMDIR` in the data.
* Build the `ct-view` tool in `helpers`.  `ct-view` requires `dcmtk` and expects to find the configuration of `dcmtk` libraries and headers using `pkg-config`.

```
cd helpers
make ct-view
```

* `ct-view`  prints the DICOM hierarchy and then the DICOM "sequences" so you can find your sequence of interest in the hierarchy, e.g.:
  * `./ct-view /Users/grantham/Documents/brad_chest_ct/DICOMDIR | less` 
  * The sequences are the last thing printed.  My sequence of note is for my calcium score, so I tried `CAL SCORE` first and that turned out to be the right one.  You may be able to guess yours.
  * Search the structure, which is the first thing printed; you're probably looking for a sequence with a hundred or more sub-images.  For example in mine there's a section indicating "CAL SCORE" has 114 length.

````
            (0008,103e) LO [CAL SCORE]                              #  10, 1 SeriesDescription
            (0020,000e) UI [1.3.12.2.1107.5.1.7.123387.30000023091514270973500000198] #  56, 1 SeriesInstanceUID
            (0020,0011) IS [2]                                      #   2, 1 SeriesNumber
            (0081,0008) CS [ORIGINAL\PRIMARY\AXIAL\CT_SOM5 SEQ]     #  34, 4 Unknown Tag & Data
            (0004,1220) SQ (Sequence with explicit length #=114)    #   0, 1 DirectoryRecordSequence
````

* Note the first subrecord's `ReferencedFileId`.  That's probably the first image in the sequence.

```	
              (fffe,e000) na "Directory Record" IMAGE #=10            # u/l, 1 Item
[...]
                (0004,1500) CS [DICOM\12]                               #   8, 2 ReferencedFileID
```

* Run `extract-all.sh` to pull raw 16-bit data out of the references slices, providing the first slice and the count of slices, e.g.:

```
./extract-all.sh 12 114 ~/Documents/brad_chest_ct
```

* When this script is complete, there will be a series of raw 16-bit data files like `file_000.bin`, `file_001.bin`, etc.
  * Take note of the dimensions printed for every slice (e.g. `512 by 512`) and the slope and intercept.  They should be the same.  If you have selected the correct and should be relatively large.  Write them down.  This is the last one of mine:


```
extracting binary data file_113.bin from DICOM file 125...
512 by 512
slope 1.000000, intercept -8192
```

* Build `volume`.  It requires `GLFW3` and Vulkan and `cmake` to build:

```
cd ..
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release .
(cd build ; make)
```

* Run `volume`, providing the dimensions of the slices and the number of slices, and a `printf`-style string for the binary filenames, then the slope and intercept.

```
./build/volume 512 512 114 helpers/file_%03d.bin 1.0 -8192
```

`volume` may take a while to start up; it's reading a lot of data and then calculating gradients for the volume data.
Rotate the volume by clicking and dragging.  Zoom in on the volume by pressing 'Z' to zoom, then click and drag.  To move the volume left and right and up and down, press 'X' and then click and drag.  Press 'R' to return to rotating the volume.
To change the isosurface threshold by 10 units up or down, press `,` or `.`.  To change 100 units, press ';' or "'".  To change 1000 units, press '[' or ']'.
There's a hardcoded colortable with white for bone and more radiopaque tissue and red for less radiopaque tissue like blood and muscle.

