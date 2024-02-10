* Add a color / transparency table
    -1000 : air
    >3000 : metals
    so 4096 might be sufficient?
* Join color tables together
* Add an ImGUI
* Add clip regions - view, then circle a region to clip away
  * Will want undo pretty quickly
* Add multiple color tables and switching between them
* Add colortable editor
* Add an option to save / restore view parameters
* Step through volume cell boundaries
  * adapt from ray12
* Step through range between cell boundaries to catch all transitions
  * Step segment at color table resolution to guarantee not missing an opaque boundary
* Do lighting along ray stepping
  * make a templated function in order to reuse the stepping code?
  * yielding segments
* Use a better FOV
* Allow click-scroll and scroll-wheel to zoom
* Allow moving the light
* Read DICOM and then get images directly
* Throw exceptions for all failures and catch them
* Use <iostream> and <format> and as little printf as possible
* Do what preprocessing?
* Use a Vulkan shader
* Better manipulator - need touchpad two-finger drag to zoom
