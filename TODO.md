Fall back to something realistic:
* Some cleanup
  * unify image?
  * get image into a file
  * more factorization of helpers out of huge prep function
  * get vulkan helpers into a file, share with vulkan-testing
* build a version that does fragment program on quad
  * Step through volume cell boundaries using planes for accuracy
* Set policy - attenuation per unit, scattering per unit
  * factor out a function that determines attenuation and scattering
    * appropriate to a path tracer?
        * review RTIOW

Wish list
* Set policy - attenuation per unit, scattering per unit
* Step through volume cell boundaries
  * Method 1: dx, dy, dz, step each closest
    * Precision issue - each step may be shorter than should be
    * Adapt from Ray12?
    * Harder to integrate with triangles?
  * Method 2: keep track of plane, step closest, then intersect next plane
    * Maybe slightly lower performance
    * Could be easiest to implement on CPU...
  * Method 3: indexed grid of triangles at cell walls
    * Probably this one.
    * Weird, but may fit BLAS/TLAS better and is homogenous with clip revions
    * More complicated on CPU - need BVH code, but I have BVH code...
* Step through range between cell boundaries to catch all transitions
  * Step between cell boundaries at color table resolution to guarantee not missing an opaque boundary
* Key repeat
* Get working on Linux
* Get working on Windows again
* "Project"
  * JSON store
    * image data file
    * Camera parameter sets
    * current camera parameter set index
    * transfer function list
    * current transfer function
* Add an ImGUI
  * reset view
  * save camera 1 - 10, restore camera
  * https://github.com/aiekick/ImGuiFileDialog
* Add clip regions - view, then circle a region to clip away
  * Will want undo pretty quickly
* Add multiple color tables and switching between them
  * "builtin" transfer functions
  * "user" transfer functions
* Add colortable editor
* Add an option to save / restore view parameters
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
