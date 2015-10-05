# Generates a binary STL surface from an image

First the contours of the image are found, then these sontours are sampled and the sampled points are Delauney-triangulated. Finally, the triangulation is saved as a binary STL file.

# TO DO

This version only generates a surface, which 3D programs don't like too much. Next iteration will add a frame to the surface to give it some meat. 


![Plugin current screenshot](https://github.com/aelialper/STL_surface_python/blob/master/screenshot_image.png)

![Plugin current screenshot](https://github.com/aelialper/STL_surface_python/blob/master/screenshot_surface.png)