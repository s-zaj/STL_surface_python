# Generates a binary STL surface from an image

Works well if the heatmap is differentiable (maybe blur it a little first?).

First the contour lines of the heatmap are drawn, then these contours are sampled and the sampled points are Delauney-triangulated. Finally, the triangulation is saved as a binary STL file.

# TO DO

* This version only generates a surface, which 3D programs don't like too much. Next iteration will add a frame to the surface to give it some meat. 
* Improve the contour sampling (sample the distance evenly).
* Find out how sharp the edges can be before it fails.


![Plugin current screenshot](https://github.com/aelialper/STL_surface_python/blob/master/screenshot_image.png)

![Plugin current screenshot](https://github.com/aelialper/STL_surface_python/blob/master/screenshot_surface.png)