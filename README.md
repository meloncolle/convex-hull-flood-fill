# convex-hull-flood-fill
### Interactive tool to do a quick convex-hull flood fill on a series of images.

![Alt text](demo.gif?raw=true "Title")

You can click a point on the image to do a flood-fill, except all of the interior holes of the region are also filled.
I made this because I had to clean up a large batch of manga panels, but its use is pretty limited. 



#### Controls:
- Click on the image to preview the flood fill area using the clicked position as a seed.
- Press 'A' to confirm and apply the previewed fill. Repeat as necessary.
- Press 'A' again to save a copy of the image in {source}\clean\finished.
- Or press SHIFT + 'A' to save in {source}\clean\unfinished. (This is for images that require further manual editing later on.)



#### Options:
```
  -h, --help            show this help message and exit
  -i INPUT_DIRS [INPUT_DIRS ...], --input-dirs INPUT_DIRS [INPUT_DIRS ...]
                        Directories that contain images to be processed.
  -z ZOOM, --zoom ZOOM  Optional zoom multiplier for image preview.
```
