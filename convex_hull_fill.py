import argparse, ctypes, os, sys, time
import cv2 as cv
import numpy as np

DESC = '''Interactive tool to do a quick convex-hull flood fill on a series of images.
    
    -Click on the image to preview the flood fill area using the clicked position as a seed. 
    -Press 'A' to confirm and apply the previewed fill. Repeat as necessary.
    -Press 'A' again to save a copy of the image in {source}\\clean\\finished.
    -Or press SHIFT + 'A' to save in {source}\\clean\\unfinished. 
'''

ZOOM = 1
IN_DIRS = []

KEY_CONFIRM = 97  # 'A' key. Confirm fill (when preview is shown), or save image in finished folder
KEY_RESERVE = 65  # 'A' key + SHIFT. Save image in unfinished folder (do this if it still needs manual editing)

DIR_NAME_FINISHED = "clean\\finished"
DIR_NAME_UNFINISHED = "clean\\unfinished"

WIN_NAME_PREVIEW = "PREVIEW"
WIN_NAME_FILL = "FILL MASK"
WIN_NAME_HULL = "CONVEX HULL"

# Color and opacity for fill preview overlay
PREVIEW_COLOR = (255, 0, 255)
PREVIEW_OPACITY = 0.5

# Binary state: either ready to save, or previewing fill
IS_READY = True

CURRENT_IMG = None
CURRENT_FILL = None
CURRENT_PATH = None

START_TIME = None
TOTAL_FINISHED = 0
TOTAL_UNFINISHED = 0


def imshow_scaled(mat, winname=WIN_NAME_PREVIEW):
    scaled = cv.resize(mat, None, fx=ZOOM, fy=ZOOM, interpolation=cv.INTER_NEAREST)
    cv.imshow(winname, scaled)


def imclick(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        preview(x, y)


def preview(x, y):
    x_pos_real = int(x / ZOOM)
    y_pos_real = int(y / ZOOM)

    # Prepare convex hull of fill mask
    flood_flags = cv.FLOODFILL_MASK_ONLY | (255 << 8)
    _, _, mask_initial, _ = cv.floodFill(CURRENT_IMG, None, (x_pos_real, y_pos_real), (0, 0), flags=flood_flags)
    _, _, mask_hull, _ = cv.floodFill(mask_initial, None, (1, 1), (0, 0), flags=flood_flags)

    # Trim off extra edges added by floodfill mask
    mask_initial = mask_initial[1:-1, 1:-1]
    mask_hull = mask_hull[2:-2, 2:-2]
    mask_hull = cv.bitwise_not(mask_hull)

    # Make the preview overlay thingy
    preview_overlay = np.zeros((CURRENT_IMG.shape[0], CURRENT_IMG.shape[1], 4), dtype=np.uint8)
    preview_overlay = cv.cvtColor(preview_overlay, cv.COLOR_BGRA2RGBA)
    preview_overlay[:] = (*PREVIEW_COLOR, 0)
    preview_overlay[:, :, 3] = mask_hull
    preview_overlay[:, :, 3] = preview_overlay[:, :, 3] * PREVIEW_OPACITY

    preview = alpha_composite(preview_overlay, CURRENT_IMG.copy())

    imshow_scaled(preview)
    cv.imshow(WIN_NAME_FILL, mask_initial)
    cv.imshow(WIN_NAME_HULL, mask_hull)

    # Make the final overlay (used if fill is confirmed)
    fill = np.zeros((CURRENT_IMG.shape[0], CURRENT_IMG.shape[1], 4), dtype=np.uint8)
    fill = cv.cvtColor(fill, cv.COLOR_BGRA2RGBA)
    fill[:] = (*CURRENT_IMG[y_pos_real][x_pos_real], 0)
    fill[:, :, 3] = mask_hull
    global CURRENT_FILL
    CURRENT_FILL = fill

    global IS_READY
    IS_READY = False


def next_img(img_paths):
    if cv.getWindowProperty(WIN_NAME_FILL, cv.WND_PROP_VISIBLE) >= 1:
        cv.destroyWindow(WIN_NAME_FILL)
    if cv.getWindowProperty(WIN_NAME_HULL, cv.WND_PROP_VISIBLE) >= 1:
        cv.destroyWindow(WIN_NAME_HULL)

    global CURRENT_PATH
    CURRENT_PATH = next(img_paths, None)
    if CURRENT_PATH is None:
        quit()

    global CURRENT_IMG
    CURRENT_IMG = cv.imread(CURRENT_PATH)
    imshow_scaled(CURRENT_IMG)


def save_img(not_done=False):
    global TOTAL_FINISHED
    global TOTAL_UNFINISHED

    dir, filename = os.path.split(CURRENT_PATH)

    if not_done:
        save_dir = os.path.join(dir, DIR_NAME_UNFINISHED)
        TOTAL_UNFINISHED += 1
    else:
        save_dir = os.path.join(dir, DIR_NAME_FINISHED)
        TOTAL_FINISHED += 1

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Saving: " + os.path.join(save_dir, filename))
    cv.imwrite(os.path.join(save_dir, filename), CURRENT_IMG)


def alpha_composite(fg, bg):
    alpha = fg[:, :, 3] / 255.0

    top = weight_rgb(fg[:, :, 0:3], alpha)
    bottom = weight_rgb(bg[:, :, 0:3], 1.0 - alpha)

    return cv.add(top, bottom, bg).astype(np.uint8)


def weight_rgb(rgb, alpha):
    return cv.merge([rgb[:, :, 0] * alpha, rgb[:, :, 1] * alpha, rgb[:, :, 2] * alpha])


def confirm_fill():
    global IS_READY
    global CURRENT_IMG
    CURRENT_IMG = alpha_composite(CURRENT_FILL, CURRENT_IMG.copy())

    imshow_scaled(CURRENT_IMG)
    if cv.getWindowProperty(WIN_NAME_FILL, cv.WND_PROP_VISIBLE) >= 1:
        cv.destroyWindow(WIN_NAME_FILL)
    if cv.getWindowProperty(WIN_NAME_HULL, cv.WND_PROP_VISIBLE) >= 1:
        cv.destroyWindow(WIN_NAME_HULL)
    IS_READY = True


def quit():
    cv.destroyAllWindows()
    print("\n------------------------------------------------------\n")
    print("Finished {} images in {} seconds.".format(TOTAL_FINISHED + TOTAL_UNFINISHED, time.time() - START_TIME))
    print("Fully cleaned: {}".format(TOTAL_FINISHED))
    print("Partially cleaned: {}".format(TOTAL_UNFINISHED))
    sys.exit()


START_TIME = time.time()

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=DESC)
parser.add_argument('-i', '--input-dirs', nargs='+', type=str, required=True,
                    help="Directories that contain images to be processed.")
parser.add_argument('-z', '--zoom', nargs=1, type=int, required=False,
                    help="Optional zoom multiplier for image preview.")

args = parser.parse_args()
ZOOM = args.zoom[0]
IN_DIRS = args.input_dirs

# Find all images in list of paths
img_paths = []
for d in IN_DIRS:
    img_paths += [os.path.join(d, p) for p in os.listdir(d) if os.path.isfile(os.path.join(d, p))]
img_paths = iter(img_paths)

# Initialize preview window
cv.namedWindow(WIN_NAME_PREVIEW)
cv.setMouseCallback(WIN_NAME_PREVIEW, imclick)

next_img(img_paths)

# Input loop
while True:
    k = cv.pollKey()
    if cv.getWindowProperty(WIN_NAME_PREVIEW, cv.WND_PROP_VISIBLE) < 1:
        # Exit if main window closed
        break

    if k == -1:
        # No key press
        continue

    elif k == KEY_CONFIRM:
        if IS_READY:
            save_img()
            next_img(img_paths)
        else:
            confirm_fill()

    elif k == KEY_RESERVE:
        save_img(not_done=True)
        next_img(img_paths)

    elif k == 27:
        # Exit if ESC pressed
        break
quit()
