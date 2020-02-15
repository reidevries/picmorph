from linemorph import *
import sys

if (len(sys.argv) < 4):
    print("expected arguments: src1 src2 destdir")
    sys.exit()

srca = sys.argv[1]
srcb = sys.argv[2]
destdir = sys.argv[3]
if (destdir[-1] != "/"):
	destdir = destdir+"/"

print(srca)
a_orig = Image.open(srca)
a_edge = edgedetect(reduceSize(a_orig))
print(srcb)
b_orig = Image.open(srcb)
b_edge = edgedetect(reduceSize(b_orig))

a, b = equalSize(a_orig, b_orig)
a_p_scale = (a.size[0]/reduce_width, a.size[1]/reduce_height)
b_p_scale = (b.size[0]/reduce_width, b.size[1]/reduce_height)
a_p = sortPointListByDistance(getPointsFromAutotrace(a_edge, a_p_scale), (a.size[0]/2, a.size[1]/2))
b_p = sortPointListByDistance(getPointsFromAutotrace(b_edge, b_p_scale), (b.size[0]/2, b.size[1]/2))

a_edge.save(destdir+"a_edge.png")
b_edge.save(destdir+"b_edge.png")

matches = matchPointLists(a_p, b_p)

interp_frames = 7

a_pixel = a.load()
b_pixel = b.load()
dots_image = [a]*interp_frames
for i in range(0, interp_frames):
    dots_image[i] = interpolateWithDots(a_pixel, b_pixel, a.size, a_p, b_p, matches, float(i/interp_frames))
    print(type(dots_image[i]))

new_image = a
for i in range(0,interp_frames):
    if (i > 0):
        blend_image = Image.blend(a,b,(1-math.cos(M_PI*float(i/interp_frames)))/2)
        new_image = Image.blend(blend_image, new_image, 0.5)
    new_image = Image.alpha_composite(new_image, dots_image[i])
    new_image.save(destdir+str(i)+".png")