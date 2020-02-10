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
a_p = getPointsFromAutotrace(a_edge, a_p_scale)
b_p = getPointsFromAutotrace(b_edge, b_p_scale)

a.save(destdir+"a.png")
b.save(destdir+"b.png")

matches = matchPointLists(a_p, b_p)
a_d = getDelaunay(a_p)
b_d = getDelaunay(b_p)

for i in range(0, 11):
	blend_image = interpolateWithDots(a, b, a_p, b_p, matches, float(i/10))
	blend_image.save(destdir+str(i)+".png")
