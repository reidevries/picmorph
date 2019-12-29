from linemorph import *
import sys

if (len(sys.argv) < 4):
    print("expected arguments src1 src2 destname")
    sys.exit()

srca = sys.argv[1]
srcb = sys.argv[2]
dest = sys.argv[3]
a = open_image(srca)
b = open_image(srcb)

apoints = getPointList(a, 320)
newimage = drawImageFromPoints(apoints)
save_image(newimage, dest)
