from PIL import Image, ImageDraw, ImageChops
from scipy.spatial import Delaunay
import numpy as np
import os
import math
import subprocess
import shlex

M_PI = 3.14
reduce_width = 512
reduce_height = 512

def getPixel(image, coord):
	i = coord[0]
	j = coord[1]
	width, height = image.size
	
	pixel = image.getpixel((max(0,min(width-1,i)),max(0,min(height-1,j))))
	return pixel

def sumRGB(rgb):
	return rgb[0]+rgb[1]+rgb[2]

def addColors(colora, colorb):	#expects three-element tuples representing the colours
	newr = max(0, min(1, colora[0]+colorb[0]))
	newg = max(0, min(1, colora[1]+colorb[1]))
	newb = max(0, min(1, colora[2]+colorb[2]))
	return (newr, newg, newb, 255)

def blendColors(colora, colorb, pos, alpha):
	newr = int((colora[0]*(1-pos) + colorb[0]*pos))
	newg = int((colora[1]*(1-pos) + colorb[1]*pos))
	newb = int((colora[2]*(1-pos) + colorb[2]*pos))

	return (newr, newg, newb, int(alpha))

def reduceSize(image):
	print("resizing to " +str(reduce_width)+"x"+str(reduce_height)+"...")
	return image.resize((reduce_width, reduce_height), resample=Image.BILINEAR)

def equalSize(a, b):
	aw, ah = a.size
	bw, bh = b.size
	if (bw == aw and bh == ah):
		a_out = Image.new("RGBA", (bw,bh), "white")
		a_out.paste(a)
		b_out = Image.new("RGBA", (bw,bh), "white")
		b_out.paste(b)
		return a_out,b_out

	ow, oh = a.size
	if (bw > aw):
		ow = bw
	if (bh > ah):
		oh = bh
	print("resizing both images to " +str(ow)+"x"+str(oh)+"...")

	a_out = Image.new("RGBA", (ow,oh), "white")
	a_out.paste(a.resize((ow,oh), resample=Image.BILINEAR))
	b_out = Image.new("RGBA", (ow, oh), "white")
	b_out.paste(b.resize((ow, oh), resample=Image.BILINEAR))
	return a_out,b_out

def edgedetect(image, line_width=1):
	print("applying sobel edge detection...")
	width,height = image.size
	pixel = image.load()
	newimage = Image.new("RGBA", (width, height), "white")
	newdrawing = ImageDraw.Draw(newimage)
	
	for i in range(width):
		for j in range(height):

			il = max(i-1, 0)
			ir = min(i,width-1)
			ju = max(j-1, 0)
			jd = min(j,height-1)
			
			tl = sumRGB(pixel[il,ju])
			t = sumRGB(pixel[i,ju])
			tr = sumRGB(pixel[ir,ju])
			l = sumRGB(pixel[il,j])
			r = sumRGB(pixel[ir,j])
			bl = sumRGB(pixel[il,jd])
			b = sumRGB(pixel[i,jd])
			br = sumRGB(pixel[ir,jd])
			
			gx = abs(tr-tl+2*(r-l)+br-bl)
			gy = abs(tl-bl+2*(t-b)+tr-br)

			g = int(math.sqrt(gx*gx + gy*gy))
			
			if (g > 96):
				if (line_width > 1):
					newdrawing.ellipse([(i-line_width/2, j-line_width/2), (i+line_width/2, j+line_width/2)], fill=(0,0,0))
				else:
					newdrawing.point((i,j), fill=(0,0,0))

	return newimage

def twotone(image, split=127):
	print("applying two tone filter to r,g,b channels with split point", split, "...")
	width,height = image.size
	newimage = Image.new("RGB", (width, height), "white")
	newpixel = newimage.load()

	for i in range(width):
		for j in range(height):
			pixel = getPixel(image,(i,j))
			r = 0
			g = 0
			b = 0
			if (pixel[0] > split):
				r = 255
			if (pixel[1] > split):
				g = 255
			if (pixel[2] > split):
				b = 255

			newpixel[i,j] = (r,g,b)

	return newimage

def normalise(image):
	print("normalising...")
	width,height = image.size
	newimage = Image.new("RGB", (width, height), "white")
	newpixel = newimage.load()
	maxbright = 0
	minbright = 765
	
	for i in range(width):
		for j in range(height):
			pixel = getPixel(image,(i,j))
			maxbright = max(maxbright, sumRGB(pixel))
			minbright = min(minbright, sumRGB(pixel))
	
	if (maxbright > 0):
		maxbright = 765/maxbright
	else:
		maxbright = 255
	minbright = minbright/3

	for i in range(width):
		for j in range(height):
			pixel = getPixel(image,(i,j))
			newpixel[i,j] = (int(maxbright*(pixel[0]-minbright)), int(maxbright*(pixel[1]-minbright)), int(maxbright*(pixel[2]-minbright)))

	return newimage

def drawEllipseMask(xy, quality=10):
	newimage = Image.new("RGBA", (xy[1][0]-xy[0][0], xy[1][1]-xy[0][1]), (0,0,0,0))
	newdrawing = ImageDraw.Draw(newimage)
	centre_xy = (xy[1][0]/2 + xy[0][0]/2, xy[1][1]/2 + xy[0][1]/2)
	for i in range(quality):
		pos = (i/quality)
		new_xy = ((centre_xy[0]*(1-pos)+xy[0][0]*pos, centre_xy[1]*(1-pos)+xy[0][1]*pos),
				  (centre_xy[0]*(1-pos)+xy[1][0]*pos, centre_xy[1]*(1-pos)+xy[1][1]*pos))
		newdrawing.ellipse(new_xy, fill=(255,255,255,255/10))

def drawPolygonMask(xy, size):
	newimage = Image.new("RGBA", size, (0,0,0,0))
	newdrawing = ImageDraw.Draw(newimage)
	fake_quad = np.array((xy[0], xy[1], xy[2], xy[2]))	#for some reason PIL only likes quads and not triangles, so I convert it to a fake quad
	newdrawing.polygon(fake_quad, fill=(255,255,255,255), outline=(128,128,128,128))
	return newimage

def getDelaunay(points):
	nppoints = np.array(points)
	return Delaunay(nppoints).points

def polygonCrop(image, xy):
	mask = drawPolygonMask(xy, image.size)
	new_mask, new_image = equalSize(mask, image)
	return ImageChops.composite(new_image,image,new_mask)

def transformTriangle(image, xy, target_xy):
	cropped_image = polygonCrop(image, xy)
	coefficients = (target_xy[0][0], target_xy[0][1], target_xy[1][0], target_xy[1][1],
					target_xy[2][0], target_xy[2][1], target_xy[2][0], target_xy[2][1])
	new_image = cropped_image.transform(image.size, Image.PERSPECTIVE, coefficients, resample=Image.BILINEAR)
	return ImageChops.composite(image, new_image, new_image)

def sortPointListByDistance(points, centre):	#sort with the points closest to the centre first
	p_num = len(points)
	p_dist2 = [0]*p_num
	for i in range(p_num):
		dist2 = (points[i][0]-centre[0])**2 + (points[i][1]-centre[1])**2
		p_dist2[i] = dist2

	new_p = points
	for j in range(p_num):
		furthest = -1
		furthest_i = -1
		for i in range(len(points)):
			if (p_dist2[i] > furthest):
				furthest = p_dist2[i]
				furthest_i = i
		p_dist2[furthest_i] = -1
		new_p[p_num-j-1] = points[furthest_i]

	return new_p


def matchPointLists(a, b):
	#find the list with more elements and the one with fewer elements
	less = a
	more = b
	swapped = False
	if (len(b) < len(a)):
		less = b
		more = a
		swapped = True

	more_matched = []
	for j in range(len(more)):
		more_matched.append(-1)	#stores which indices of the larger  have been matched
	matches = []

	for i in range(len(less)):	#first, go through the smaller array and match every element to something
		nearest = 1000
		nearestj = -1
		for j in range(len(more)):
			dist = abs(less[i][0]-more[j][0])+abs(less[i][1]-more[j][1])
			if (more_matched[j] < 0 or dist < more_matched[j]):
				if (dist < nearest):
					nearest = dist
					nearestj = j
				
		more_matched[nearestj] = nearest
		if swapped:
			matches.append((nearestj, i))
		else:
			matches.append((i,nearestj))

	for j in range(len(more)):	#second pass to match all the as-of-yet unmatched elements of 'more'
		if (more_matched[j] < 0):
			nearest = 1000
			nearesti = -1
			for i in range(len(less)):
				dist = abs(less[i][0]-more[j][0])+abs(less[i][1]-more[j][1])
				if (dist < nearest):
					nearest = dist
					nearesti = i

			if swapped:
				matches.append((j, nearesti))
			else:
				matches.append((nearesti, j))
	return matches

#interpolate two sets of point lists, 0 < pos < 1. 'matches' is an array of tuples (i,j) where 'i' is an index of a_p and 'j' is an index of b_p
def interpolatePointLists(a_p, b_p, matches, pos):
	newlist = []
	for m in matches:
		newlist.append((a_p[m[0]][0]*(1-pos) + b_p[m[1]][0]*pos, a_p[m[0]][1]*(1-pos) + b_p[m[1]][1]*pos))
	return newlist

def clampToSize(coord, size):
	return (max(min(coord[0], size[0]-1), 0), max(min(coord[1], size[1]-1),0))

def interpolateWithDots(a_pixel,b_pixel,size,a_p,b_p,matches,pos): #expects a and b to be same-sized images
	print("interpolating two images using dots...")

	if pos == 0 or pos == 1:
		return Image.new("RGBA", size, (255,255,255,0))

	close_to_b = False
	if (pos > 0.5):
		close_to_b = True

	blend_pos = min(max(pos*2-0.5,0),1)

	new_image = Image.new("RGBA", size, (255,255,255,0))
	new_drawing = ImageDraw.Draw(new_image)

	points = interpolatePointLists(a_p,b_p,matches,pos)
	m = matches

	for i in range(len(points)):
		a_p_i = a_p[m[i][0]]
		b_p_i = b_p[m[i][1]]
		dist2_p = 0
		if close_to_b:
			dist2_p = (int(abs(b_p_i[0] - points[i][0]) + abs(b_p_i[1] - points[i][1])))
		else:
			dist2_p = (int(abs(a_p_i[0] - points[i][0]) + abs(a_p_i[1] - points[i][1])))
		dist2_p = dist2_p/2
		dist2_p_07 = dist2_p*0.7
		dist2_p_15 = dist2_p*1.5

		for u in range(int(-dist2_p), int(dist2_p)):
			for v in range(int(-dist2_p), int(dist2_p)):
				uv_dist2 = abs(u)+abs(v)
				if (uv_dist2 <= dist2_p_15):
					if (uv_dist2 <= dist2_p_07):
						x = u
						y = v
					else:
						x = abs(u*3)-dist2_p_07
						y = abs(v*3)-dist2_p_07
						x = math.copysign(u,x)
						y = math.copysign(v, y)

					a_coord = clampToSize((a_p_i[0]+x, a_p_i[1]+y), size)
					b_coord = clampToSize((b_p_i[0]+x, b_p_i[1]+y), size)
					coord = clampToSize((points[i][0]+x, points[i][1]+y), size)

					alpha = int(256-200*float(uv_dist2)/dist2_p)
					new_colour = blendColors(a_pixel[a_coord], b_pixel[b_coord], blend_pos, alpha)
					new_drawing.point(coord, fill=new_colour)

		print("drew dot " + str(i) + "/" + str(len(points)) + " of size " + str(dist2_p*2), end="\t\t\t", flush=True)
	return new_image


def drawImageFromPoints(pointlist):
	print("saving image drawn from points...")
	width,height = reduce_width,reduce_height
	newimage = Image.new("RGB", (width,height), "white")
	newdrawing = ImageDraw.Draw(newimage)

	for i in range(len(pointlist)):
		pointa = pointlist[i]
		x = pointa[0]
		y = pointa[1]
		newdrawing.ellipse([(x-1, y-1), (x+1, y+1)], fill=(1,1,1))

	del newdrawing

	return newimage

def getPointsFromAutotrace(image, output_scale=(1,1)):
	if not os.path.exists("./autotrace_temp"):
		os.makedirs("./autotrace_temp")
	print("saving image as .bmp format...")
	width,height = image.size
	image.save("./autotrace_temp/input.bmp", "BMP")

	print("starting autotrace to get points from image... (please make sure autotrace is installed)")
	cmd = "autotrace --centerline --color-count=2 --output-file=./autotrace_temp/output.gnuplot --output-format=gnuplot ./autotrace_temp/input.bmp"
	args = shlex.split(cmd)
	subprocess.run(args)

	print("getting autotraced image and converting to a list of points...")

	pointlist = []
	plot = open("./autotrace_temp/output.gnuplot", "r")
	plotlines = plot.readlines()
	for p in plotlines:
		if p[0].isdigit():
			twostrings = p.split()
			pointtuple = (output_scale[0]*float(twostrings[0]), output_scale[1]*(height-float(twostrings[1])))	#the y value is inverted in the gnuplot
			pointlist.append(pointtuple)
	plot.close()

	print("deleting temporary files...")
	os.remove("./autotrace_temp/input.bmp")
	os.remove("./autotrace_temp/output.gnuplot")

	return pointlist


