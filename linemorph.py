from PIL import Image, ImageDraw, ImageChops
from scipy.spatial import Delaunay
import numpy as np
import os
import math
import subprocess
import shlex

M_PI = 3.14
reduce_width = 400
reduce_height = 300

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

def blendColors(colora, colorb, colorc, pos, pixel_alpha):
	alpha_pos =  float(pixel_alpha)/255
	newr = int((colora[0]*(1-pos) + colorb[0]*pos)*alpha_pos + colorc[0]*(1-alpha_pos))
	newg = int((colora[1]*(1-pos) + colorb[1]*pos)*alpha_pos + colorc[1]*(1-alpha_pos))
	newb = int((colora[2]*(1-pos) + colorb[2]*pos)*alpha_pos + colorc[2]*(1-alpha_pos))

	return (newr, newg, newb, 255)

def reduceSize(image):
	print("resizing to " +str(reduce_width)+"x"+str(reduce_height)+"...")
	return image.resize((reduce_width, reduce_height), resample=Image.BILINEAR)

def equalSize(a, b):
	aw, ah = a.size
	bw, bh = b.size
	if (bw == aw and bh == ah):
		a_out = Image.new("RGBA", (bw,bh), (0,0,0,0))
		a_out.paste(a)
		b_out = Image.new("RGBA", (bw,bh), (0,0,0,0))
		b_out.paste(b)
		return a_out,b_out

	ow, oh = a.size
	if (bw > aw):
		ow = bw
	if (bh > ah):
		oh = bh
	print("resizing both images to " +str(ow)+"x"+str(oh)+"...")

	a_out = Image.new("RGBA", (ow,oh), (0,0,0,0))
	a_out.paste(a.resize((ow,oh), resample=Image.BILINEAR))
	b_out = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
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
			
			if (g > 128):
				newdrawing.ellipse([(i-line_width, j-line_width), (i+line_width, j+line_width)], fill=(0,0,0))

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

#similar to above, but work on the pixels of two images.
def interpolateImageFromPointLists(a, b, a_p, b_p, a_d, b_d, matches, pos):
	print("interpolating two images based on points...")

	if pos == 0:
		return a
	elif pos == 1:
		return b

	close_to_b = False
	close_d = a_d
	close = a
	if (pos > 0.5):
		close_to_b = True
		close_d = b_d
		close = b

	triangles = getDelaunay(interpolatePointLists(a_p, b_p, matches, pos))
	base = Image.blend(a, b, pos)
	new_image = base
	tri_accumulator = []
	close_d_accumulator = []
	counter = 0
	for i in range(len(triangles)):
		tri_accumulator.append(triangles[i])
		if i < len(close_d):
			close_d_accumulator.append(close_d[i])
		if i%3 == 2:
			if (len(close_d_accumulator) == 3 and len(tri_accumulator) == 3):
				new_triangle = transformTriangle(base, close_d_accumulator, tri_accumulator)
				blend_pos = math.cos(-3*abs(pos-0.5))	#show the polygon mostly in the middle of the interpolation
				new_image = ImageChops.blend(new_image, new_triangle, blend_pos)
				print("created " + str(counter) + "th triangle")
				counter = counter+1
			tri_accumulator.clear()
			if i < len(close_d):
				close_d_accumulator.clear()

	return new_image

def interpolateImageWithEllipses(a,b,a_p,b_p,matches,pos):
	print("interpolating two images using ellipses...")

	if pos == 0:
		return a
	elif pos == 1:
		return b

	close_to_b = False
	close_p = a_p
	close = a
	if (pos > 0.5):
		close_to_b = True
		close_p = b_p
		close = b

	base = Image.blend(a,b,pos)
	points = interpolatePointLists(a_p,b_p,matches,pos)
	m = matches
	new_image = Image.new("RGBA", base.size, (0,0,0,0))
	new_image.paste(base)
	counter = 0
	for i in range(len(points)):
		counter = counter+1
		coord = points[i]
		a_p_i = a_p[m[i][0]]
		b_p_i = b_p[m[i][1]]
		delta = (b_p_i[0]-a_p_i[0], b_p_i[1]-a_p_i[1])
		centre = (a_p_i[0]+delta[0]/2, a_p_i[1]+delta[1]/2)
		perp = math.atan2(delta[1], delta[0])-M_PI/2
		mag2 = delta[0]*delta[0] + delta[1]*delta[1]
		perp_edge = (centre[0] + math.cos(perp)*mag2, centre[1] + math.sin(perp)*mag2,
					 centre[0] - math.cos(perp)*mag2, centre[1] - math.sin(perp)*mag2)
		cropped = polygonCrop(base, perp_edge)
		blend_pos = math.cos(-3*abs(pos-0.5))	#show the polygon mostly in the middle of the interpolation
		new_image = ImageChops.blend(new_image, cropped, blend_pos)
		print("created ellipse" + str(counter) + "/" + str(len(points)), end="\t")
	print("DONE")
	return new_image

def interpolateWithDots(a,b,a_p,b_p,matches,pos):
	print("interpolating two images using dots...")

	if pos == 0:
		return a
	elif pos == 1:
		return b

	close_to_b = False
	close_p = a_p
	close = a
	if (pos > 0.5):
		close_to_b = True
		close_p = b_p
		close = b

	blend_alpha = min(max(pos*2-0.5,0),1)
	base = Image.blend(a,b,blend_alpha)

	new_image = Image.new("RGBA", base.size, (0,0,0,0))
	new_image.paste(base)
	new_drawing = ImageDraw.Draw(new_image)

	points = interpolatePointLists(a_p,b_p,matches,pos)
	m = matches

	for i in range(len(points)):
		a_p_i = a_p[m[i][0]]
		b_p_i = b_p[m[i][1]]
		dist2_p = 0
		if close_to_b:
			dist2_p = int(abs(b_p_i[0] - points[i][0]) + abs(b_p_i[1] - points[i][1]))
		else:
			dist2_p = int(abs(a_p_i[0] - points[i][0]) + abs(a_p_i[1] - points[i][1]))

		for x in range(int(-dist2_p), int(dist2_p+1)):
			for y in range(int(-dist2_p), int(dist2_p+1)):
				if (abs(x)+abs(y) <= 2*dist2_p+1):
					coord = (points[i][0]+x, points[i][1]+y)
					pixel_alpha = min(8*max(2*float(dist2_p)-float(abs(x)+abs(y)),0),255)
					a_coord = (a_p_i[0]+x, a_p_i[1]+y)
					b_coord = (b_p_i[0]+x, b_p_i[1]+y)
					a_colour = getPixel(a, a_coord)
					a_colour = (a_colour[0], a_colour[1], a_colour[2], pixel_alpha)
					b_colour = getPixel(b, b_coord)
					b_colour = (b_colour[0], b_colour[1], b_colour[2], pixel_alpha)
					c_colour = getPixel(new_image, coord)
					new_colour = blendColors(a_colour, b_colour, c_colour, blend_alpha, pixel_alpha)
					new_drawing.point(coord, fill=new_colour)

		print("drew dot " + str(i) + "/" + str(len(points)) + " of size " + str(dist2_p), end="\t\t", flush=True)
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


