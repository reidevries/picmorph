from PIL import Image
import math

def open_image(path):
    newImage = Image.open(path)
    return newImage

def save_image(image, path):
    image.save(path, 'png')

def create_image(i, j):
    image = Image.new("RGB", (i,j), "white")
    return image

def get_pixel(image, i, j):
    width, height = image.size
    if i>width or j>height:
        return None

    pixel = image.getpixel((i,j))
    return pixel

def sumrgb(rgb):
    return rgb[0]+rgb[1]+rgb[2]

def edgedetect(image):
    print("applying sobel edge detection...")
    width,height = image.size
    pixel = image.load()
    newimage = create_image(width, height)
    newpixel = newimage.load()
    
    for i in range(width):
        for j in range(height):
            il = max(i-1, 0)
            ir = min(i,width-1)
            ju = max(j-1, 0)
            jd = min(j,height-1)
            
            tl = sumrgb(pixel[il,ju])
            t = sumrgb(pixel[i,ju])
            tr = sumrgb(pixel[ir,ju])
            l = sumrgb(pixel[il,j])
            r = sumrgb(pixel[ir,j])
            bl = sumrgb(pixel[il,jd])
            b = sumrgb(pixel[i,jd])
            br = sumrgb(pixel[ir,jd])
            
            gx = abs(tr-tl+2*(r-l)+br-bl)
            gy = abs(tl-bl+2*(t-b)+tr-br)

            g = int(math.sqrt(gx*gx + gy*gy))
            
            newpixel[i,j] = (g,g,g)

    return newimage

def twotone(image, split=127):
    print("applying two tone filter to r,g,b channels with split point", split, "...")
    width,height = image.size
    newimage = create_image(width,height)
    newpixel = newimage.load()

    for i in range(width):
        for j in range(height):
            pixel = get_pixel(image,i,j)
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
    newimage = create_image(width,height)
    newpixel = newimage.load()
    maxbright = 0
    minbright = 765
    
    for i in range(width):
        for j in range(height):
            pixel = get_pixel(image,i,j)
            maxbright = max(maxbright, sumrgb(pixel))
            minbright = min(minbright, sumrgb(pixel))
    
    if (maxbright > 0):
        maxbright = 765/maxbright
    else:
        maxbright = 255
    minbright = minbright/3

    for i in range(width):
        for j in range(height):
            pixel = get_pixel(image,i,j)
            newpixel[i,j] = (int(maxbright*(pixel[0]-minbright)), int(maxbright*(pixel[1]-minbright)), int(maxbright*(pixel[2]-minbright)))

    return newimage

def reducePoints(pointlist, size=256):
    oldsize = len(pointlist)+1
    newsize = len(pointlist)
    while (oldsize > newsize):
        oldsize = newsize
        delinterval = max(0,len(pointlist)-size)
        if (delinterval > 0):
            delinterval = math.floor(len(pointlist)/delinterval)
            for k in range(len(pointlist)):
                bdx = 0
                bdy = 0
                adx = 0
                ady = 0
                if (k > 1 and k < len(pointlist)-1):
                    bdx = pointlist[k-1][0]-pointlist[k][0]
                    bdy = pointlist[k-1][1]-pointlist[k][1]
                    adx = pointlist[k][0]-pointlist[k+1][0]
                    ady = pointlist[k][1]-pointlist[k+1][1]
                    if (k%delinterval == 0):
                        if (((bdx > 0 and adx > 0) or (bdx < 0 and adx < 0)) and ((bdy > 0 and ady > 0) or (bdy < 0 and ady < 0))):
                            pointlist.remove(pointlist[k])
        newsize = len(pointlist)
    return pointlist

def annihilatePoints(pointlist, size=256):
    oldsize = len(pointlist)+1
    newsize = len(pointlist)
    while (oldsize > newsize):
        oldsize = newsize
        delinterval = max(0,len(pointlist)-size)
        if (delinterval > 0):
            delinterval = math.floor(len(pointlist)/delinterval)
            for k in range(len(pointlist)):
                if (k%delinterval == 0 and k < len(pointlist)):
                    pointlist.remove(pointlist[k])
        newsize = len(pointlist)
    return pointlist

def getPointList(image, listsize=256):
    print("getting a list of points representing the lines of the object...")
    print("\t", end=" ")
    newimage = edgedetect(image.resize((300,300)))
    print("\t", end=" ")
    newimage = normalise(newimage)

    print("\t creating point list...")
    pointlist = []
    for i in range(300):
        for j in range(300):                               #if the pixel is part of a "line", then try find the nearest point found so far, and insert it after it in the list
            if (sumrgb(get_pixel(newimage,i,j)) > 127):
                nearest = 600
                nearestk = 0
                for k in range(len(pointlist)):
                    dist = i-pointlist[k][0]+j-pointlist[k][1]
                    if (dist < nearest):
                         nearest = dist
                         nearestk = k
                pointlist.insert(nearestk, (i,j))

    print("\t reducing points...")
    
    reducePoints(pointlist, listsize)
    
    annihilatePoints(pointlist, listsize)
        
    return pointlist

def matchPointLists(a, b):
    matches = []
    for i in range(len(a)):
        nearest = 600
        nearestj = 0
        for j in range(len(b)):
            dist = a[i][0]-b[j][0]+a[i][1]-b[j][1]
            if (dist < nearest):
                nearest = dist
                nearestj = j
        matches.append((i,j))
    return matches

#interpolate two sets of point lists, 0 < pos < 1
def interpolatePointLists(a, b, matches, pos):
    newlist = a
    for m in matches:
        newlist[m[0]] = (a[m[0]][0]*(1-pos) + b[m[1]][0]*pos, a[m[0]][1]*(1-pos) + b[m[1]][1]*pos)
    return newlist

def drawImageFromPoints(pointlist):
    newimage = create_image(300,300)
    newpixel = newimage.load()

    for i in range(len(pointlist)):
        pointa = pointlist[i]
        if (i < len(pointlist)-1):
            pointb = pointlist[i+1]
            for j in range(300):
                interp = j/300.0
                x = pointa[0]*(1-interp) + pointb[0]*interp
                y = pointa[1]*(1-interp) + pointb[1]*interp
                newpixel[x,y] = (1,1,1)
    return newimage       
                
                
