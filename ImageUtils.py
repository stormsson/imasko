#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageChops
from PIL import ImageStat


def preprocessImage(img, blackThreshold=150):
    # enhance edges
    # img = img.filter(ImageFilter.EDGE_ENHANCE)

    # to black or white
    img = getBlackAndWhite(img, blackThreshold)

    #invert image
    # img = ImageOps.invert(img)
    return img

def preprocessMask(img, blackThreshold=150):
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    img = getPNGBWMask(img, blackThreshold)
    img = ImageOps.invert(img)

    return img

def splitImageInChunks(img, chunkWidth, chunkHeight):
    chunks = []
    imgwidth, imgheight = img.size

    for row in range(0, imgwidth, chunkWidth):
        for col in range(0, imgheight, chunkHeight):
            box = ( row,
                    col,
                    row + chunkWidth,
                    col + chunkHeight )
            # print "cropping %s,%s,%s,%s" % box

            chunks.append(img.crop(box))

    return chunks

def getBlackAndWhite(img, blackThreshold=150):
    gray = img.convert('L')
    bw = gray.point(lambda x: 0 if x<blackThreshold else 255)
    return bw

# given a png image returns a b/w image
def getPNGBWMask(img, blackThreshold=150):
    gray = img
    if img.mode == 'RGBA':
        r,g,b,a = img.split()
        gray = Image.merge('L',(a,))
    elif img.mode == 'LA':
        gray = img.convert('L')
    gray = gray.point(lambda x: 0 if x<blackThreshold else 255)
    return gray


def getDifferenceChunks(imageChunks, maskChunks):
    pairs = zip(imageChunks, maskChunks)
    chunks = []
    for p in pairs:
        # immagine - maschera
        differenceChunk = ImageChops.add(p[0],p[1])

        # maschera - immagine
        # differenceChunk = ImageChops.difference(p[1],p[0])
        chunks.append(differenceChunk)
        # print ImageStat.Stat(differenceChunk).mean
        # print "modes: %s , %s " % (p[0].mode, p[1].mode)
        # print "jpg chunk: %sx%s" % (p[0].size)
        # print "png chunk: %sx%s" % (p[1].size)

    return chunks

# true if
def isUselessImage(image):
    mean = ImageStat.Stat(image).mean
    if mean == 0.0 or mean == 255.0:
        return True

    return False

def isWhiterThan(image, threshold=250.0):
    return ImageStat.Stat(image).mean >= threshold

def isBlackererThan(image, threshold=5.0):
    return ImageStat.Stat(image).mean <= threshold