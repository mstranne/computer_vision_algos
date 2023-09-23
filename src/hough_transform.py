import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# Hough transform is used asa voting scheme in CV, mostly for detection of shapes with
# a good mathematical representation eg. in 2D Lines or Circles, but can also be applied in 3D
# see: https://www.analyticsvidhya.com/blog/2022/06/a-complete-guide-on-hough-transform/

input_folder = 'input_imgs'

# read input image
#img = cv.imread(os.path.join(input_folder, 'road1.jpg'))
img = cv.imread(os.path.join(input_folder, 'img_x.png'))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.medianBlur(img, 5)

img_h, img_w, _ = img.shape

plt.imshow(img)
plt.show()

#do some edge detection
edge_img = cv.Canny(img,190,200)
plt.imshow(edge_img)
plt.show()

# now we want to find Lines with the given edge detection
# lines can be described in different ways in 2D
# one way would be y=mx+c
# --- Hough Space ----
# we can create a Hough Space for this line equation by using m & c as axis,
# now every point on the line y=mx+c would be represented by the same point in the
# hough space [m, c]. The idea is now to have a count array aka. Bins to count points
# in hough space that go into same bin, all maxima Bins should correspond to Lines in
# the image, the bigger the bin -> the more noise can be on the line and still fall in same bin
# however this can also lead to detection of different line in the same bin.
# Nevertheless, in this line example the representation m goes to infinity for vertical lines.
# therefore we use xcosθ+ysinθ=r to create the Hough space for line detection [θ, r]
# θ is in range -90 - +90 so we can build every possible line
# p is the distance from origin, and therefor capped with 0 to sqrt(x^2 + y^2) which is the diagonal of the image
# now we can make our voting space with θ and r
# given x & y is an edge point -> we can now increase every bin which lays in xcosθ+ysinθ=r
# so for every point we have a curve in hough space
# all maxima, where loads od curves intersect should correspod to a good line guess in the image

#define backed size and create hough space
angle_step= 1 # all 1 degree
distance_step = 4 # like 3 pixel
diag = int(np.sqrt(img_h**2 + img_w**2))
hough_space_h = int(diag/distance_step)
hough_space_w = int(180/angle_step)
hough_space = np.zeros((hough_space_h, hough_space_w))

#non optimized but easy to understand voting scheme
#now we loop over the edge image and search for edge pixels

for x in range(0, img_h):
    for y in range(0, img_w):
        if (edge_img[x,y] == 255):
            # we have an edge pixel
            # now for every possible θ we compute the corresponding r
            # and increase the hough space value
            for theta in np.arange (-90, 90, angle_step) :
                r = x * np.cos(theta/180*np.pi) + y * np.sin(theta/180*np.pi)
                if (r < diag):
                    hough_space[int(r/distance_step), int(theta/angle_step)+90] += 1

# now lets have a look at the hough space
plt.imshow(hough_space)
plt.show()

# see the maxima should correspond to the lines
max_count = 2

# lets check some maxima
max_idxs = np.argsort((-hough_space).flatten())
idxs = []
cnt = 0
while cnt < max_count:
    w = int(max_idxs[cnt] / hough_space_w)
    h = max_idxs[cnt] % hough_space_w
    idxs.append((w,h))
    theta = (h-90) * angle_step / 180*np.pi
    r = w * distance_step

    #xcosθ+ysinθ=r -> y = (r-xcosθ)/sinθ
    start_point = np.array((int((r)/np.sin(theta)),0))
    end_point = np.array((int((r-img_w*np.cos(theta))/np.sin(theta)), img_w))

    cv.line(img, start_point, end_point, (255,0,0), 2)
    cnt+=1

# result these lines are still infinit, a possibility would be to check back where edge points where,
# to limit the length. However, this is also a really good thing with this algorithm,
# imagine a line or a circle which is partly hidden, here we would also calculate the hidden parts
#

plt.imshow(img)
plt.show()

print('fin')
