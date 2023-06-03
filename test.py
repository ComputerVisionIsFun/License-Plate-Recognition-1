import myDataset as md
import os 
import cv2
import matplotlib.pyplot as plt


axs = [16+i*32 for i in range(13)]
ays = [16+i*32 for i in range(13)]
xml_img_folder = '/Users/chiang-en/Desktop/LPR_dataset/xml/'
rWidth = 416
rHeight = 416
oWidth=1280
oHeight=720 

ds = md.myDataset(axs=axs,ays=ays,xml_img_folder=xml_img_folder,rWidth=rWidth,rHeight=rHeight,oWidth=oWidth,oHeight=oHeight)
ind = 40
ranno = ds.rannos[ind]
image_path = os.path.join(xml_img_folder, ranno['filename'])
objs = ranno['objs']
image = cv2.imread(image_path, 1)
rimage = cv2.resize(image, (rWidth,rHeight), 1)
for obj in objs:
    xmin, xmax = int(obj['xmin']), int(obj['xmax'])
    ymin, ymax = int(obj['ymin']), int(obj['ymax'])
    cv2.rectangle(rimage, (xmin, ymin), (xmax, ymax), (255,1,1),3,8,0)

# cv2.namedWindow('img', 2)
# cv2.imshow("img",rimage)
# cv2.waitKey(0)
plt.yticks([i for i in range(0, rHeight, 32)])
plt.xticks([i for i in range(0, rWidth, 32)])
plt.grid(color='black', alpha=.5)

for anchor in ds.anchors.anchors:
    plt.scatter(anchor[1], anchor[0], color='green', s = 3)


plt.imshow(rimage)

plt.show()


print(ranno)