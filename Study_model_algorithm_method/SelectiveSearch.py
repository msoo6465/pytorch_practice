import skimage
import selective_search
import cv2

image = skimage.data.astronaut()

# Propose boxes
# mode = ['single','fast','quality']
boxes = selective_search.selective_search(image, mode='quality', random_sort=True)

# Filter box proposals
boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=80)

print(boxes_filter)
for box in boxes_filter:
    qw=cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
    cv2.imshow('as',qw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()