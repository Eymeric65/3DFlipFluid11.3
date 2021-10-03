from PIL import ImageGrab
import numpy as np
import cv2

t=0
while 1 :
    t+=1

    img = ImageGrab.grab(bbox=(100,100,1700,880)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_np = np.array(img) #this is the array obtained from conversion
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", img_np)
    cv2.imwrite('D:/bureau/prepa/TIPE/aide en tout genre/frame/pic'+str(t)+".jpeg", img_np)
    cv2.waitKey(1)

cv2.destroyAllWindows()