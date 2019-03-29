import cv2
import numpy as np
import csv

filename="190318072620-190318072700.avi"
cap = cv2.VideoCapture(filename)
count = 0

def contrast(img, param):
    look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
    for i in range(256):
        look_up_table[i] = 255.0 / (1+np.exp(-param*(i-128)/255));
    return cv2.LUT(img, look_up_table)

def gamma(img,gamma):
    look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0
    for i in range(256):
        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    # ガンマ変換後の出力
    return cv2.LUT(img, look_up_table)


if __name__ == '__main__':
    with open('test.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        output = ["count","x","y"]
        writer.writerow(output)
        while(cap.isOpened()):###変更
            # Capture frame-by-frame
            ret, img = cap.read()

            #γとコントラストの比較
            img = gamma(img,1.5)#書き換えチェック
            img = contrast(img,4)#書き換えチェック

            cv2.imshow('img',img)
            #pink
            if ret:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower = np.array([150, 100, 100])   
                upper = np.array([175, 255, 255])
                img_mask = cv2.inRange(hsv, lower, upper)
                img_color = cv2.bitwise_and(img, img, mask=img_mask)

                
                #kernel = np.ones((5,5),np.uint8)
                #img_color = cv2.dilate(img_color,kernel,iterations = 1)

                #□の座標を取る
                show_img = img_color.copy()
                img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)        
                image, contours, _ = cv2.findContours(img_color, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                rects = []
                for contour in contours:
                    approx = cv2.convexHull(contour)
                    rect = cv2.boundingRect(approx)
                    rects.append(np.array(rect))

                if len(rects) > 0:
                    rect = max(rects, key=(lambda x: x[2] * x[3]))
                    if(rect[2]>20 and rect[3]>20):
                        cv2.rectangle(show_img, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0, 0, 255), thickness=2)        
                        output = [count,  rect[2]/2+rect[0],  rect[3]/2+rect[1]]
                        writer.writerow(output)   
                # Display the resulting frame
                cv2.imshow('frame',show_img)
                count += 1
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break






        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()