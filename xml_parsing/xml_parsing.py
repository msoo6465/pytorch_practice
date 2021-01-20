from xml.etree.ElementTree import parse
import cv2
import os
import numpy as np
from PIL import Image
from pprint import pprint

xml_path = 'D:/data/testworks/02. 2차/02. xml'
img_path = 'D:/data/testworks/02. 2차/01. image'
actions = [os.path.join(xml_path,i)for i in os.listdir(xml_path)]
font = cv2.FONT_HERSHEY_SIMPLEX
# print(actions)
for action in actions:
    times = [os.path.join(action,i)for i in os.listdir(action)]
    for time in times:
        aids = [os.path.join(time,i)for i in os.listdir(time)]
        for aid in aids:
            places = [os.path.join(aid,i)for i in os.listdir(aid)]
            for place in places:
                xmls = [os.path.join(place,i)for i in os.listdir(place)]
                for xml in xmls:
                    tree = parse(xml)
                    root = tree.getroot()
                    img_tags = root.findall('image')
                    for img_tag in img_tags:
                        img_name = img_tag.attrib['name']
                        print(img_name)
                        img_rout = os.path.join(img_path,'\\'.join(xml.split('\\')[1:]))
                        img_rout = '\\'.join(img_rout.split('\\')[:-1])
                        # img_rout = os.path.join(img_rout,img_name)
                        if 'spring' in xml:
                            weather = '봄'
                        elif 'summer' in xml:
                            weather = '여름'
                        elif 'winter' in xml:
                            weather = '겨울'
                        else:
                            print(xml)
                            exit()
                        for dir in os.listdir(img_rout):
                            if weather in dir:
                                img_path_1 = os.path.join(img_rout, dir)
                                img_path_1 = os.path.join(img_path_1,img_name)

                        im = np.array(Image.open(img_path_1))
                        for child in img_tag.getchildren():
                            xbr = int(float(child.attrib['xbr']))
                            ybr = int(float(child.attrib['ybr']))
                            xtl = int(float(child.attrib['xtl']))
                            ytl = int(float(child.attrib['ytl']))
                            label = child.attrib['label']
                            cv2.rectangle(im,(xbr,ybr),(xtl,ytl),(0,255,0),3)
                            cv2.putText(im, label, (xbr-(xbr-xtl), ybr), font, 2, (255, 255, 255), 2)
                        cv2.imshow('img',im)
                        cv2.waitKey(0)




# tree = parse