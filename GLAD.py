"""
Created on 2023/1/1
@Author: Guo Hanqing
@Email : guohanqing@westlake.edu.cn
@File : GLAD.py
"""

import cv2
import numpy as np
import time

from detector_trt import Detector
from detector2_trt import Detector2
import ctypes

from MOD2 import MOD2_global
from MOD2 import MOD2_local

from Functions import enlarge_region
from Functions import enlarge_region2
from Functions import readGTbox
from Functions import cal_iou

PLUGIN_LIBRARY = "./weights/libmyplugins.so"
ctypes.CDLL(PLUGIN_LIBRARY)
engine_file_path1 = './weights/yolov5s_DT-Drone2.engine'
engine_file_path2 = './weights/yolov5s_DT-Drone2-crop.engine'
detector1 = Detector(engine_file_path1)
detector2 = Detector2(engine_file_path2)

video_name = 'phantom09'
cap = cv2.VideoCapture('./input/' + video_name + '.mp4')


count = 0
# interval = 500
flag = 0
prveframe = None
local_num = 0
a = 150
TP = 0
FN = 0
FP = 0
TN = 0
# fps = 0
total_fps = []
x2 = 0
y2 = 0
w2 = 0
h2 = 0
border = 1
border1 = 3

# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# filename = "/home/user-guo/Documents/YOLOv5_MOD_Tracking/output/New_Domain/" + video_name + '.mp4'
# vw = cv2.VideoWriter(filename, fourcc, int(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))

# XML文件的地址
# anno_path = "/home/user-guo/data/drone-dataset/phantom-dataset/Annotations/" + video_name + "/"

while cap.isOpened():
    ret, frame = cap.read()
    # print(ret)
    if not ret:
        break

    if prveframe is None:
        print('first frame input')
        prveframe = frame
        count = count + 1
        continue

    frame_show = frame.copy()
    width = frame.shape[1]
    height = frame.shape[0]
    # file_count = str(count + 1)
    # filename = video_name + '_' + file_count.zfill(4) + '.xml'
    # xml_file = anno_path + filename
    # box_GT = readGTbox(xml_file)
    # x3 = box_GT[0]
    # y3 = box_GT[1]
    # w3 = box_GT[2]
    # h3 = box_GT[3]
    t1 = time.time()

    if flag == 0:
        boxes = detector1.detect(frame)
        if len(boxes) == 0:
            # if prveframe is None:
            #     print('first frame input')
            #     prveframe = frame
            #     continue
            boxes_MOD = MOD2_global(prveframe, frame)
            # print(boxes_MOD)
            if len(boxes_MOD) != 0:
                (x, y) = (boxes_MOD[0], boxes_MOD[1])
                (w, h) = (boxes_MOD[2], boxes_MOD[3])
                init_rect = [x, y, w, h]
                # 画出边框和标签
                color = (255, 0, 0)
                cv2.rectangle(frame_show, (x - border1, y - border1), (x + w + border1, y + h + border1), color, border, lineType=cv2.LINE_AA)
                # cv2.putText(frame_show, "Global MOD Detection Success", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                # cv2.putText(frame_show, "Frame: {}".format(count), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow('Detection', frame_show)
                # print('MOD Initialize Success:', init_rect)
                flag = 1
                local_num = 0
                # x1, y1, w1, h1 = enlarge_region(x, y, w, h, a, width, height)
                x1, y1, w1, h1 = enlarge_region2(x, y, a, width, height)
                x2 = x - x1
                y2 = y - y1
                w2 = w
                h2 = h
                status = 'Global MOD'
            else:
                # cv2.putText(frame_show, "Global YOLO and MOD Failed", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # cv2.putText(frame_show, "Frame: {}".format(count), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow('Detection', frame_show)
                flag = 0
                init_rect = []
                status = 'Both Failure'
                # init_rect = np.array([0, 0, 0, 0])
        else:
            (x, y) = (boxes[0], boxes[1])
            (w, h) = (boxes[2], boxes[3])
            init_rect = [x, y, w, h]
            color = (0, 255, 255)
            # cv2.putText(frame_show, "Global YOLO Detection Success", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            # cv2.putText(frame_show, "Frame: {}".format(count), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame_show, (x - border1, y - border1), (x + w + border1, y + h + border1), color, border, lineType=cv2.LINE_AA)
            # cv2.imshow('Detection', frame_show)
            # print('YOLO Initialize Success:', init_rect)
            local_num = 0
            flag = 1
            # x1, y1, w1, h1 = enlarge_region(x, y, w, h, a, width, height)
            x1, y1, w1, h1 = enlarge_region2(x, y, a, width, height)
            x2 = x - x1
            y2 = y - y1
            w2 = w
            h2 = h
            status = 'Global YOLO'
    else:
        track_crop1 = prveframe[y1:y1 + h1, x1:x1 + w1, :]
        track_crop2 = frame[y1:y1 + h1, x1:x1 + w1, :]
        # boxes = YOLO2(track_crop2)
        x_prev = x2 + w2 / 2
        y_prev = y2 + h2 / 2
        boxes = detector2.detect(track_crop2, x_prev, y_prev)
        # boxes = detector2.detect(track_crop2)
        if len(boxes) == 0:
            boxes_MOD = MOD2_local(track_crop1, track_crop2, x_prev, y_prev)
            # print(boxes_MOD)
            if len(boxes_MOD) != 0:
                (x2, y2) = (boxes_MOD[0], boxes_MOD[1])
                (w2, h2) = (boxes_MOD[2], boxes_MOD[3])
                init_rect = [x2 + x1, y2 + y1, w2, h2]
                # 画出边框和标签
                color = (255, 0, 0)
                cv2.rectangle(frame_show, (x2 + x1 - border1, y2 + y1 - border1), (x2 + x1 + w2 + border1, y2 + y1 + h2 + border1), color, border, lineType=cv2.LINE_AA)
                # cv2.putText(frame_show, "Local MOD Success", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                # cv2.putText(frame_show, "Frame: {}".format(count), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # cv2.rectangle(frame_show, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 2, lineType=cv2.LINE_AA)
                # cv2.putText(frame_show, "search region", (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # # cv2.imshow('Detection', frame_show)
                # print('MOD Re-detection Success:', init_rect)
                local_num = 0
                flag = 1
                # x1, y1, w1, h1 = enlarge_region(x2 + x1, y2 + y1, w2, h2, a, width, height)
                x1, y1, w1, h1 = enlarge_region2(x2 + x1, y2 + y1, a, width, height)
                status = 'Local MOD'
            else:
                # cv2.putText(frame_show, "Local YOLO and MOD Failed", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                # cv2.putText(frame_show, "Frame: {}".format(count), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow('Detection', frame_show)
                init_rect = []
                status = 'Local Both Failure'
        else:
            (x2, y2) = (boxes[0], boxes[1])
            (w2, h2) = (boxes[2], boxes[3])
            init_rect = [x2 + x1, y2 + y1, w2, h2]
            # color = (0, 255, 255)
            color = (0, 255, 255)
            cv2.rectangle(frame_show, (x2 + x1 - border1, y2 + y1 - border1), (x2 + x1 + w2 + border1, y2 + y1 + h2 + border1), color, border, lineType=cv2.LINE_AA)
            # cv2.putText(frame_show, "Local YOLO Detection Success", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
            # cv2.putText(frame_show, "Frame: {}".format(count), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # cv2.rectangle(frame_show, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 2, lineType=cv2.LINE_AA)
            # cv2.putText(frame_show, "search region", (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imshow('Detection', frame_show)1
            # print('YOLO Re-detection Success:', init_rect)
            local_num = 0
            flag = 1
            # x1, y1, w1, h1 = enlarge_region(x2 + x1, y2 + y1, w2, h2, a, width, height)
            x1, y1, w1, h1 = enlarge_region2(x2 + x1, y2 + y1, a, width, height)
            status = 'Local YOLO'
        local_num = local_num + 1
        if local_num == 30:
            print('turn to global detection')
            color0 = (0, 0, 255)
            # cv2.putText(frame_show, "turn to global detection", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color0, 2)
            flag = 0

    # if len(init_rect) != 0:
    #     init_rect = np.array(init_rect)
    #     if len(box_GT) != 0:
    #         iou = cal_iou(box_GT, init_rect)
    #         if iou > 0.5:
    #             TP = TP + 1
    #             box_status = 'TP'
    #         else:
    #             FP = FP + 1
    #             box_status = 'FP'
    #     else:
    #         FP = FP + 1
    # else:
    #     if len(box_GT) != 0:
    #         FN = FN + 1
    #         box_status = 'FN'
    #     else:
    #         TN = TN + 1
    #         box_status = 'TN'

    fps = (1. / (time.time() - t1))
    total_fps.append(fps)
    # print('frame per second: ', fps)
    cv2.putText(frame_show, "Frame: {}".format(count), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(frame_show, 'FPS: {:.2f}'.format(fps), (50, 30), 0, 1, (0, 0, 255), 2)
    # cv2.putText(frame_show, box_status, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    color2 = (0, 0, 255)
    # cv2.rectangle(frame_show, (x3 - border1, y3 - border1), (x3 + w3 + border1, y3 + h3 + border1), color2, border, lineType=cv2.LINE_AA)
    cv2.imshow('Detection', frame_show)
    # vw.write(frame_show)
    print('frame count: %d fps: %.2f' % (count, fps), end=' ')
    print(status, end=' ')
    # print(box_status, end=' ')
    print('bbox:', init_rect)
    count = count + 1
    prveframe = frame
    key = cv2.waitKey(10) & 0xff

    if key == 27 or key == ord('q'):
        break

# FPS = np.array(total_fps)
# FPS_ave = np.mean(FPS)
# print(video_name)
# print('average fps: ', FPS_ave)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 = 2 * Precision * Recall / (Precision + Recall)
# print("True Positive: ", TP)
# print("False Negative: ", FN)
# print("False Positive: ", FP)
# print("True Negative: ", TN)
# print("Precision: ", Precision)
# print("Recall: ", Recall)
# print("F1 Score: ", F1)
cap.release()
# vw.release()
cv2.destroyAllWindows()



