# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

def yolo_image(open_img_file, yolo_directory, transparent):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=open_img_file,
        help="path to input image")
    ap.add_argument("-y", "--yolo", default=yolo_directory,
        help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    # COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    # 알파 채널까지 포함하여 색 랜덤 생성 후 알파 채널 불투명 처리
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 4), dtype="uint8")
    for i in range(0, len(COLORS)):
        COLORS[i][3] = 255

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    #image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED) # 이미지 파일을 alpha channel 까지 포함해 읽음.
    image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    if transparent:
        # 동일한 크기의 투명 이미지로 변경하기
        # 투명 이미지에 박스만 그릴 때 사용함
        # 알파 0: 완전 투명, 알파 255: 완전 불투명
        #image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        image = np.zeros((H, W, 4), dtype=np.uint8)
        #image = np.full((image.shape[0], image.shape[1], 4), 0, dtype=np.uint8)

    # ensure at least one detection exists
    objdata = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            objdata.append(LABELS[classIDs[i]])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

    # show the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    return image, objdata


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))

    yolo_directory = os.path.join(dir_path, 'yolo-coco')
    open_img_file = os.path.join(dir_path, 'input/baggage_claim.jpg')
    save_img_file = os.path.join(dir_path, 'output/test.png')
    transparent = True

    img, _ = yolo_image(open_img_file, yolo_directory, transparent)

    cv2.imwrite(save_img_file, img, [cv2.IMWRITE_PNG_COMPRESSION, 100])

    cv2.imshow("Image", img)
    cv2.waitKey(0)
