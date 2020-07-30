hogFaceDetector = dlib.get_frontal_face_detector()
faceRects = hogFaceDetector(frameDlibHogSmall, 0)
for faceRect in faceRects:
    x1 = faceRect.left()
    y1 = faceRect.top()
    x2 = faceRect.right()
    y2 = faceRect.bottom()



