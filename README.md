# facepoints
facepoints detection
facepoint_model - model trained on pictures with face (coder - decoder)
cam.py - script which runs computer cam, and on each frame finds each face on frame with cv2.CascadeClassifier and for each face runs facepoint_model.hdf5 to find it keypoints(14 points)
