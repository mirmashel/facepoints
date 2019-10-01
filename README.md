# facepoints
This programm and model predicts feature keypoints on human face in real-time

facepoint_model - model trained on pictures with face (coder - decoder)
cam.py - script which runs computer cam, and on each frame finds each face on frame with cv2.CascadeClassifier and for each face runs facepoint_model.hdf5 to find it keypoints(14 points)
