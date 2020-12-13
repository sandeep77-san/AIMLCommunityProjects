import streamlit as st 
import cv2
import io
import tensorflow as tf
import time
import posenet
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def keypoints(choice):
    frameST = st.empty()
    with tf.Session() as sess:                
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']
        model = pickle.load(open("RNNmodel.pkl","rb"))
        col =['nose_xCoord', 'nose_yCoord','leftEye_xCoord', 'leftEye_yCoord', 'rightEye_xCoord', 'rightEye_yCoord', 'leftEar_xCoord', 'leftEar_yCoord', 'rightEar_xCoord', 'rightEar_yCoord', 'leftShoulder_xCoord', 'leftShoulder_yCoord', 'rightShoulder_xCoord', 'rightShoulder_yCoord', 'leftElbow_xCoord', 'leftElbow_yCoord', 'rightElbow_xCoord', 'rightElbow_yCoord', 'leftWrist_xCoord', 'leftWrist_yCoord', 'rightWrist_xCoord', 'rightWrist_yCoord', 'leftHip_xCoord', 'leftHip_yCoord', 'rightHip_xCoord', 'rightHip_yCoord', 'leftKnee_xCoord', 'leftKnee_yCoord', 'rightKnee_xCoord', 'rightKnee_yCoord', 'leftAnkle_xCoord', 'leftAnkle_yCoord', 'rightAnkle_xCoord', 'rightAnkle_yCoord']
        dummy_frame = pd.DataFrame(columns = col)
        Mean = np.array([148.66119491, 305.30049224, 142.6646119 , 310.25743596,
       141.70017197, 302.93952183, 141.27736617, 317.44867283,
       139.46369719, 297.02198297, 163.55299211, 318.43891733,
       161.08800759, 288.89443786, 201.92641401, 327.4968114 ,
       198.14737829, 268.80913136, 227.76026519, 321.95571328,
       222.86624951, 266.31670662, 231.37762059, 301.21041038,
       229.25918355, 278.40873512, 279.47660651, 299.97468079,
       275.04189452, 267.78987458, 332.20440178, 298.95615126,
       326.76802751, 265.76282502])
        Std = np.array([ 96.38779959, 127.16401286,  97.13470333, 129.87910693,
        95.74375047, 127.46249832,  93.7858246 , 129.05961629,
        90.41257012, 125.11638235,  87.46516144, 123.3528153 ,
        82.68437065, 115.98756097,  93.88162078, 126.86459209,
        88.92850539, 111.58051582, 106.44469339, 126.27066327,
       104.85191381, 111.8015252 ,  86.30759081, 119.30815504,
        84.94167523, 113.47366614,  91.67524166, 121.34462458,
        90.18384711, 115.57534719, 105.70870987, 131.18502572,
       104.6401585 , 126.9626246 ])
        if choice == 'Webcam':
            cap = cv2.VideoCapture(0)
            cap.set(3, 500)
            cap.set(4, 500)        
        elif choice == 'upload video':
            cap = cv2.VideoCapture('test.mp4')
        frame_count = 0
        result = 'Loading..'
        ### for writing text on frame
        font = cv2.FONT_HERSHEY_PLAIN 
        org = (0, 50)
        fontScale = 2
        color = (0, 0, 255) ### colour on BGR
        thickness = 2
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=0.7125, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,   ## detectiong only single position
                min_pose_score=0.15)
            keypoint_coords *= output_scale
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            #cv2.imshow('posenet', overlay_image)
 
            nose_xCoord = keypoint_coords[0][0][0]
            nose_yCoord = keypoint_coords[0][0][1]
            leftEye_xCoord = keypoint_coords[0][1][0]
            leftEye_yCoord = keypoint_coords[0][1][1]
            rightEye_xCoord = keypoint_coords[0][2][0]
            rightEye_yCoord = keypoint_coords[0][2][1]
            leftEar_xCoord = keypoint_coords[0][3][0]
            leftEar_yCoord = keypoint_coords[0][3][1]
            rightEar_xCoord = keypoint_coords[0][4][0]
            rightEar_yCoord = keypoint_coords[0][4][1]
            leftShoulder_xCoord = keypoint_coords[0][5][0]
            leftShoulder_yCoord = keypoint_coords[0][5][1]
            rightShoulder_xCoord = keypoint_coords[0][6][0]
            rightShoulder_yCoord = keypoint_coords[0][6][1]
            leftElbow_xCoord = keypoint_coords[0][7][0]
            leftElbow_yCoord = keypoint_coords[0][7][1]
            rightElbow_xCoord = keypoint_coords[0][8][0]
            rightElbow_yCoord = keypoint_coords[0][8][1]
            leftWrist_xCoord = keypoint_coords[0][9][0]
            leftWrist_yCoord = keypoint_coords[0][9][1]
            rightWrist_xCoord = keypoint_coords[0][10][0]
            rightWrist_yCoord = keypoint_coords[0][10][1]
            leftHip_xCoord = keypoint_coords[0][11][0]
            leftHip_yCoord = keypoint_coords[0][11][1]
            rightHip_xCoord = keypoint_coords[0][12][0]
            rightHip_yCoord = keypoint_coords[0][12][1]
            leftKnee_xCoord = keypoint_coords[0][13][0]
            leftKnee_yCoord = keypoint_coords[0][13][1]
            rightKnee_xCoord = keypoint_coords[0][14][0]
            rightKnee_yCoord = keypoint_coords[0][14][1]
            leftAnkle_xCoord = keypoint_coords[0][15][0]
            leftAnkle_yCoord = keypoint_coords[0][15][1]
            rightAnkle_xCoord = keypoint_coords[0][16][0]
            rightAnkle_yCoord = keypoint_coords[0][16][1]
            d ={'nose_xCoord': nose_xCoord, 'nose_yCoord': nose_yCoord, 'leftEye_xCoord': leftEye_xCoord, 'leftEye_yCoord': leftEye_yCoord, 'rightEye_xCoord': rightEye_xCoord, 'rightEye_yCoord': rightEye_yCoord, 'leftEar_xCoord': leftEar_xCoord, 'leftEar_yCoord': leftEar_yCoord, 'rightEar_xCoord': rightEar_xCoord, 'rightEar_yCoord': rightEar_yCoord, 'leftShoulder_xCoord': leftShoulder_xCoord, 'leftShoulder_yCoord': leftShoulder_yCoord, 'rightShoulder_xCoord': rightShoulder_xCoord, 'rightShoulder_yCoord': rightShoulder_yCoord, 'leftElbow_xCoord': leftElbow_xCoord, 'leftElbow_yCoord': leftElbow_yCoord, 'rightElbow_xCoord': rightElbow_xCoord, 'rightElbow_yCoord': rightElbow_yCoord, 'leftWrist_xCoord': leftWrist_xCoord, 'leftWrist_yCoord': leftWrist_yCoord, 'rightWrist_xCoord': rightWrist_xCoord, 'rightWrist_yCoord': rightWrist_yCoord, 'leftHip_xCoord': leftHip_xCoord, 'leftHip_yCoord': leftHip_yCoord, 'rightHip_xCoord': rightHip_xCoord, 'rightHip_yCoord': rightHip_yCoord, 'leftKnee_xCoord': leftKnee_xCoord, 'leftKnee_yCoord': leftKnee_yCoord, 'rightKnee_xCoord': rightKnee_xCoord, 'rightKnee_yCoord': rightKnee_yCoord, 'leftAnkle_xCoord': leftAnkle_xCoord, 'leftAnkle_yCoord': leftAnkle_yCoord, 'rightAnkle_xCoord': rightAnkle_xCoord, 'rightAnkle_yCoord': rightAnkle_yCoord}
            dummy_frame = dummy_frame.append(pd.DataFrame(data = d , index = [frame_count]))
            #dummy_frame = pd.DataFrame(data = d, columns = col, index = [frame_count])
            #print(dummy_frame.shape)
            if (((frame_count+1) % 30 == 0) and (frame_count+1 >= 120)):
                #sc=StandardScaler() 
                #X = np.zeros((120,34))
                X = (dummy_frame.values)[(frame_count-119):(frame_count+1)]
                X = (X-Mean)/Std
                #print(X)
                #print(X.shape)
                X = np.asarray(X).reshape(-1,120,34)
                #print(X)
                #print(X.shape)
                #X = tf.transpose(X, [1,0,2])
                #X = tf.reshape(X,[-1, 34])
                #print(X.shape)
                result= model.predict_classes(X) ## predicting from the model
                prob = model.predict_proba(X)
                if (result == 0):
                    result = 'Falling'
                elif (result == 1):
                    result = 'Pushups'
                elif (result == 2):
                    result = 'Sitting'
                elif (result == 3):
                    result = 'Walking'
                else:
                    result = 'Error'
                #st.write(prob)
            cv2.putText(overlay_image, 'Action: '+result, org, font, fontScale, color, thickness, cv2.LINE_AA, False) 
            frameST.image(overlay_image, channels="BGR")
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                frameST = st.empty()
                cap.release()
                cv2.destroyAllWindows()
                break
            if choice == 'upload video':
                if frame_count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):# total frames considered are (video_length_sec)*(frames/sec)
                    frameST = st.empty()
                    cap.release()
                    cv2.destroyAllWindows()
                    break

def main():
    """Face Detection App"""
    st.title("Fall Detection")
    st.text("Build with Streamlit,OpenCV and PosNet")
    activities = ["upload video", "Webcam"]
    choice = st.sidebar.selectbox("Select Activty",activities)
  
    if choice == 'upload video':
        #cv2.destroyAllWindows()
        st.subheader("Fall Detection on uploaded video")
        uploaded_file = st.file_uploader("Upload video",type=["mp4", "mpeg"])
        if uploaded_file is not None:
            g = io.BytesIO(uploaded_file.read())
            temporary_location = "test.mp4"
            with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file
            out.close()
            keypoints(choice)
    elif choice == 'Webcam':
        #cv2.destroyAllWindows()
        st.subheader("Fall Detection on Webcam")
        keypoints(choice)
            
if __name__ == '__main__':
        main()    