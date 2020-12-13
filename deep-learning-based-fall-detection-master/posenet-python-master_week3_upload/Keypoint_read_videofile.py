import tensorflow as tf
import cv2
import time
import argparse
import posenet
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default='test_video.mp4', help="Optionally use a video file instead of a live camera")# read video file
args = parser.parse_args()

def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0   
   
        col_names = ['action', 'frame_number', 'input_number', 'x_inputs', 'y_inputs']
        csv_reader = pd.read_csv(r'C:\Users\insapab\Desktop\Python\Projects\Deep Learning based fall-detection\Week3\posenet-python-master\dataset1.csv',names=col_names, header=None)# reading old data to a dataframe       
        
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

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
            
            ## vectorized code to extract key points data from first pose and appends to csv_reader dataframe ##
            input_number_array= (np.arange(len(keypoint_scores[0])).reshape(1,len(keypoint_scores[0])) + 1)[0] # an array [1,2 ------ no.of keypoints]
            x_input_array = keypoint_coords[0][:,0]# an array having x-cordinates of key points in a frame
            y_input_array = keypoint_coords[0][:,1]# an array having y-cordinates of key points in a frame
            d ={'action': 'Pushups', 'frame_number': frame_count+1, 'input_number': input_number_array, 'x_inputs': x_input_array, 'y_inputs': y_input_array}
            dummy_frame =  pd.DataFrame(data=d)
            csv_reader = csv_reader.append(dummy_frame)#appending new frame data to data frame
            
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_count == 25:# total frames considered are (video_length_sec)*(frames/sec)
                break
        csv_reader.to_csv('dataset1.csv')# writing data in the same csv file
        print('Average FPS: ', frame_count / (time.time() - start))

if __name__ == "__main__":
    main()