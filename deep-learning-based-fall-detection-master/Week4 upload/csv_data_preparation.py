import numpy as np
import pandas as pd

def main():    
    csv_reader = pd.read_csv(r'C:\Users\insapab\Desktop\Python\Projects\Deep Learning based fall-detection\Week4\dataset_new.csv')      
    nose_xCoord = []
    nose_yCoord = []
    leftEye_xCoord = []
    leftEye_yCoord = []
    rightEye_xCoord = []
    rightEye_yCoord = []
    leftEar_xCoord = []
    leftEar_yCoord = []
    rightEar_xCoord = []
    rightEar_yCoord = []
    leftShoulder_xCoord = []
    leftShoulder_yCoord = []
    rightShoulder_xCoord = []
    rightShoulder_yCoord = []
    leftElbow_xCoord = []
    leftElbow_yCoord = []
    rightElbow_xCoord = []
    rightElbow_yCoord = []
    leftWrist_xCoord = []
    leftWrist_yCoord = []
    rightWrist_xCoord = []
    rightWrist_yCoord = []
    leftHip_xCoord = []
    leftHip_yCoord = []
    rightHip_xCoord = []
    rightHip_yCoord = []
    leftKnee_xCoord = []
    leftKnee_yCoord = []
    rightKnee_xCoord = []
    rightKnee_yCoord = []
    leftAnkle_xCoord = []
    leftAnkle_yCoord = []
    rightAnkle_xCoord = []
    rightAnkle_yCoord = []
    action_name_Coord = []
    for row in range(0,len(csv_reader),17):
        data = csv_reader[(0 + row):(17 + row)]
        nose_xCoord = np.append(nose_xCoord,[data.iloc[0][3]])
        nose_yCoord = np.append(nose_yCoord,[data.iloc[0][4]])
        leftEye_xCoord = np.append(leftEye_xCoord,[data.iloc[1][3]])
        leftEye_yCoord = np.append(leftEye_yCoord,[data.iloc[1][4]])
        rightEye_xCoord = np.append(rightEye_xCoord,[data.iloc[2][3]])
        rightEye_yCoord = np.append(rightEye_yCoord,[data.iloc[2][4]])
        leftEar_xCoord = np.append(leftEar_xCoord,[data.iloc[3][3]])
        leftEar_yCoord = np.append(leftEar_yCoord,[data.iloc[3][4]])
        rightEar_xCoord = np.append(rightEar_xCoord,[data.iloc[4][3]])
        rightEar_yCoord = np.append(rightEar_yCoord,[data.iloc[4][4]])
        leftShoulder_xCoord = np.append(leftShoulder_xCoord,[data.iloc[5][3]])
        leftShoulder_yCoord = np.append(leftShoulder_yCoord,[data.iloc[5][4]])
        rightShoulder_xCoord = np.append(rightShoulder_xCoord,[data.iloc[6][3]])
        rightShoulder_yCoord = np.append(rightShoulder_yCoord,[data.iloc[6][4]])
        leftElbow_xCoord = np.append(leftElbow_xCoord,[data.iloc[7][3]])
        leftElbow_yCoord = np.append(leftElbow_yCoord,[data.iloc[7][4]])
        rightElbow_xCoord = np.append(rightElbow_xCoord,[data.iloc[8][3]])
        rightElbow_yCoord = np.append(rightElbow_yCoord,[data.iloc[8][4]])
        leftWrist_xCoord = np.append(leftWrist_xCoord,[data.iloc[9][3]])
        leftWrist_yCoord = np.append(leftWrist_yCoord,[data.iloc[9][4]])
        rightWrist_xCoord = np.append(rightWrist_xCoord,[data.iloc[10][3]])
        rightWrist_yCoord = np.append(rightWrist_yCoord,[data.iloc[10][4]])
        leftHip_xCoord = np.append(leftHip_xCoord,[data.iloc[11][3]])
        leftHip_yCoord = np.append(leftHip_yCoord,[data.iloc[11][4]])
        rightHip_xCoord = np.append(rightHip_xCoord,[data.iloc[12][3]])
        rightHip_yCoord = np.append(rightHip_yCoord,[data.iloc[12][4]])
        leftKnee_xCoord = np.append(leftKnee_xCoord,[data.iloc[13][3]])
        leftKnee_yCoord = np.append(leftKnee_yCoord,[data.iloc[13][4]])
        rightKnee_xCoord = np.append(rightKnee_xCoord,[data.iloc[14][3]])
        rightKnee_yCoord = np.append(rightKnee_yCoord,[data.iloc[14][4]])
        leftAnkle_xCoord = np.append(leftAnkle_xCoord,[data.iloc[15][3]])
        leftAnkle_yCoord = np.append(leftAnkle_yCoord,[data.iloc[15][4]])
        rightAnkle_xCoord = np.append(rightAnkle_xCoord,[data.iloc[16][3]])
        rightAnkle_yCoord = np.append(rightAnkle_yCoord,[data.iloc[16][4]])
        action_name_Coord = np.append(action_name_Coord,[data.iloc[0][0]])
    
    d ={'nose_xCoord': nose_xCoord, 'nose_yCoord': nose_yCoord, 'leftEye_xCoord': leftEye_xCoord, 'leftEye_yCoord': leftEye_yCoord, 'rightEye_xCoord': rightEye_xCoord, 'rightEye_yCoord': rightEye_yCoord, 'leftEar_xCoord': leftEar_xCoord, 'leftEar_yCoord': leftEar_yCoord, 'rightEar_xCoord': rightEar_xCoord, 'rightEar_yCoord': rightEar_yCoord, 'leftShoulder_xCoord': leftShoulder_xCoord, 'leftShoulder_yCoord': leftShoulder_yCoord, 'rightShoulder_xCoord': rightShoulder_xCoord, 'rightShoulder_yCoord': rightShoulder_yCoord, 'leftElbow_xCoord': leftElbow_xCoord, 'leftElbow_yCoord': leftElbow_yCoord, 'rightElbow_xCoord': rightElbow_xCoord, 'rightElbow_yCoord': rightElbow_yCoord, 'leftWrist_xCoord': leftWrist_xCoord, 'leftWrist_yCoord': leftWrist_yCoord, 'rightWrist_xCoord': rightWrist_xCoord, 'rightWrist_yCoord': rightWrist_yCoord, 'leftHip_xCoord': leftHip_xCoord, 'leftHip_yCoord': leftHip_yCoord, 'rightHip_xCoord': rightHip_xCoord, 'rightHip_yCoord': rightHip_yCoord, 'leftKnee_xCoord': leftKnee_xCoord, 'leftKnee_yCoord': leftKnee_yCoord, 'rightKnee_xCoord': rightKnee_xCoord, 'rightKnee_yCoord': rightKnee_yCoord, 'leftAnkle_xCoord': leftAnkle_xCoord, 'leftAnkle_yCoord': leftAnkle_yCoord, 'rightAnkle_xCoord': rightAnkle_xCoord, 'rightAnkle_yCoord': rightAnkle_yCoord,'action': action_name_Coord}
    dummy_frame =  pd.DataFrame(data=d)
    dummy_frame.to_csv('dataset1.csv')
    print('Done')

if __name__ == "__main__":
    main()
