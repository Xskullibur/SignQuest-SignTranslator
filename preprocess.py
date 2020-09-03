import pandas as pd
import numpy as np
import cv2

JSON_FILE = './input/WLASL_v0.3.json'
OUTPUT_FILE = './output/WLASL.csv'


# Convert JSON file into CSV
def jsonToCSV():
    # Read JSON File
    data = pd.read_json(JSON_FILE, 'values')

    # Convert instance (object) into DataFrame
    instances = pd.DataFrame.from_records(data['instances'])
    data = data.join(instances)

    # Drop Old Instances
    data = data.drop('instances', 1)

    # Reformat DataFrame
    data = data.melt(id_vars=['gloss'], value_name='instance')

    # Drop Missing values
    data = data.dropna()
    data = data.drop('variable', 1)

    data = data.join(pd.DataFrame.from_records(data['instance']))
    data = data.loc[:, ['gloss', 'bbox', 'video_id', 'fps', 'frame_end', 'frame_start']]
    data.to_csv(OUTPUT_FILE, index=False)


# Convert Video into Frames
def mp4ToFrames(video_id, fps, frame_start, frame_end):

    images = []

    video_dir = f'./input/videos/{video_id}.mp4'
    cap = cv2.VideoCapture(video_dir)
    cap.set(1, fps)

    for i in range(frame_start, frame_end):
        retval, image = cap.read()
        images.append(image)


if __name__ == '__main__':
    # jsonToCSV()
    mp4ToFrames('69241', 25, 5122, 5181)