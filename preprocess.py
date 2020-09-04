import pandas as pd
import numpy as np
import cv2
import os

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
    data = data.drop('variable', 1)
    data = data.dropna()

    data = data.join(pd.DataFrame.from_records(data['instance']))
    data = data.dropna()

    data = pd.DataFrame(data.loc[:, ['video_id', 'gloss', 'bbox', 'frame_start', 'frame_end']])
    data = data.sort_values(['gloss', 'video_id'])
    data[['frame_start', 'frame_end']] = data[['frame_start', 'frame_end']].astype(int)
    data.to_csv(OUTPUT_FILE, index=False)


def readCSV():
    return pd.read_csv(OUTPUT_FILE, dtype={'video_id': str})


# Convert Video into Frames
def mp4ToFrames(video):
    img_list = []
    video_dir = f'./input/videos/{video.video_id}.mp4'

    cap = cv2.VideoCapture(video_dir)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(int(frame_count)):
        _, image = cap.read()

        if video.frame_end != -1:
            if i not in range(video.frame_start, video.frame_end):
                continue

        cv2.imshow("Images", image)
        cv2.waitKey()

        img_list.append(image)

    cap.release()
    cv2.destroyAllWindows()
    return img_list


if __name__ == '__main__':
    # jsonToCSV()
    data = readCSV()

    for video in data.head().iterrows():
        mp4ToFrames(video[1])
        break