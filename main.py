import preprocess

if __name__ == '__main__':
    # Load Data
    data = preprocess.readCSV()

    # Loop through each video
    for video in data.head().iterrows():

        # Convert video into frames
        image_list = preprocess.mp4ToFrames(video[1])
        break

    print()