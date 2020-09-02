import pandas as pd
import numpy as np

JSON_FILE = './input/WLASL_v0.3.json'
OUTPUT_FILE = './output/WLASL.csv'


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
    data = data.loc[:, ['gloss', 'bbox', 'video_id', 'frame_end', 'frame_start']]
    data.to_csv(OUTPUT_FILE, index=False)


if __name__ == '__main__':
    jsonToCSV()