import pandas as pd
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paths import get_raw_memory_data_file_path, get_preprocessed_memory_data_file_path

def preprocess_and_save_memory_data():

    # ---- Load raw data
    df = pd.read_csv(get_raw_memory_data_file_path())

    # --- Preprocess data

    # only keep V3 is [ S3_Memory ] Stimuli Placed or [ S3_Memory ] Memory Placement confidence
    df = df[df['V3'].isin(['[ S3_Memory ] Stimuli Placed', '[ S3_Memory ] Memory Placement confidence', '[ S3_Memory ] Memory Quiz2 Response', '[ S3_Memory ] Memory Quiz2 confidence'])]
    # Give adjacent indices to each stimulus placement and confidence rating
    placed = df[df['V3'] == '[ S3_Memory ] Stimuli Placed']
    placed_confidence = df[df['V3'] == '[ S3_Memory ] Memory Placement confidence']
    timed = df[df['V3'] == '[ S3_Memory ] Memory Quiz2 Response']
    timed_confidence = df[df['V3'] == '[ S3_Memory ] Memory Quiz2 confidence']

    # reset index
    placed = placed.reset_index(drop=True)
    placed_confidence = placed_confidence.reset_index(drop=True)
    timed = timed.reset_index(drop=True)
    timed_confidence = timed_confidence.reset_index(drop=True)

    timed = timed.rename(columns={'V4': 'Timed'})[['Timed']]
    placed_confidence = placed_confidence.rename(columns={'V4': 'Placed Confidence'})[['Placed Confidence']]
    timed_confidence = timed_confidence.rename(columns={'V4': 'Timed Confidence'})[['Timed Confidence']]

    # merge on index
    merged = pd.merge(placed, placed_confidence, left_index=True, right_index=True)
    merged = pd.merge(merged, timed, left_index=True, right_index=True)
    merged = pd.merge(merged, timed_confidence, left_index=True, right_index=True)

    # Only keep relevant columns
    df = merged[['Participant','V1', 'V4', 'V8', 'V9', 'V10', 'Placed Confidence', 'Timed', 'Timed Confidence']]

    # Extract X, Y, Z from V9
    df.loc[:, ['X', 'Y', 'Z']] = df['V9'].str.extract(
        r'Stimuli Pos\s*:\s*\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)'
    ).astype(float).to_numpy()

    # Add true time column 1 if e3 in V4, 7 if e2 in V4, 14 if e1 in V4
    df = df.copy()
    df.loc[:, 'True Time'] = df['V4'].map(lambda x: 0 if 'e3' in x else (1 if 'e2' in x else (2 if 'e1' in x else 100)))

    # Only keep the first two elements of V4 split by '_'
    df.loc[:, 'V4'] = df['V4'].astype(str).apply(lambda x: '_'.join(x.split('_')[:2]))

    # change Timed column to numeric values
    df.loc[:, 'Timed'] = df['Timed'].map({'yesterday': 0, 'one_week': 1, 'two_weeks': 2, 'never': 100})
    df = df.copy()

    # remove 'Distance : ' from V10 and convert to float
    df['Distance'] = df['V10'].str.replace('Distance :', '').astype(float)
    df.drop('V10', axis=1, inplace=True)

    df['Classified as old'] = df['Timed'].apply(lambda x: 1 if x != 100 else 0)
    df['Old'] = df['True Time'].apply(lambda x: 1 if x != 100 else 0)

    df.to_csv(get_preprocessed_memory_data_file_path(), index=False, sep=',')

def load_memory_data():
    if not os.path.exists(get_preprocessed_memory_data_file_path()):
        preprocess_and_save_memory_data()
    df = pd.read_csv(get_preprocessed_memory_data_file_path())
    return df