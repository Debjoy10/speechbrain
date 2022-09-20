import numpy as np
import os
import json
import argparse
from datetime import datetime
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument('--scores_file', help='Scores File', default='/media/absp/4E897AE46CE3ABFA/IMSV/results/imsv_test/speaker_verification_ecapa_big_trained/scores.txt')
parser.add_argument('--clip', help='Whether to clip or normalise negative scores', action='store_true')
args = parser.parse_args()

def main():
    # Read files 
    with open(args.scores_file) as f:
        scores = f.readlines()
    with open('/media/absp/4E897AE46CE3ABFA/IMSV/I-MSV-DATA/file_ID.json', 'r') as fp:
        fileid = json.load(fp)

    # Get scores for pairs
    classification = {}
    for score in scores:
        k = score.split(' ')[1] + '.wav'
        if k in classification:
            classification[k][score.split(' ')[0].split('_')[0]] = float(score.split(' ')[3].strip())
        else:
            classification[k] = {score.split(' ')[0].split('_')[0]: float(score.split(' ')[3].strip())}
    
    # Combine scores to csv
    lines = ['utterance_id,speaker_id,score\n']
    for k, v in classification.items():
        ids = fileid[k]
        for id_ in ids:
            if not args.clip:
                lines.append('{},{},{}\n'.format(k, id_, (v[id_]+1)/2))
            else:
                lines.append('{},{},{}\n'.format(k, id_, v[id_] if v[id_] > 0 else 0.0001))
    
    with open('results.csv', 'w') as f:
        f.writelines(lines)

    # Zip to folder
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    zfile = "submissions/submission_{}.zip".format(date_time)
    ZipFile(zfile, mode='w').write('results.csv')
    os.remove('results.csv')

if __name__ == '__main__':
    main()