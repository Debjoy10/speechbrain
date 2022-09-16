import numpy as np
from string import ascii_lowercase
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', help='Main Yaml File', default='/home/absp/Debjoy/speechbrain/recipes/IMSV/SpeakerRec/hparams/verification_ecapa_trained_test.yaml')
parser.add_argument('--bash', help='Location of Bash File', default='/home/absp/Debjoy/speechbrain/recipes/IMSV/SpeakerRec/bash/full_test.sh')
args = parser.parse_args()

def main():
    bash_cmds = []
    path = args.yaml
    
    # Location of verification_file and output dir location
    with open(path) as f:
        yaml = f.readlines()
    x = ['verification_file' in y for y in yaml]
    assert sum(x) == 1
    idx = np.argmax(x)
    z = ['output_folder' in y for y in yaml]
    assert sum(x) == 1
    output = os.path.join(yaml[np.argmax(z)].strip('output_folder:').strip(), 'save')

    # Write 12 yaml files from one
    for c in ascii_lowercase:
        if c == 'm':
            break
        new_path = '/home/absp/Debjoy/speechbrain/recipes/IMSV/SpeakerRec/hparams/test_divided_hparams/verification_ecapa_trained_test_a{}.yaml'.format(c)
        with open(new_path, 'w') as f:
            newyaml = np.copy(yaml)
            newyaml[idx] = 'verification_file: /media/absp/4E897AE46CE3ABFA/IMSV/I-MSV-DATA/test_divided/IMSV_public_test_a{}'.format(c)
            f.writelines(newyaml)
        bash_cmds.append('python speaker_verification_cosine.py {}\n'.format(new_path))
        bash_cmds.append('rm -rf {}\n'.format(output))

    with open(args.bash, 'w') as f:
        f.writelines(bash_cmds)

main()