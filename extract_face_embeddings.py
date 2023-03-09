import argparse

import os
import torch
import pandas as pd
import cv2
import numpy as np

from insightface.app import FaceAnalysis

import multiprocessing
from joblib import Parallel, delayed

def read_file(in_filenm):
    paths = []
    with open(in_filenm, 'r') as f:
        lines = f.readlines()
        for l in lines:
            paths.append(l.strip())

    return np.array(paths)

def main(input_dir, infiles, output_file_name):
   
    # gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, len(infiles))

    # init face embedding model
    app = FaceAnalysis('antelopev2', root='./face_model_checkpoints', providers=['CUDAExecutionProvider'])
    #app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
    
    csvs = []
    file_name = []
    npys = []
    for i, f in enumerate(infiles):
        img = cv2.imread(os.path.join(input_dir, f))
        faces = app.get(img,max_num=224)
        
        for face in faces:
            box = face.bbox.astype(np.int32)
            csvs.append([box[0], box[1], box[2], box[3], face.det_score])
            file_name.append(f)
            npys.append(face.normed_embedding[np.newaxis, :])
        if (i+1)%1000==0:
            print(i,f)
    
    if len(npys)>0:
        cur_csvs = pd.DataFrame(data=np.array(csvs), columns=['x', 'y', 'width', 'height', 'score'])
        cur_csvs['name'] = file_name
        cur_csvs.to_csv(f'{output_file_name}.csv')
        npys = np.vstack(npys)
        print(len(cur_csvs), npys.shape)
        np.save(f'{output_file_name}.npy', npys)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CAI Fingerprint Extraction from patches and saving the embedings on the drive')
    
    # all the args are read here
    parser.add_argument('-i', '--basefolder', default='/sensei-fs/users/shragarw/project_data/Stock_6M_1000px_data/', help='base path to the input folder')
    parser.add_argument('-f', '--inputfile', type=str, default='data_creation/files.txt', help='list of files in the basefolder to extract')
    parser.add_argument('-nj', '--njobs', type=int, default=1, help='number of multithreading')
    parser.add_argument('-o', '--outputdir', default='/sensei-fs/users/shragarw/project_data/Stock_6M_1000px_faces', help='folder to extract the patch embeddings')

    # patch size, overlap, input image basefolder, names of file for patches
    args = parser.parse_args()

    # get the names of all the image files
    #inputfile = os.path.join('data_creation/files', f'Stock_6M_1000px_data_{args.file_number}.txt')
    inputfile = args.inputfile #os.path.join('data_creation/files_clio', f'CLIO__{args.file_number}.txt')
    files = read_file(inputfile)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # these are the total number of cpus
    njobs = np.min([multiprocessing.cpu_count(), args.njobs])
    jobs_per_cpu = int(np.ceil(len(files)/njobs))
    print(len(files), jobs_per_cpu)

    # get all files to run
    full_struct = []
    for i in range(0, len(files), jobs_per_cpu):
        # name of the output file
        out_file_name = os.path.join(args.outputdir, os.path.basename(inputfile).split('.')[0] + f'_{i}')
        full_struct.append((args.basefolder, 
                            files[i:i+jobs_per_cpu],
                            out_file_name))
    
    print('total processes {}'.format(len(full_struct)))
    if len(full_struct)>1:
        Parallel(n_jobs=njobs, verbose=20)(delayed(main)(*full_struct[c]) for c in range(len(full_struct)))
    else:
        for c in range(len(full_struct)):
            main(*full_struct[c])
