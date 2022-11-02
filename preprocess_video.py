import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
import random
import os
import glob
from torchvision import transforms
import torch.nn.functional as F
from insightface_func.face_detect_crop_multi import Face_detect_crop
import subprocess, platform, shutil

parser = argparse.ArgumentParser(description='face recognition test')
# general
parser.add_argument('--input_video', type = str, default='ZWRFD9uOQnY.mp4', help='input video')
parser.add_argument('--output_video', type = str, default = 'ZWRFD9uOQnY_out.mp4', help = 'output video')
parser.add_argument('--person_name', default='craig_kelly', type=str, help='name of the person')
parser.add_argument('--threshold', default=0.3, type=float, help='same identity threshold')
args = parser.parse_args()

result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        	"format=nb_streams", "-of",
                            "default=noprint_wrappers=1:nokey=1", args.input_video],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT)
has_audio = (int(result.stdout)-1) > 0
if has_audio:
    command = 'ffmpeg -y -hide_banner -loglevel error -i {} temp.wav'.format(args.input_video)
    subprocess.call(command, shell=True)

if os.path.exists('input/'):
	shutil.rmtree('input/')
os.mkdir('input/')
if os.path.exists('output/'):
	shutil.rmtree('output/')
os.mkdir('output/')
command = 'ffmpeg -y -hide_banner -loglevel error -i {} -q:v 1 -qmin 1 -qmax 1 input/%08d.jpg'.format(args.input_video)
subprocess.call(command, shell=True)

transformer_Arcface = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

device = torch.device("cuda:0")
netArc_checkpoint = torch.load('checkpoints/arcface_checkpoint.tar')
netArc = netArc_checkpoint['model'].module
netArc = netArc.to(device)
netArc.eval()

app = Face_detect_crop(name='antelope', root='./checkpoints')
app.prepare(ctx_id= 0, det_thresh=0.2, det_size=(640,640))

ref_feat = []
all_frames = glob.glob(os.path.join('facebank/'+args.person_name, '*.jpg'))
all_frames.sort()
for path_img in all_frames:
	img = cv2.imread(path_img)
	img, _ = app.get(img,224)
	img = img[0]
	img = transformer_Arcface(img)
	img = img.view(-1, img.shape[0], img.shape[1], img.shape[2])
	img = img.cuda()
	img = F.interpolate(img, scale_factor=0.5)
	with torch.no_grad():
		feat = netArc(img)
	feat = F.normalize(feat, p=2, dim=1).cpu().detach().numpy()
	ref_feat.append(feat[0])

list_pathfile_img = glob.glob(os.path.join('input/', '*.jpg'))
list_pathfile_img.sort()

for i in tqdm(range(len(list_pathfile_img))):
	#print('{}/{}'.format(i+1, len(list_pathfile_img)))
	#print(list_pathfile_img[i])
	img_whole = cv2.imread(list_pathfile_img[i])
	black = img_whole.copy()
	black[:] = 0
	cv2.imwrite(os.path.join('output', os.path.basename(list_pathfile_img[i])), black)
	img_crops, img_mat, img_bbox = app.get_with_bbox(img_whole,224)
	if img_crops == None:
		continue
	if len(img_crops) == 1:
		img_a = transformer_Arcface(img_crops[0])
		img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
		img_id = img_id.cuda()
		img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
		latend_id = netArc(img_id_downsample)
		latend_id = F.normalize(latend_id, p=2, dim=1).cpu().detach().numpy()
		for ref_f in ref_feat:
			if np.dot(latend_id[0], ref_f) > args.threshold:
				cv2.imwrite(os.path.join('output', os.path.basename(list_pathfile_img[i])), img_whole)
				break
	else:	
		for j, img in enumerate(img_crops):
			img_a = transformer_Arcface(img)
			img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
			img_id = img_id.cuda()
			img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
			latend_id = netArc(img_id_downsample)
			latend_id = F.normalize(latend_id, p=2, dim=1).cpu().detach().numpy()
			for ref_f in ref_feat:
				if np.dot(latend_id[0], ref_f) > args.threshold:
					bbox = img_bbox[j, 0:4]
					x = int(bbox[0])
					w = int(bbox[2]) - x
					y = int(bbox[1])
					h = int(bbox[3]) - y
					x = x - int(w/2)
					if x < 0:
						x = 0
					w = w*2
					if x+w > black.shape[1]-1:
						w = black.shape[1]-1 - x
					y = y - int(h/2)
					if y < 0:
						y = 0
					h = h*2
					if y+h > black.shape[0]-1:
						h = black.shape[0]-1 - y
					nimg = black.copy()
					nimg[y:y+h, x:x+w] = img_whole[y:y+h, x:x+w]
					cv2.imwrite(os.path.join('output', os.path.basename(list_pathfile_img[i])), nimg)
					break

video_stream = cv2.VideoCapture(args.input_video)
fps = video_stream.get(cv2.CAP_PROP_FPS)
video_stream.release()

if has_audio:
    command = 'ffmpeg -y -hide_banner -loglevel error -framerate {} -i output/%08d.jpg -i temp.wav -c:v libx264 -preset veryslow -crf 10 -pix_fmt yuv420p {}'.format(fps, args.output_video)
else:
    command = 'ffmpeg -y -hide_banner -loglevel error -framerate {} -i output/%08d.jpg -c:v libx264 -preset veryslow -crf 10 -pix_fmt yuv420p {}'.format(fps, args.output_video)
subprocess.call(command, shell=platform.system() != 'Windows')

if os.path.exists('input/'):
	shutil.rmtree('input/')
if os.path.exists('output/'):
	shutil.rmtree('output/')
if has_audio:
    os.remove('temp.wav')