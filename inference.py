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

parser = argparse.ArgumentParser(description='face recognition test')
# general
parser.add_argument('--input_video', default='P_0840NAzLQ.mp4', help='input video')
parser.add_argument('--img_num', default=100, type=int, help='number of the images in the video used in face recognition')
parser.add_argument('--threshold', default=0.5, type=float, help='same identity threshold')
args = parser.parse_args()

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

embeddings = []
names = []
for path in os.listdir('facebank'):
	all_frames = glob.glob(os.path.join('facebank/'+path, '*.jpg'))
	all_frames.sort()
	embs = []
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
		embs.append(feat[0])
	embeddings.append(embs)
	names.append(os.path.basename(path))

cap = cv2.VideoCapture(args.input_video)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if length < args.img_num:
	selected_idx = range(length)
else:
	selected_idx = random.sample(range(length), args.img_num)
selected_idx.sort()

full_frames = []
count = 0
while cap.isOpened():
	cap.set(1, selected_idx[count])
	ret, frame = cap.read()
	if ret:
		full_frames.append(frame)
		count += 1
		if count == len(selected_idx):
			cap.release()
			break 
		cap.set(1, selected_idx[count])	
	else:
		cap.release()
		break

person_detected = np.zeros(len(names))
for img_whole in tqdm(full_frames):
	img_crops, _ = app.get(img_whole,224)
	if img_crops == None:
		continue
	for img in img_crops:
		img_a = transformer_Arcface(img)
		img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
		img_id = img_id.cuda()
		img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
		latend_id = netArc(img_id_downsample)
		latend_id = F.normalize(latend_id, p=2, dim=1).cpu().detach().numpy()
		for i, feats in enumerate(embeddings):
			for ref_feat in feats:
				if np.dot(latend_id[0], ref_feat) > args.threshold:
					person_detected[i] = person_detected[i]+1
					break

print(person_detected)
print(names[np.argmax(person_detected)])