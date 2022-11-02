import cv2
import torch
import numpy as np
import os
import glob
from torchvision import transforms
import torch.nn.functional as F
from insightface_func.face_detect_crop_multi import Face_detect_crop

transformer_Arcface = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

device = torch.device("cuda:0")
netArc_checkpoint = torch.load('arcface_checkpoint.tar')
netArc = netArc_checkpoint['model'].module
netArc = netArc.to(device)
netArc.eval()

app = Face_detect_crop(name='antelope', root='./insightface_func/models')
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