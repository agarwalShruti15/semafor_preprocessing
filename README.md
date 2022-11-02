# fr_semafor

pip install -r requirement.txt

download checkpoints https://drive.google.com/drive/folders/17rQq-P8O0r3-SVeuc8q-fPZ5FxkylKj3?usp=sharing

# identity person
python inference.py --input_video P_0840NAzLQ.mp4

# extract person
python preprocess_video.py --input_video P_0840NAzLQ.mp4 --output_video P_0840NAzLQ_out.mp4 --person_name rahul_gandhi

# identify speaker
this step will mask the subject when he is not speaking

./download_model.sh

python identify_speaker.py --input_video test_out.mp4 --output_video test_final.mp4