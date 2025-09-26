python ./tsne_medmamba.py \
  --data-dir ../dataset/val \
  --weights ./models/ClassifierNet.pth \
  --num-classes 6 \
  --feat-source fs \
  --out tsne_fs.png
