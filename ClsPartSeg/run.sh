## classification
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_ssg_cls
python test_classification.py --log_dir pointnet2_ssg_cls

python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_msg_cls --batch_size 18
python test_classification.py --log_dir pointnet2_msg_cls

python train_classification.py --model dgcnn --log_dir dgcnn_cls
python train_classification.py --model dgcnn --log_dir dgcnn_cls_small
python train_classification.py --model dgcnn --log_dir dgcnn_cls_sgd --optimizer SGD --learning_rate 0.0001
python test_classification.py --log_dir dgcnn_cls


# Point_Transformer2 with downsampling, must to use sgd optimizer, adam optimizer not working
python train_classification.py --model pct_cls --log_dir pct_cls
python train_classification.py --model pct_cls --log_dir pct_cls_sgd --optimizer SGD --learning_rate 0.0001 --decay_rate 0.0005
# Point_Transformer2 without downsampling, adam optimizer, not working
python test_classification.py --log_dir pct_cls

python train_classification.py --model pointmlp --log_dir pointMLP_cls --optimizer SGD
python test_classification.py --log_dir pointMLP_cls

# test on scanobjectnn
# adam optimizer, epochs=50, step_size=5
python train_classification_sim2rel.py --model pointmlp --log_dir pointMLP_cls_v2
python train_classification_sim2rel.py --model pointmlp --log_dir pointMLP_cls_v2_sgd --optimizer SGD --learning_rate 0.0001 --decay_rate 0.0005
python train_classification_sim2rel.py --model pointmlp --log_dir pointMLP_cls --dataset ModelNet40
python test_classification_sim2rel.py --log_dir pointMLP_cls

### part seg
python train_partseg.py --model pointnet2_part_seg_ssg --log_dir pointnet2_ssg_seg_n1024

python train_partseg.py --model pct_seg --log_dir pct_seg_sgd_b48 --batch_size 48 --optimizer SGD --learning_rate 0.0001 --decay_rate 0.0005
python test_partseg.py --log_dir pct_seg_n1024

s