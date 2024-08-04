python datafree_kd_imagenet.py \
--method fast_meta \
--adv 0.01 \
--bn 0.1 \
--oh 0.1 \
--save_dir run/tinyimagenet_resnet34 \
--data_root data \
--teacher resnet34 \
--student resnet18 \
--dataset tinyimagenet \
--lr 0.01 \
--lr_g 0.0002 \
--lr_z 0.01 \
--lr_g_meta 0.0001 \
--lr_z_meta 0.005 \
--T 5.0 \
--epochs 100 \
--g_steps 50 \
--kd_steps 20 \
--ep_steps 200 \
--reinit 50 \
--apply_kd 0 \
--bn_mmt 0.9 \
--lr_diff 1 \
--is_maml 0 \
--bn_mmt 0.9 \
--batch_size 128 \
--synthesis_batch_size 16 \
--gpu 0