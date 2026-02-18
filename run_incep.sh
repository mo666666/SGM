# PGD attack 生成与测试
CUDA_VISIBLE_DEVICES=2 python attack_sgm.py --gamma 1.0 --output_dir adv_images_inceptionv3_gamma_10 --arch inceptionv3 --batch-size 50
CUDA_VISIBLE_DEVICES=2 python attack_sgm.py --gamma 0.6 --output_dir adv_images_inceptionv3_gamma_06 --arch inceptionv3 --batch-size 50
CUDA_VISIBLE_DEVICES=2 python evaluate_all.py --default --input_dir adv_images_inceptionv3_gamma_10
CUDA_VISIBLE_DEVICES=2 python evaluate_all.py --default --input_dir adv_images_inceptionv3_gamma_06