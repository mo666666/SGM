# PGD attack 生成与测试
CUDA_VISIBLE_DEVICES=6 python attack_sgm.py --gamma 1.0 --output_dir adv_images_mixer_b16_224_gamma_10 --arch mixer_b16_224 --batch-size 50
CUDA_VISIBLE_DEVICES=6 python attack_sgm.py --gamma 0.6 --output_dir adv_images_mixer_b16_224_gamma_06 --arch mixer_b16_224 --batch-size 50
CUDA_VISIBLE_DEVICES=6 python evaluate_all.py --default --input_dir adv_images_mixer_b16_224_gamma_10
CUDA_VISIBLE_DEVICES=6 python evaluate_all.py --default --input_dir adv_images_mixer_b16_224_gamma_06

