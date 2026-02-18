# PGD attack 生成与测试
pip install timm==0.4.5
CUDA_VISIBLE_DEVICES=3 python attack_sgm.py --gamma 1.0 --output_dir adv_images_vit_base_patch16_224_gamma_10 --arch vit_base_patch16_224 --batch-size 50
CUDA_VISIBLE_DEVICES=3 python attack_sgm.py --gamma 0.6 --output_dir adv_images_vit_base_patch16_224_gamma_06 --arch vit_base_patch16_224 --batch-size 50
pip install timm==0.6.13
CUDA_VISIBLE_DEVICES=3 python evaluate_all.py --default --input_dir adv_images_vit_base_patch16_224_gamma_10
CUDA_VISIBLE_DEVICES=3 python evaluate_all.py --default --input_dir adv_images_vit_base_patch16_224_gamma_06