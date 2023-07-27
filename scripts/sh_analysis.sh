CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=2 \
python analysis/error_analysis.py \
--config ddim_cifar10.yml \
--model DDIM \
--model_path /cw/working-eowyn/mingxiao/TS-DPM/models/ddim_cifar10.ckpt \
--sampler pnm_solver \
--batch_size 32 \
--total_num_imgs 50000 \
--method f_pndm \
--sample_speed 10