CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES=0 \
python run_generate.py \
--config ddim_cifar10.yml \
--model DDIM \
--model_path /cw/working-eowyn/mingxiao/TS-DPM/models/ddim_cifar10.ckpt \
--sampler pnm_solver \
--batch_size 32 \
--total_num_imgs 50000 \
--method euler \
--sample_speed 50