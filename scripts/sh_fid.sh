CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=0 \
python evaluate/fid_score.py \
--batch_size 50 \
--num_workers 4 \
--device "cuda" \
--path ./ddim_imgs/50_steps_scale_step_0.0025 /cw/working-eowyn/mingxiao/TS-DPM/fids/fid_cifar10_train.npz