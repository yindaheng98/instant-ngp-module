python3 scripts/train_seq.py \
    --init_steps 10000 --steps 1000 --start 1 --end 101 \
    --trainargs=--decay=0.5 \
    --initformat "data/nerf/taekwondo/frame%d" \
    --dataformat "data/nerf/taekwondo/residuals/frame%d" \
    --saveformat "results/stnerf-taekwondo-partial/frame%d.bson" \
    --executable "./instant-ngp-train"
python3 scripts/train_seq.py \
    --init_steps 10000 --steps 1000 --start 1 --end 101 \
    --trainargs=--decay=0.5 \
    --initformat "data/nerf/taekwondo/frame%d" \
    --dataformat "data/nerf/taekwondo/frame%d" \
    --saveformat "results/stnerf-taekwondo-1000steps/frame%d.bson" \
    --executable "./instant-ngp-train"

python3 scripts/train_seq.py \
    --init_steps 10000 --steps 1000 --start 1 --end 75 \
    --trainargs=--decay=0.5 \
    --initformat "data/nerf/walking/frame%d" \
    --dataformat "data/nerf/walking/residuals/frame%d" \
    --saveformat "results/stnerf-walking-1000steps/frame%d.bson" \
    --executable "./instant-ngp-train"
python3 scripts/train_seq.py \
    --init_steps 10000 --steps 1000 --start 1 --end 75 \
    --trainargs=--decay=0.5 \
    --initformat "data/nerf/walking/frame%d" \
    --dataformat "data/nerf/walking/frame%d" \
    --saveformat "results/stnerf-walking-partial/frame%d.bson" \
    --executable "./instant-ngp-train"

python3 scripts/train_seq.py \
    --init_steps 30000 --steps 10000 --start 1 --end 300 --decay 0.5 \
    --dataformat "data/nerf/coffee_martini/frame%d" \
    --saveformat "results/coffee_martini/frame%d.bson" \
    --executable "./instant-ngp-train"
python3 scripts/train_seq.py \
    --init_steps 30000 --steps 10000 --start 1 --end 300 --decay 0.5 \
    --dataformat "data/nerf/flame_steak/frame%d" \
    --saveformat "results/flame_steak/frame%d.bson" \
    --executable "./instant-ngp-train"
python3 scripts/train_seq.py \
    --init_steps 30000 --steps 10000 --start 1 --end 300 --decay 0.5 \
    --dataformat "data/nerf/sear_steak/frame%d" \
    --saveformat "results/sear_steak/frame%d.bson" \
    --executable "./instant-ngp-train"
