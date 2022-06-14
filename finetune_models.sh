# Training parameters for replication
model_name="vgg16"
lr="1e-3"
gamma=0.95
batch_size=64
epochs=50

# Desired paths and num workers
images_dir="/media/lassi/Data/datasets/coco/images/"
ann_dir="/media/lassi/Data/datasets/coco/annotations/"
checkpoint_dir="/media/lassi/Data/checkpoints/"
num_workers=16

# Parameters for wandb.init (experiment tracking)
wandb_entity="lassiraa"
wandb_project="coco-finetune"


python fine_tune.py --images_dir ${images_dir} \
    --ann_dir ${ann_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --model_name ${model_name} \
    --lr ${lr} \
    --gamma ${gamma} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --wandb_entity ${wandb_entity} \
    --wandb_project ${wandb_project}


# Next model to train
model_name="vit_b_32"
lr="2e-3"
gamma=0.95

python fine_tune.py --images_dir ${images_dir} \
    --ann_dir ${ann_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --model_name ${model_name} \
    --lr ${lr} \
    --gamma ${gamma} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --wandb_entity ${wandb_entity} \
    --wandb_project ${wandb_project}


# Next model to train
model_name="resnet50"
lr="1e-3"
gamma=0.95

python fine_tune.py --images_dir ${images_dir} \
    --ann_dir ${ann_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --model_name ${model_name} \
    --lr ${lr} \
    --gamma ${gamma} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --num_workers ${num_workers} \
    --wandb_entity ${wandb_entity} \
    --wandb_project ${wandb_project}