rm testA/*
rm testB/*

# convert to underwater looking
python main.py --which_direction=AtoB --phase=test --dataset=underwater_imagenet --checkpoint_dir=checkpoint/underwater_imagenet_256/

#python main.py --which_direction=BtoA --phase=test --dataset=underwater_imagenet --checkpoint_dir=checkpoint/underwater_imagenet_256/underwater_imagenet_256/
