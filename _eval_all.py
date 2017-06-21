export CUDA_VISIBLE_DEVICES=""
find checkpoints/ -iname 'info.pkl' -exec python eval.py {} \;
