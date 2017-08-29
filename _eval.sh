export CUDA_VISIBLE_DEVICES=""
find . -iname 'info.pkl' -exec python eval.py {} \;
