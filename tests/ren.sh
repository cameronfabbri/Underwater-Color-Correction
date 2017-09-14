# eval.py spits out images called n*_gen.png and n*_real.png
# This script renames them to scrap the real and gen in the test images folder


rm test_images/ugan_0.0/*real*
rm test_images/ugan_1.0/*real*

rm test_images/ugan_fl_0.0/*real*
rm test_images/ugan_fl_1.0/*real*

rename 's/_gen//' test_images/ugan_0.0/*
rename 's/_gen//' test_images/ugan_1.0/*

rename 's/_gen//' test_images/ugan_fl_0.0/*
rename 's/_gen//' test_images/ugan_fl_1.0/*

