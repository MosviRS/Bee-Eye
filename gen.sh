$ find positive_images/ -name '.jpg' -exec echo {} 1 0 0 32 32 ; > pos.dat
$ find negative_images/ -name '.jpg' > neg.dat