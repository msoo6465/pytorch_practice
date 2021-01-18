Training
-----------
To train a model

    python testMINST.py --lr 0.1 -b 100 --epochs 100 
    
if checkpoint or bestmodel are existed

    python testMINST.py --lr 0.1 --resume 'checkpoint.pth' -b 100 

Test
--------------
    python testMINST.py -e --resume 'model_best.pth.tar'