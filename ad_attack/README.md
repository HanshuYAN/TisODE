On Robustness of Neural Ordinary Differential Equations
=====================================
Codes 

Requirements
-------------------------------------
python3.6.8
torchvision                                                0.2.2  
pytorch                   1.0.1  
torchdiffeq

Run attacks
--------------------------------------
Codes for adversarial attack are in `attack_mnist` and `attack_imagenet3000`   
Execute `cnn.sh` for some examples of attacks. 

```
python test_all.py --dir_model <path> --learning_rate 6e-2 --epsilon 3e-1 --maxiter 10
```
--maxiter defines the maximum iteration for FGSM/PGD attack; when --maxiter > 2, the attack method would be PGD; otherwise the attack method is FGSM.  

`test_all.py` interface for running adversarial attack.   
`data.py` scripts for dataloader.  
`load_attacked_and_meta_model.py` scripts for loading the targeted networks.   
`options.py` scripts for arguments parser.     
`network.py` networks.  
`attack/` adversarial attack codes


