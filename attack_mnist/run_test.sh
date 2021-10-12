#-**********************************************************************************-#
# CNN
#-**********************************************************************************-#
# gaus
# python main_cnn.py --device_ids 0 --isTrain False --isODE False --dir_model ../../checkpoints/mnist/MNIST_CNN_gaus_1/model_best.pth --logging_file sharing_MNIST_FGSM.txt
# python main_cnn.py --device_ids 0 --isTrain False --isODE False --dir_model ../../checkpoints/mnist/MNIST_CNN_gaus_2/model_best.pth --logging_file sharing_MNIST_FGSM.txt
# python main_cnn.py --device_ids 0 --isTrain False --isODE False --dir_model ../../checkpoints/mnist/MNIST_CNN_gaus_3/model_best.pth --logging_file sharing_MNIST_FGSM.txt


#-**********************************************************************************-#
# ODE
#-**********************************************************************************-#
# gaus
# python main_ode.py --device_ids 0 --isTrain False --isODE True --TimePeriod 0_1 --step_size 0.1 --dir_model ../../checkpoints/mnist/MNIST_ODE_gaus_1/model_best.pth --logging_file sharing_MNIST_FGSM.txt
# python main_ode.py --device_ids 0 --isTrain False --isODE True --TimePeriod 0_1 --step_size 0.1 --dir_model ../../checkpoints/mnist/MNIST_ODE_gaus_2/model_best.pth --logging_file sharing_MNIST_FGSM.txt
# python main_ode.py --device_ids 0 --isTrain False --isODE True --TimePeriod 0_1 --step_size 0.1 --dir_model ../../checkpoints/mnist/MNIST_ODE_gaus_3/model_best.pth --logging_file sharing_MNIST_FGSM.txt


#-**********************************************************************************-#
# Tisode
#-**********************************************************************************-#
# tisode (new of new version)
python main_tisode.py --device_ids 0_1 --isTrain False --isODE True --TimePeriod 0_0.75_1_1.05 --step_size 0.05 --dir_model ../../checkpoints/mnist/MNIST_tisode_gaus_1/model_best.pth --logging_file test.txt
# python main_tisode.py --device_ids 0_1 --isTrain False --isODE True --TimePeriod 0_0.75_1_1.05 --step_size 0.05 --dir_model ../../checkpoints/mnist/MNIST_tisode_gaus_2/model_best.pth --logging_file test.txt
# python main_tisode.py --device_ids 0_1 --isTrain False --isODE True --TimePeriod 0_0.75_1_1.05 --step_size 0.05 --dir_model ../../checkpoints/mnist/MNIST_tisode_gaus_3/model_best.pth --logging_file test.txt





