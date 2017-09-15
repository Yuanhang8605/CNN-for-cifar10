This is a framework for cnn training and eval.
The directories contain:

to run the model, use script in ./scripts directory
	-- ./scripts:
		train.sh    	script to train the model
		eval.sh		script to eval the model
		demo.sh		script to run the demo

	-- ./network:
		ops.py	  	capsulate the common ops used in cnn, include conv, pooling, fc etc. It's very useful for construct deep networks
		net.py    	define base class Net for cnn networks class, define the interface to Solver class 
		resnet.py  	resnet network class, derive from Net class
		inception.py  	inception network class
		vgg.py		vgg network class 
		
	-- ./train
		solver.py  	define a Solver class to config training and eval  
		run.py
			main function to run training or eval
			args define:
			--mode	   			train or test
			--dataset  			dataset name, like cifar10 
			--train_data_path		training data file path
			--eval_data_path		evaluate data file path
			--evaluate_once			whether or not evaluate only once
			--eval_batch_count		in every loop, how many batch count to be evaluate
			--num_gpus			the number of GPU	

		demo.py    restore the saved model and show demo
	
	-- ./data:
		cifar_input.py		code to read binary format data and preprocess data, transfer data to tf queue style
		./cifar10
		./Imagenet
		...
	
	-- ./config
		hyperparameters in dict format in python file, like resnet_config.py
		include the config file

	-- ./cc
		c++ code
		setup.py use cython to setup the c++ code
