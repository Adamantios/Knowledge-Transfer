# Knowledge-Transfer
Multiple methods' implementations to transfer the knowledge between Neural Networks and save/plot/compare the results.  
- Built with **Python 3.8** and **Tensorflow 2.2**.
- Implements **Knowledge Distillation (KD)**, **Probabilistic Knowledge Transfer (PKT)** and **KD + PKT**.  
- Includes an expirimental method implementation for *student selective learning*.

# Where it was used
- My diploma thesis: [**Lightweight Deep Learning For Embedded Intelligence**](https://github.com/Adamantios/AI-MSc-Thesis)
- PRLetters Journal Publcation: [**Improving knowledge distillation using unified ensembles of specialized teachers**](https://authors.elsevier.com/a/1crJf_3qHiUPK5)

# Usage
```
usage: knowledge_transfer.py [-h]
                             [-m {distillation,pkt,pkt+distillation} [{distillation,pkt,pkt+distillation} ...]]
                             [-sl] [-w START_WEIGHTS] [-t TEMPERATURE]
                             [-kdl KD_LAMBDA_SUPERVISED]
                             [-pktl PKT_LAMBDA_SUPERVISED] [-k NEIGHBORS]
                             [-kdw KD_IMPORTANCE_WEIGHT]
                             [-pktw PKT_IMPORTANCE_WEIGHT] [-ufm]
                             [-s {all,best,none}] [-or]
                             [-res RESULTS_NAME_PREFIX] [-out OUT_FOLDER]
                             [-o {adam,rmsprop,sgd,adagrad,adadelta,adamax}]
                             [-lr LEARNING_RATE] [-lrp LEARNING_RATE_PATIENCE]
                             [-lrd LEARNING_RATE_DECAY]
                             [-lrm LEARNING_RATE_MIN]
                             [-esp EARLY_STOPPING_PATIENCE] [-cn CLIP_NORM]
                             [-cv CLIP_VALUE] [-b1 BETA1] [-b2 BETA2]
                             [-rho RHO] [-mm MOMENTUM] [-d DECAY]
                             [-bs BATCH_SIZE] [-ebs EVALUATION_BATCH_SIZE]
                             [-e EPOCHS] [-v VERBOSITY] [--debug]
                             teacher student
                             {cifar10,cifar100,svhn_cropped,fashion_mnist,mnist}

Transfer the knowledge between two Neural Networks, using different methods
and compare the results.

positional arguments:
  teacher               Path to a trained teacher network.
  student               Path to a student network to be used.
  {cifar10,cifar100,svhn_cropped,fashion_mnist,mnist}
                        The name of the dataset to be used.

optional arguments:
  -h, --help            show this help message and exit
  -m {distillation,pkt,pkt+distillation} [{distillation,pkt,pkt+distillation} ...], 
      --method {distillation,pkt,pkt+distillation} [{distillation,pkt,pkt+distillation} ...]
                        The KT method(s) to be used. 
                        (default ['distillation', 'pkt', 'pkt+distillation']).
  -sl, --selective_learning
                        Whether the models should be designed for the KT with
                        Selective Learning framework (default False).
  -w START_WEIGHTS, --start_weights START_WEIGHTS
                        Filepath containing existing weights to initialize the
                        model.
  -t TEMPERATURE, --temperature TEMPERATURE
                        The temperature for the distillation (default 2).
  -kdl KD_LAMBDA_SUPERVISED, --kd_lambda_supervised KD_LAMBDA_SUPERVISED
                        The lambda value for the KD supervised term (default
                        0.1).
  -pktl PKT_LAMBDA_SUPERVISED, --pkt_lambda_supervised PKT_LAMBDA_SUPERVISED
                        The lambda value for the PKT supervised term (default
                        0.0001).
  -k NEIGHBORS, --neighbors NEIGHBORS
                        The number of neighbors for the PKT method evaluation
                        (default 5).
  -kdw KD_IMPORTANCE_WEIGHT, --kd_importance_weight KD_IMPORTANCE_WEIGHT
                        The importance weight for the KD loss, if method is
                        PKT plus KD (default 1).
  -pktw PKT_IMPORTANCE_WEIGHT, --pkt_importance_weight PKT_IMPORTANCE_WEIGHT
                        The importance weight for the PKT loss, if method is
                        PKT plus KD (default 1).
  -ufm, --use_final_model
                        Whether the final model should be used for saving and
                        results evaluation and not the best one achieved
                        through the training procedure (default False).
  -s {all,best,none}, --save_students {all,best,none}
                        The save mode for the final student networks. (default
                        best).
  -or, --omit_results   Whether the KT comparison results should not be saved
                        (default False).
  -res RESULTS_NAME_PREFIX, --results_name_prefix RESULTS_NAME_PREFIX
                        The prefix for the results filenames (default ).
  -out OUT_FOLDER, --out_folder OUT_FOLDER
                        Path to the folder where the outputs will be stored
                        (default out).
  -o {adam,rmsprop,sgd,adagrad,adadelta,adamax}, --optimizer {adam,rmsprop,sgd,adagrad,adadelta,adamax}
                        The optimizer to be used. (default adam).
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate for the optimizer (default 0.001).
  -lrp LEARNING_RATE_PATIENCE, --learning_rate_patience LEARNING_RATE_PATIENCE
                        The number of epochs to wait before decaying the
                        learning rate (default 8).
  -lrd LEARNING_RATE_DECAY, --learning_rate_decay LEARNING_RATE_DECAY
                        The learning rate decay factor. If 0 is given, then
                        the learning rate will remain the same during the
                        training process. (default 0.1).
  -lrm LEARNING_RATE_MIN, --learning_rate_min LEARNING_RATE_MIN
                        The minimum learning rate which can be reached
                        (default 1e-08).
  -esp EARLY_STOPPING_PATIENCE, --early_stopping_patience EARLY_STOPPING_PATIENCE
                        The number of epochs to wait before early stopping. If
                        0 is given, early stopping will not be applied.
                        (default 15).
  -cn CLIP_NORM, --clip_norm CLIP_NORM
                        The clip norm for the optimizer (default None).
  -cv CLIP_VALUE, --clip_value CLIP_VALUE
                        The clip value for the optimizer (default None).
  -b1 BETA1, --beta1 BETA1
                        The beta 1 for the optimizer (default 0.9).
  -b2 BETA2, --beta2 BETA2
                        The beta 2 for the optimizer (default 0.999).
  -rho RHO              The rho for the optimizer (default 0.9).
  -mm MOMENTUM, --momentum MOMENTUM
                        The momentum for the optimizer (default 0.0).
  -d DECAY, --decay DECAY
                        The decay for the optimizer (default 1e-06).
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size for the optimization (default 64).
  -ebs EVALUATION_BATCH_SIZE, --evaluation_batch_size EVALUATION_BATCH_SIZE
                        The batch size for the evaluation (default 128).
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs to train the network (default
                        125).
  -v VERBOSITY, --verbosity VERBOSITY
                        The verbosity for the optimization procedure (default
                        1).
  --debug               Whether debug mode should be enabled (default False).
```

# Example Run on CIFAR-10
## Command
```
knowledge_transfer.py \
'teacher.h5' \
'student.h5' \
cifar10 \
--method distillation, pkt, pkt+distillation \
--results_name_prefix SL_strong \
--temperature 6 \
--kd_lambda_supervised 0.3 \
--kd_importance_weight 1 \
--learning_rate 1e-4 \
--optimizer adam \
--epochs 150 \
--learning_rate_decay 0.5 \
--learning_rate_min 1e-8 \
--early_stopping_patience 0 \
--selective_learning \
--out_folder 'Knowledge-Transfer/out/cifar10/baseline_v2'
```
## Results
### Log
```
2020-02-04 19:17:42,967 [MainThread  ] [INFO ]  
---------------------------------------------------------------------------------------------

2020-02-04 19:17:42,969 [MainThread  ] [INFO ]  Loading dataset...
2020-02-04 19:18:03,476 [MainThread  ] [INFO ]  Preprocessing data...
2020-02-04 19:18:03,862 [MainThread  ] [INFO ]  Preparing selective_learning KT framework...
2020-02-04 19:18:04,620 [MainThread  ] [INFO ]  Getting teacher's predictions...
2020-02-04 19:18:10,947 [MainThread  ] [INFO ]  Starting KT method(s)...
2020-02-04 19:18:10,948 [MainThread  ] [INFO ]  Performing Knowledge Distillation...
2020-02-04 20:18:04,402 [MainThread  ] [INFO ]  Performing Probabilistic Knowledge Transfer...
2020-02-04 21:14:39,540 [MainThread  ] [INFO ]  Performing PKT plus Distillation...
2020-02-04 22:27:24,137 [MainThread  ] [INFO ]  Evaluating results...
2020-02-04 22:27:24,155 [MainThread  ] [INFO ]  Evaluating Knowledge Distillation...
2020-02-04 22:27:31,018 [MainThread  ] [INFO ]  Evaluating Probabilistic Knowledge Transfer...
2020-02-04 22:27:43,570 [MainThread  ] [INFO ]  Evaluating PKT plus Distillation...
2020-02-04 22:27:47,472 [MainThread  ] [INFO ]  Evaluating Teacher...
2020-02-04 22:27:58,568 [MainThread  ] [INFO ]  Final results: 
Parameters:
    Teacher params: 925182
    Student params: 106922
    Ratio: T/S=8.653 S/T=0.1156
Knowledge Distillation: 
    loss: 0.04038
    categorical_accuracy: 0.8419
    categorical_crossentropy: 0.7592
Probabilistic Knowledge Transfer: 
    loss: 0.1777
    categorical_accuracy: 0.7925
    categorical_crossentropy: 5.226
PKT plus Distillation: 
    loss: 0.08202
    categorical_accuracy: 0.8379
    categorical_crossentropy: 1.585
Teacher: 
    loss: 0.3615
    categorical_accuracy: 0.8475
    categorical_crossentropy: 0.465

2020-02-04 22:27:58,569 [MainThread  ] [INFO ]  Saving student network(s)...
2020-02-04 22:28:06,425 [MainThread  ] [INFO ]  Student network has been saved as Knowledge-Transfer/out/cifar10/baseline/SL_strong_Knowledge Distillation_model.h5.
2020-02-04 22:28:06,449 [MainThread  ] [INFO ]  Student network has been saved as Knowledge-Transfer/out/cifar10/baseline/SL_strong_Probabilistic Knowledge Transfer_model.h5.
2020-02-04 22:28:11,739 [MainThread  ] [INFO ]  Student network has been saved as Knowledge-Transfer/out/cifar10/baseline/SL_strong_PKT plus Distillation_model.h5.
2020-02-04 22:28:11,741 [MainThread  ] [INFO ]  Saving results...
2020-02-04 22:28:11,766 [MainThread  ] [INFO ]  Finished!
```

### Comparison
![Methods Comparison](https://github.com/Adamantios/Knowledge-Transfer/blob/master/examples/SL_strong_KT_Methods_Comparison_accuracy_vs_epoch.png?raw=true)

### Useful output files
Some useful files (unless otherwise specified) are saved in the chosen out folder destination. 
These include:
- the final student(s) weights
- the student model(s) (in `keras` format)
- all the methods resutls in a pickle file, which also contains all the training histories
