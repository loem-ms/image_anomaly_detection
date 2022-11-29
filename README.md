# image_anomaly_detection

## Setup
- We confirmed this work with `Python3.10` on macOS
- Please install `node.js` and `npm ` to run the application
- Install required libraries/modules in `requirements.txt` for backend-related requirements
```bash
    % pip install -r requirements
```
- Install additional requirments for frontend (loading symbol while waiting response)
```bash
    % npm install react-promise-tracker
    % npm install react-loader-spinner
```

## Front-end
```bash
    % cd frontend/app
    % npm start
```

## Back-end
```bash
    % cd backend
    # Run server
    % python server.py \
        --model_checkpoint ./checkpoints/checkpoint_for_demo.pth \
        --encoded_trainingdata ./checkpoints/train_for_demo.csv
```

## How to prepare model for backend?
```bash
    # Training model
    % python train.py --epoch_num 200 \
        --image_class "pill" \
        --learning_rate 0.001 \
        --batch_size 8 \
        --seed 0 \
        --encoded_dim 64 \
        --feedforward_hidden_dim 512 \
        --dropout_rate 0.3 \
        --log_file ./checkpoints/train.log \
        --dataset_dir ./dataset \
        --checkpoint_dir ./checkpoints
    # Encoding: encode training data with trained model 
    # The encoded vectors will be used for Gaussian Density Estimation
    # for setting threshold in checking anomaly 
    % python encode.py --batch_size 8 \
        --encoded_dim 64 \
        --feedforward_hidden_dim 512 \
        --log_file log_encode.txt \
        --dataset_dir ./dataset \
        --image_class "pill" \
        --checkpoint ./checkpoints/checkpoint_last.pth \
        --subset_to_encode "train" \
        --output_file_pref ./checkpoints
    
```