# image_anomaly_detection

## Setup
- We confirmed process with `Python3.10` on macOS
- Install `node.js` and `npm `
- Install requirements descripted in `requirements.txt`
```bash
    % pip install -r requirements
```
- Install requirments for frontend
```bash
    % npm install react-promise-tracker
    % npm install react-loader-spinner
```

## Front-end
```bash
    % cd frontend
    % npx create-react-app app
```

## Back-end
```bash
    % cd backend
    # Training
    % python train.py --epoch_num 200 \
        --learning_rate 0.001 \
        --batch_size 8 \
        --seed 0 \
        --encoded_dim 64 \
        --feedforward_hidden_dim 512 \
        --dropout_rate 0.3 \
        --log_file ./checkpoints/train.log \
        --dataset_dir ./dataset \
        --checkpoint_dir ./checkpoints
    # Encoding
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

## Run Server
```bash
    % cd backend
    % python server.py \
        --model_checkpoint ./checkpoints/checkpoint_for_demo.pth \
        --encoded_trainingdata ./checkpoints/train_for_demo.csv
```

```bash
    % cd frontend/app
    % npm start
```