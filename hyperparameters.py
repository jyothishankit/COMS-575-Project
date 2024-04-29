CUT_LENGTH_CONSTANT = 16000*2
#Cut length of audio samples, for denoising and deverberation
INITIAL_LEARNING_RATE_CONSTANT = 5e-4
#Learning Rate during training
TOTAL_LOSS_WEIGHTS_CONSTANT = [0.1, 0.9, 0.2, 0.05]
#Weighted loss weight measures
#Weights of real component, imaginary component, magnitude, time loss, and Metric Disc
BATCH_SIZE_CONSTANT = 4
#Batch size of samples during training
EPOCHS_CONSTANT = 120
#Total number of epochs
DECAY_EPOCH_INTERVAL_CONSTANT = 30
#Epoch interval for decay of learning rate