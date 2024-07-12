# Modelsaved file
The code is set to save the model after a certain period of time, and the model saves the following:
```sh
Steps:   0%|                                                                 | 1000/100000000000 [17:11<27774189:14:01,  1.00it/s, los
- INFO - accelerate.accelerator - Saving current state to ../modelsaved/OUTPUT_DIR/checkpoint-1000
Configuration saved in ../modelsaved/OUTPUT_DIR/checkpoint-1000/controlnet_nd/config.json
Model weights saved in ../modelsaved/OUTPUT_DIR/checkpoint-1000/controlnet_nd/diffusion_pytorch_model.safetensors
- INFO - accelerate.checkpointing - Model weights saved in ../modelsaved/OUTPUT_DIR/checkpoint-1000/pytorch_model.bin
- INFO - accelerate.checkpointing - Optimizer state saved in ../modelsaved/OUTPUT_DIR/checkpoint-1000/optimizer.bin
- INFO - accelerate.checkpointing - Optimizer state saved in ../modelsaved/OUTPUT_DIR/checkpoint-1000/optimizer_1.bin
- INFO - accelerate.checkpointing - Scheduler state saved in ../modelsaved/OUTPUT_DIR/checkpoint-1000/scheduler.bin
- INFO - accelerate.checkpointing - Sampler state for dataloader 0 saved in ../modelsaved/OUTPUT_DIR/checkpoint-1000/sampler.bin
- INFO - accelerate.checkpointing - Random states saved in ../modelsaved/OUTPUT_DIR/checkpoint-1000/random_states_0.pkl
- INFO - __main__ - Saved state to ../modelsaved/OUTPUT_DIR/checkpoint-1000
... ...
````
