import torch
import torch.optim as optim
import torch.nn as nn

from src.generator import Generator
import src.mediator import Mediator
import src.helpers

from src.config import *


def init_models():
    oracle = Generator(
        EMB_DIM, 
        HIDDEN_DIM, 
        VOCAB_SIZE, 
        SEQ_LENGTH, 
        gpu=CUDA, 
        oracle_init=True)

    gen = Generator(
        EMB_DIM, 
        HIDDEN_DIM, 
        VOCAB_SIZE, 
        SEQ_LENGTH, 
        gpu=CUDA)
    
    med = Mediator(
        EMB_DIM*2, 
        HIDDEN_DIM*2, 
        VOCAB_SIZE, 
        SEQ_LENGTH, 
        gpu=CUDA)
    
       oracle_samples = torch.load(
           oracle_samples_path).type(torch.long)
    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        mid = mid.cuda()
        oracle_samples = oracle_samples.cuda()
    return oracle, gen, med, oracle_samples 
        

if __name__ == '__main__':
    # First init models
    oracle, gen, med, oracle_samples = init_models()
    
    if RESTORE:
        # TODO: load pretrained model
        pass
    for epoch in range(PRE_EPOCH_NUM):
        # TODO: Just MLE training
        pass
    
    print('#########################################################################')
    print('Start Cooperative Training...')
    for iter_idx in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = gen.sample(BATCH_SIZE)
            rewards = med.get_reward(samples)

            g_loss = gen.g_loss(samples, rewards)

            g_loss.backward()
            g_opt.step()

        # Train the mediator
        for _ in range(1):
            bnll_ = []
            # update mediator main 198
        
        # every 10 epoch status / save
        #if iter_idx % 10 == 0:
            #jsd = jsd_calculate(sess, generator, target_lstm)
            #saver.save(sess, "saved_model/CoT")