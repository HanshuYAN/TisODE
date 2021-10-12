"""
    =================version1 ======================
    This is the first version of our meta model, 
    we first use original zoo attack method to generate the final noise on 10 randomly selected images,
    we use the resulted images and noise to finetune our original meta model for five steps.
    Then we select another 100 images for attacking,
    we input the image to the finetuned meta-model to generate the noise as desired.
    we add the noise to the image and query whether the image is classified.
    if successful, we return the query times and l2 distance
    if not, we construct the input and the noise together as a new input to meta model to generate noise and repeat the above process.
    finally we return the query times, l2 distortion and whether it is sucessful
    
    the final result is saved in meta_log_version1.txt

"""
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from numba import jit
from torch import optim
from torch import autograd
from .helpers import *
import pdb

#@jit(nopython=True)
def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
    #for i in range(batch_size):
    #    grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002
   
    #grad = torch.from_numpy(grad).cuda()
    # pdb.set_trace()
    
    mt = mt_arr[indice]
    mt = beta1 * mt + (1 - beta1) * grad
    mt_arr[indice] = mt
    vt = vt_arr[indice]
    vt = beta2 * vt + (1-beta2) * (grad * grad)
    vt_arr[indice] = vt
    epoch = adam_epoch[indice]
    corr = (torch.sqrt(1 - torch.pow(beta2, epoch))) / (1 - torch.pow(beta1, epoch))
    #if self.cuda:
    corr = corr.cuda()
    m = real_modifier.reshape(-1)
    old_val = m[indice]
    old_val -= lr * corr * mt / (torch.sqrt(vt) + 1e-8)
    if proj:
        old_val = torch.max(torch.min(old_val, up[indice]), down[indice])
        #old_val = old_val/torch.norm(old_val, 2)*1.5
    m[indice] = old_val
    adam_epoch[indice] = epoch + 1.

class BlackBoxL2:

    def __init__(self, targeted = False, search_steps = None, max_steps = None, cuda = True, debug = False):
        self.debug = debug
        self.targeted = targeted # false
        self.num_classes = 10 
        self.confidence = 0  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = 0.5  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 5
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or 1000
        self.abort_early = True
        self.clip_min = -1
        self.clip_max = 1
        self.cuda = cuda
        self.clamp_fn = ''  # set to something else perform a simple clamp instead of tanh
        self.init_rand = False  # an experiment, does a random starting point help?
        self.use_log = True
        self.use_tanh = False
        self.batch_size = 784
       
        self.use_importance = True
        self.LEARNING_RATE = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        
        self.reset_adam_after_found = False
        self.num_channels = 1
        self.small_x = 28
        self.small_y = 28
        var_size = self.small_x * self.small_y * self.num_channels
        self.use_var_len = var_size
        self.var_list = np.array(range(0, self.use_var_len), dtype = np.int32)
        self.sample_prob = np.ones(var_size, dtype = np.float32) / var_size

        self.mt = torch.zeros(var_size, dtype = torch.float32)
        self.vt = torch.zeros(var_size, dtype = torch.float32)
        self.modifier_up = torch.zeros(var_size, dtype = torch.float32)
        self.modifier_down = torch.zeros(var_size, dtype = torch.float32)
        self.grad = torch.zeros(self.batch_size, dtype = torch.float32)
        self.hess = torch.zeros(self.batch_size, dtype = torch.float32)
        self.adam_epoch = torch.ones(var_size, dtype = torch.float32)

        self.solver_name = 'adam'

        if self.solver_name == 'adam':
            self.solver = coordinate_ADAM
        elif solver != 'fake_zero':
            print('unknown solver', self.solver_name)
            self.solver = coordinate_ADAM
        print('Using', self.solver_name, 'solver')

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            if self.use_log:
                loss1 = torch.clamp(torch.log(other + 1e-30) - torch.log(real + 1e-30), min = 0.)
            else:
            # if targeted, optimize for making the other class most likely
                loss1 = torch.clamp(other - real + self.confidence, min = 0.)  # equiv to max(..., 0.)
        else:
            if self.use_log:
                loss1 = torch.clamp(torch.log(real + 1e-30) - torch.log(other + 1e-30), min = 0.)
            else:
            # if non-targeted, optimize for making this class least likely.
                loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = scale_const * loss1

        loss2 = dist.squeeze(1)
        #print('loss1 and loss2 is:', loss1, loss2)
        loss = loss1 + loss2
        #pdb.set_trace()
        return loss, loss1, loss2

    def _optimize(self, model, meta_model, step, input_var, modifier_var, target_var, scale_const_var, input_orig = None):
        noise = meta_model(input_var)
        noise = noise/ torch.norm(noise, 2)
        modifier_var += noise
        input_adv = input_var + modifier_var/torch.norm(modifier_var, 2)
        input_adv = torch.clamp(input_adv, 0, 1)
        output = F.softmax(model(input_adv), dim = 1)
        
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim = True).squeeze(2).squeeze(2)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim = True).squeeze(2).squeeze(2)

        loss, loss1, loss2 = self._loss(output.data, target_var, dist, scale_const_var)
        
        loss_np = loss[0].item()
        loss1 = loss1[0].item()
        loss2 = loss2[0].item()
        dist_np = dist[0].data.cpu().numpy()
        output_np = output[0].unsqueeze(0).data.cpu().numpy()
        input_adv_np = input_adv[0].unsqueeze(0).data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
        
        return loss_np, loss1, loss2, dist_np, output_np, input_adv_np

    def run(self, model, meta_model, input, target, batch_idx = 0):
        batch_size, c, h, w = input.size()
        
        #pdb.set_trace() 
        
        var_size = c * h * w
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        if not self.use_tanh:
            self.modifier_up = 0.5 - input.reshape(-1)
            self.modifier_down = - 0.5 - input.reshape(-1)
        
        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = input.permute(0, 2, 3, 1).cpu().numpy()

        # setup input (image) variable, clamp/scale as necessary
        if self.clamp_fn == 'tanh':
            # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
            # this as per the reference implementation or can we skip the arctanh?
            input_var = autograd.Variable(torch_arctanh(input * 2), requires_grad = False)
            input_orig = tanh_rescale(input_var, self.clip_min, self.clip_max) / 2
        else:
            input_var = autograd.Variable(input, requires_grad=False)
            input_orig = None

        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        
        self.mt = torch.zeros(var_size, dtype = torch.float32)
        self.vt = torch.zeros(var_size, dtype = torch.float32)
        self.meta_grad = torch.zeros(input_var.shape, dtype = torch.float32)
       
        self.meta_optimizer = optim.Adam(meta_model.parameters(), lr = 1e-2)

        self.adam_epoch = torch.ones(var_size, dtype = torch.float32)
        stage = 0
        eval_costs = 0
        
        if self.init_rand:
            # Experiment with a non-zero starting point...
            modifier = torch.normal(means=modifier, std=0.001)
        if self.cuda:
            modifier = modifier.cuda()
            
            self.modifier_up = self.modifier_up.cuda()
            self.modifier_down = self.modifier_down.cuda()
            self.mt = self.mt.cuda()
            self.vt = self.vt.cuda()
            
            self.meta_grad = self.meta_grad.cuda()

            self.grad = self.grad.cuda()
            self.hess = self.hess.cuda()

        #modifier_var = autograd.Variable(modifier, requires_grad=True)
        modifier_var = modifier
        
        for search_step in range(self.binary_search_steps):
            print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if self.debug:
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = autograd.Variable(scale_const_tensor, requires_grad = False)

            prev_loss = 1e6
            
            last_loss1 = 1.0
           
            first_step = 0
            # Inner loop step
            
            input_adv = input_var
            for step in range(self.max_steps):
                '''
                loss, loss1, loss2, dist, output, adv_img = self._optimize(
                    model,
                    meta_model,
                    step,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    input_orig)
                '''
                noise = meta_model(input_adv)
                noise = noise/ torch.norm(noise, 2)

                modifier_var += noise
                input_adv = input_var + 1.1 * modifier_var/torch.norm(modifier_var, 2)
                input_adv = torch.clamp(input_adv, -0.5, 0.5)
                output = F.softmax(model(input_adv), dim = 1)
                
                target_label = target[0]
                output_logits = output[0]
                dist = l2_dist(input_adv, input_var, keepdim = True).squeeze(2).squeeze(2)

                if target_label == torch.argmax(output_logits):
                    print('unsuccessful in step', step, 'and the distance is', dist[0].data.cpu().numpy())
                first_step = 0

                '''
                loss, loss1, loss2 = self._loss(output.data, target_var, dist, scale_const_var)
        
                loss = loss[0].item()
                loss1 = loss1[0].item()
                loss2 = loss2[0].item()
                dist = dist[0].data.cpu().numpy()
                output = output[0].unsqueeze(0).data.cpu().numpy()
                adv_img = input_adv[0].unsqueeze(0).data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
            
                if step % 100 == 0 or step == self.max_steps - 1:
                    print('Step: {0:>4}, loss: {1:6.4f}, loss1: {2:5f}, loss2: {3:5f}, dist: {4:8.5f}, modifier mean: {5:.5e}'.format(
                        step, loss, loss1, loss2, dist.mean(), modifier_var.data.mean()))
                
                if self.abort_early and step % (self.max_steps // 10) == 0:
                    if loss > prev_loss * .9999:
                        print('Aborting early...')
                        break
                    prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if di < best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                    if di < o_best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[i] = adv_img[i]
                        
                        first_step = step

                sys.stdout.flush()
                # end inner step loop
        
            # adjust the constants
            batch_failure = 0
            batch_success = 0
            
            for i in range(batch_size):
                if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if self.debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if self.debug:
                        print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop
                '''
        return o_best_attack, scale_const, first_step
