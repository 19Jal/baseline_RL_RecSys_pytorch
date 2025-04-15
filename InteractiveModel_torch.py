"""
Created based on TensorFlow implementation
Converted to PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple


class BasicDecoderHit(NamedTuple):
    hit: torch.Tensor


class BasicDecoderOutput(NamedTuple):
    rnn_output: torch.Tensor
    sample_id: torch.Tensor


class SimpleEmbeddingWrapper(nn.Module):
    def __init__(self, cell, embeddings):
        super(SimpleEmbeddingWrapper, self).__init__()
        self._cell = cell
        self.embeddings = embeddings
        
    def forward(self, inputs, state):
        # Convert input tokens to embeddings
        inputs_shape = inputs.shape
        inputs_flat = inputs.view(-1)
        embedded = self.embeddings(inputs_flat).view(*inputs_shape, -1)
        
        # Run cell with embedded inputs
        return self._cell(embedded, state)
    
    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class InteractiveGreedyEmbeddingHelper:
    def __init__(self, embedding, k, start_tokens, start_hit, size, sequence_length, device):
        self.embedding = embedding
        self._k = k
        self._start_tokens = torch.tensor(start_tokens, dtype=torch.long, device=device)
        self.start_hit = torch.tensor(start_hit, dtype=torch.long, device=device)
        self._batch_size = self._start_tokens.size(0)
        self._start_inputs = self.embedding(self._start_tokens) * self.start_hit
        self._output_size = size
        self._sequence_length = sequence_length
        self.device = device
    
    def initialize(self):
        finished = torch.zeros(self._batch_size, dtype=torch.bool, device=self.device)
        return finished, self._start_inputs
    
    def sample(self, time, outputs):
        sample_ids = torch.argmax(outputs, dim=-1)
        return sample_ids
    
    def next_inputs(self, time, outputs, state, history_masking, interesting):
        next_time = time + 1
        finished = (next_time >= self._sequence_length)
        all_finished = torch.all(finished)
        
        # Apply masking to outputs
        masking_outputs = outputs * history_masking
        
        # Get top-k items
        topk_values, topk_indices = torch.topk(masking_outputs, self._k, dim=-1)
        one_hot_top_n_index = torch.zeros_like(outputs)
        one_hot_top_n_index.scatter_(-1, topk_indices, 1.0)
        
        # Calculate interesting scores
        current_interesting_index = interesting * one_hot_top_n_index
        hit_flag = torch.sum(current_interesting_index, dim=-1) > 0
        hit_flag_factor = hit_flag.float().view(-1, 1)
        
        # Determine item selection strategy
        current_interesting_score = hit_flag_factor * current_interesting_index * outputs + \
                                   (1 - hit_flag_factor) * masking_outputs
        
        # Select the item
        selected_item = torch.argmax(current_interesting_score, dim=-1)
        one_hot_selected_item = torch.zeros_like(outputs)
        one_hot_selected_item.scatter_(-1, selected_item.unsqueeze(-1), 1.0)
        
        # Get embedding for the selected item
        emb = self.embedding(selected_item)
        output_emb = hit_flag_factor * emb + (hit_flag_factor - 1) * emb
        
        # Update history masking
        next_history_masking = history_masking - hit_flag_factor * one_hot_selected_item
        
        # Determine next inputs
        next_inputs = output_emb if not all_finished else self._start_inputs
        
        return (finished, next_inputs, state, selected_item, next_history_masking, 
                interesting, hit_flag_factor, current_interesting_index)


class InteractiveDecoder(nn.Module):
    def __init__(self, cell, helper, initial_state, initial_history_masking, interesting, output_layer=None):
        super(InteractiveDecoder, self).__init__()
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer
        self._initial_history_masking = initial_history_masking
        self._interesting = interesting
    
    def initialize(self):
        finished, first_inputs = self._helper.initialize()
        return finished, first_inputs, self._initial_state, self._initial_history_masking, self._interesting
    
    def step(self, time, inputs, state, history):
        # Run cell on inputs
        cell_outputs, cell_state = self._cell(inputs, state)
        
        # Apply output layer if provided
        if self._output_layer is not None:
            cell_outputs = self._output_layer(cell_outputs)
        
        # Get next inputs and state from helper
        (finished, next_inputs, next_state, sample_ids, next_history_masking, 
         next_interesting, hit_flag_factor, current_interesting_index) = self._helper.next_inputs(
            time=time,
            outputs=cell_outputs,
            state=cell_state,
            history_masking=history,
            interesting=self._interesting)
        
        # Create outputs
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        hit = BasicDecoderHit(hit_flag_factor)
        
        return (outputs, next_state, next_inputs, next_history_masking, hit, finished)


class ExternalMemInteractiveDecoder(nn.Module):
    def __init__(self, cell, helper, initial_state, initial_history_masking, interesting, mem, rnn_size, output_layer=None):
        super(ExternalMemInteractiveDecoder, self).__init__()
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer
        self._initial_history_masking = initial_history_masking
        self._interesting = interesting
        
        # Memory components
        self.mem_output_layer = nn.Linear(mem.shape[-1], rnn_size, bias=False)
        self._mem = torch.tanh(self.mem_output_layer(mem))
        self.forget_gate_mem_part = nn.Linear(rnn_size, rnn_size, bias=True)
        self.forget_gate_cell_part = nn.Linear(rnn_size, rnn_size, bias=False)
    
    def initialize(self):
        finished, first_inputs = self._helper.initialize()
        return finished, first_inputs, self._initial_state, self._initial_history_masking, self._interesting
    
    def step(self, time, inputs, state, history):
        # Run cell on inputs
        cell_outputs, cell_state = self._cell(inputs, state)
        
        # Apply memory mechanism
        mem_gate = torch.sigmoid(-self.forget_gate_mem_part(self._mem) + self.forget_gate_cell_part(cell_outputs))
        cell_outputs = mem_gate * self._mem + (1 - mem_gate) * cell_outputs
        
        # Apply output layer if provided
        if self._output_layer is not None:
            cell_outputs = self._output_layer(cell_outputs)
        
        # Get next inputs and state from helper
        (finished, next_inputs, next_state, sample_ids, next_history_masking, 
         next_interesting, hit_flag_factor, current_interesting_index) = self._helper.next_inputs(
            time=time,
            outputs=cell_outputs,
            state=cell_state,
            history_masking=history,
            interesting=self._interesting)
        
        # Create outputs
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        hit = BasicDecoderHit(hit_flag_factor)
        
        return (outputs, next_state, next_inputs, next_history_masking, hit, finished)


def dynamic_interactive_decode(decoder, output_time_major=False, impute_finished=False, 
                              maximum_iterations=None, device=None):
    """
    Dynamic decoding implementation for interactive decoders
    """
    # Initialize decoder
    initial_finished, initial_inputs, initial_state, initial_history_masking, initial_interesting = decoder.initialize()
    
    # Set up time counter and outputs
    time = 0
    outputs_ta = []
    hit_ta = []
    
    # Set up state tracking
    state = initial_state
    inputs = initial_inputs
    history_masking = initial_history_masking
    finished = initial_finished
    
    # Main decoding loop
    while not torch.all(finished) and (maximum_iterations is None or time < maximum_iterations):
        # Run one step of decoder
        outputs, state, inputs, history_masking, hit, step_finished = decoder.step(
            time, inputs, state, history_masking)
        
        # Store outputs
        outputs_ta.append(outputs)
        hit_ta.append(hit)
        
        # Update finished state
        finished = torch.logical_or(finished, step_finished)
        
        # Increment time
        time += 1
    
    # Stack all outputs
    stacked_outputs = BasicDecoderOutput(
        torch.stack([o.rnn_output for o in outputs_ta], dim=0),
        torch.stack([o.sample_id for o in outputs_ta], dim=0)
    )
    
    stacked_hit = BasicDecoderHit(torch.stack([h.hit for h in hit_ta], dim=0))
    
    # Transpose batch and time dimensions if needed
    if not output_time_major:
        stacked_outputs = BasicDecoderOutput(
            stacked_outputs.rnn_output.transpose(0, 1),
            stacked_outputs.sample_id.transpose(0, 1)
        )
        stacked_hit = BasicDecoderHit(stacked_hit.hit.transpose(0, 1))
    
    return stacked_outputs, state, history_masking, stacked_hit, time


class SimpleGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleGRUCell, self).__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
    
    def forward(self, input_tensor, hidden_state):
        return self.gru_cell(input_tensor, hidden_state), hidden_state
    
    def zero_state(self, batch_size, dtype):
        return torch.zeros(batch_size, self.hidden_size, dtype=dtype)


class MultiRNNCell(nn.Module):
    def __init__(self, cells):
        super(MultiRNNCell, self).__init__()
        self.cells = nn.ModuleList(cells)
    
    def forward(self, input_tensor, hidden_states):
        new_states = []
        output = input_tensor
        
        for i, cell in enumerate(self.cells):
            output, new_state = cell(output, hidden_states[i])
            new_states.append(new_state)
        
        return output, tuple(new_states)
    
    def zero_state(self, batch_size, dtype):
        return tuple(cell.zero_state(batch_size, dtype) for cell in self.cells)


class InteractiveModel(nn.Module):
    def __init__(self, rnn_size, layer_size, decoder_vocab_size, embedding_dim, k, lr, device):
        super(InteractiveModel, self).__init__()
        
        self.device = device
        self._k = k
        self.lr = lr
        self.postive_imediate_reward = 1.0
        self.negative_imediate_reward = 0.2
        self.account_ratio = 0.9
        self.rnn_size = rnn_size
        
        # Define embeddings
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, embedding_dim)
        
        # Define RNN cells
        self.decoder_cell = self._get_simple_lstm(rnn_size, layer_size)
        
        # Define output layer
        self.fc_layer = nn.Linear(rnn_size, decoder_vocab_size)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-4)
    
    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [SimpleGRUCell(rnn_size, rnn_size) for _ in range(layer_size)]
        return MultiRNNCell(lstm_layers)
    
    def forward(self, user_interesting, user_masking, sequence_length, init_state=None, s_token=0, s_hit=1.0):
        batch_size = user_interesting.size(0)
        
        # Initialize state if not provided
        if init_state is None:
            init_state = tuple(torch.zeros(batch_size, self.rnn_size, device=self.device) 
                              for _ in range(len(self.decoder_cell.cells)))
        
        # Create helper and decoder
        helper = InteractiveGreedyEmbeddingHelper(
            self.decoder_embedding, self._k, [s_token], [s_hit], 
            user_interesting.size(-1), sequence_length, self.device)
        
        decoder = InteractiveDecoder(
            self.decoder_cell, helper, init_state, user_masking, 
            user_interesting, lambda x: F.softmax(self.fc_layer(x), dim=-1))
        
        # Run dynamic decoding
        outputs, final_state, final_history_masking, hit, final_sequence_lengths = dynamic_interactive_decode(
            decoder, device=self.device, maximum_iterations=sequence_length[0])
        
        # Process outputs
        hit = hit.hit
        logits = outputs
        sample_ids = outputs.sample_id
        
        return logits, sample_ids, final_state, final_history_masking, hit
    
    def compute_rewards(self, hit, sequence_length):
        # Create reversed hit
        reverse_hit = torch.flip(hit, [1])
        
        # Compute immediate rewards
        reverse_imediate_reward = torch.where(
            reverse_hit > 0,
            reverse_hit * self.postive_imediate_reward,
            (reverse_hit - 1) * self.negative_imediate_reward
        )
        
        # Flip back
        imediate_reward = torch.flip(reverse_imediate_reward, [1])
        
        # Compute cumulative rewards
        cumsum_reward = torch.zeros_like(imediate_reward)
        batch_size = hit.size(0)
        
        for b in range(batch_size):
            seq_len = sequence_length[b]
            reward = 0
            for t in range(seq_len - 1, -1, -1):
                reward = reverse_imediate_reward[b, t] + self.account_ratio * reward
                cumsum_reward[b, seq_len - 1 - t] = reward
        
        return imediate_reward, cumsum_reward
    
    def reinforcement_learn(self, user_interesting, user_masking, processes_length, init_state=None, s_token=0, s_hit=1.0):
        self.train()
        self.optimizer.zero_grad()
        
        # Convert inputs to tensors
        user_interesting = torch.tensor(user_interesting, dtype=torch.float, device=self.device)
        user_masking = torch.tensor(user_masking, dtype=torch.float, device=self.device)
        processes_length = torch.tensor([processes_length], dtype=torch.long, device=self.device)
        
        # Forward pass
        logits, sample_ids, final_state, final_history_masking, hit = self.forward(
            user_interesting, user_masking, processes_length, init_state, s_token, s_hit)
        
        # Compute rewards
        _, cumsum_reward = self.compute_rewards(hit, processes_length)
        
        # Convert to numpy for compatibility
        hit_np = hit.detach().cpu().numpy()
        sample_ids_np = sample_ids.detach().cpu().numpy()
        final_history_masking_np = final_history_masking.detach().cpu().numpy()
        
        # One-hot encoding of sample ids
        onehot_sample = F.one_hot(sample_ids, num_classes=user_interesting.size(-1)).float()
        
        # Calculate policy gradient loss
        rnn_output = logits.rnn_output
        gt_ratio = torch.cumprod(torch.ones_like(cumsum_reward) * self.account_ratio, dim=1)
        
        # Compute reinforcement loss
        log_probs = torch.log(rnn_output.clamp(min=1e-8))
        selected_log_probs = torch.sum(log_probs * onehot_sample, dim=-1)
        loss = -torch.mean(selected_log_probs * cumsum_reward * gt_ratio)
        
        # Backward pass and optimize
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        
        self.optimizer.step()
        
        return None, hit_np, final_state, final_history_masking_np, sample_ids_np
    
    def supervised_learn(self, user_interesting, user_masking, targets, processes_length, 
                       init_state=None, s_token=0, s_hit=1.0):
        self.train()
        self.optimizer.zero_grad()
        
        # Convert inputs to tensors
        user_interesting = torch.tensor(user_interesting, dtype=torch.float, device=self.device)
        user_masking = torch.tensor(user_masking, dtype=torch.float, device=self.device)
        processes_length = torch.tensor([processes_length], dtype=torch.long, device=self.device)
        targets = torch.tensor(targets[:processes_length], dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Forward pass
        logits, sample_ids, final_state, final_history_masking, hit = self.forward(
            user_interesting, user_masking, processes_length, init_state, s_token, s_hit)
        
        # Convert to numpy for compatibility
        hit_np = hit.detach().cpu().numpy()
        sample_ids_np = sample_ids.detach().cpu().numpy()
        final_history_masking_np = final_history_masking.detach().cpu().numpy()
        
        # One-hot encoding of targets
        onehot_target = F.one_hot(targets, num_classes=user_interesting.size(-1)).float()
        
        # Compute supervised loss
        rnn_output = logits.rnn_output
        log_probs = torch.log(rnn_output.clamp(min=1e-8))
        loss = -torch.mean(torch.sum(log_probs * onehot_target, dim=-1))
        
        # Backward pass and optimize
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        
        self.optimizer.step()
        
        return None, hit_np, final_state, final_history_masking_np, sample_ids_np
    
    def inference(self, user_interesting, user_masking, processes_length, init_state=None, s_t=0):
        self.eval()
        
        # Convert inputs to tensors
        user_interesting = torch.tensor(user_interesting, dtype=torch.float, device=self.device)
        user_masking = torch.tensor(user_masking, dtype=torch.float, device=self.device)
        processes_length = torch.tensor([processes_length], dtype=torch.long, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, sample_ids, final_state, final_history_masking, hit = self.forward(
                user_interesting, user_masking, processes_length, init_state, s_t, 1.0)
            
            # Compute rewards
            imediate_reward, cumsum_reward = self.compute_rewards(hit, processes_length)
        
        # Convert to numpy for compatibility
        user_item_probs = logits.rnn_output.cpu().numpy()
        user_selected_items = sample_ids.cpu().numpy()
        user_final_masking = final_history_masking.cpu().numpy()
        user_hit = hit.cpu().numpy()
        user_imediate_reward = imediate_reward.cpu().numpy()
        user_cumsum_reward = cumsum_reward.cpu().numpy()
        
        return (user_item_probs, user_selected_items, user_final_masking,
                user_hit, user_imediate_reward, user_cumsum_reward)


class EMInteractiveModel(nn.Module):
    def __init__(self, rnn_size, layer_size, decoder_vocab_size, embedding_dim, k, lr, device):
        super(EMInteractiveModel, self).__init__()
        
        self.device = device
        self._k = k
        self.lr = lr
        self.postive_imediate_reward = 1.0
        self.negative_imediate_reward = 0.2
        self.account_ratio = 0.9
        self.rnn_size = rnn_size
        
        # Define embeddings
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, embedding_dim)
        
        # Define RNN cell
        self.decoder_cell = self._get_simple_lstm(rnn_size, layer_size)
        
        # Define output layer
        self.fc_layer = nn.Linear(rnn_size, decoder_vocab_size)
        
        # Memory components
        self.mem_output_layer = nn.Linear(decoder_vocab_size, rnn_size, bias=False)
        self.forget_gate_mem_part = nn.Linear(rnn_size, rnn_size, bias=True)
        self.forget_gate_cell_part = nn.Linear(rnn_size, rnn_size, bias=False)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-4)
    
    def _get_simple_lstm(self, rnn_size, layer_size):
        if layer_size == 1:
            return SimpleGRUCell(rnn_size, rnn_size)
        else:
            lstm_layers = [SimpleGRUCell(rnn_size, rnn_size) for _ in range(layer_size)]
            return MultiRNNCell(lstm_layers)
    
    def forward(self, user_interesting, user_masking, mem, sequence_length, init_state=None, s_token=0, s_hit=1.0):
        batch_size = user_interesting.size(0)
        
        # Process memory
        processed_mem = torch.tanh(self.mem_output_layer(mem))
        
        # Initialize state if not provided
        if init_state is None:
            if isinstance(self.decoder_cell, MultiRNNCell):
                init_state = tuple(torch.zeros(batch_size, self.rnn_size, device=self.device) 
                                 for _ in range(len(self.decoder_cell.cells)))
            else:
                init_state = torch.zeros(batch_size, self.rnn_size, device=self.device)
        
        # Create helper and decoder
        helper = InteractiveGreedyEmbeddingHelper(
            self.decoder_embedding, self._k, [s_token], [s_hit], 
            user_interesting.size(-1), sequence_length, self.device)
        
        # Create the output function that applies softmax to fc_layer output
        def output_fn(x):
            return F.softmax(self.fc_layer(x), dim=-1)
        
        decoder = ExternalMemInteractiveDecoder(
            self.decoder_cell, helper, init_state, user_masking, 
            user_interesting, mem, self.rnn_size, output_fn)
        
        # Run dynamic decoding
        outputs, final_state, final_history_masking, hit, final_sequence_lengths = dynamic_interactive_decode(
            decoder, device=self.device, maximum_iterations=sequence_length[0])
        
        # Process outputs
        hit = hit.hit
        logits = outputs
        sample_ids = outputs.sample_id
        
        return logits, sample_ids, final_state, final_history_masking, hit
    
    def compute_rewards(self, hit, sequence_length):
        # Create reversed hit
        reverse_hit = torch.flip(hit, [1])
        
        # Compute immediate rewards
        reverse_imediate_reward = torch.where(
            reverse_hit > 0,
            reverse_hit * self.postive_imediate_reward,
            (reverse_hit - 1) * self.negative_imediate_reward
        )
        
        # Flip back
        imediate_reward = torch.flip(reverse_imediate_reward, [1])
        
        # Compute cumulative rewards
        cumsum_reward = torch.zeros_like(imediate_reward)
        batch_size = hit.size(0)
        
        for b in range(batch_size):
            seq_len = sequence_length[b]
            reward = 0
            for t in range(seq_len - 1, -1, -1):
                reward = reverse_imediate_reward[b, t] + self.account_ratio * reward
                cumsum_reward[b, seq_len - 1 - t] = reward
        
        return imediate_reward, cumsum_reward
    
    def reinforcement_learn(self, user_interesting, user_masking, mem, processes_length, 
                          init_state=None, s_token=0, s_hit=1.0):
        self.train()
        self.optimizer.zero_grad()
        
        # Convert inputs to tensors
        user_interesting = torch.tensor(user_interesting, dtype=torch.float, device=self.device)
        user_masking = torch.tensor(user_masking, dtype=torch.float, device=self.device)
        mem = torch.tensor(mem, dtype=torch.float, device=self.device)
        processes_length = torch.tensor([processes_length], dtype=torch.long, device=self.device)
        
        # Forward pass
        logits, sample_ids, final_state, final_history_masking, hit = self.forward(
            user_interesting, user_masking, mem, processes_length, init_state, s_token, s_hit)
        
        # Compute rewards
        _, cumsum_reward = self.compute_rewards(hit, processes_length)
        
        # Convert to numpy for compatibility
        hit_np = hit.detach().cpu().numpy()
        sample_ids_np = sample_ids.detach().cpu().numpy()
        final_history_masking_np = final_history_masking.detach().cpu().numpy()
        
        # One-hot encoding of sample ids
        onehot_sample = F.one_hot(sample_ids, num_classes=user_interesting.size(-1)).float()
        
        # Calculate policy gradient loss
        rnn_output = logits.rnn_output
        gt_ratio = torch.cumprod(torch.ones_like(cumsum_reward) * self.account_ratio, dim=1)
        
        # Compute reinforcement loss
        log_probs = torch.log(rnn_output.clamp(min=1e-8))
        selected_log_probs = torch.sum(log_probs * onehot_sample, dim=-1)
        loss = -torch.mean(selected_log_probs * cumsum_reward * gt_ratio)
        
        # Backward pass and optimize
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        
        self.optimizer.step()
        
        return loss.item(), None, hit_np, final_state, final_history_masking_np, sample_ids_np
    
    def supervised_learn(self, user_interesting, user_masking, mem, targets, processes_length, 
                       init_state=None, s_token=0, s_hit=1.0):
        self.train()
        self.optimizer.zero_grad()
        
        # Convert inputs to tensors
        user_interesting = torch.tensor(user_interesting, dtype=torch.float, device=self.device)
        user_masking = torch.tensor(user_masking, dtype=torch.float, device=self.device)
        mem = torch.tensor(mem, dtype=torch.float, device=self.device)
        processes_length = torch.tensor([processes_length], dtype=torch.long, device=self.device)
        targets = torch.tensor(targets[:processes_length], dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Forward pass
        logits, sample_ids, final_state, final_history_masking, hit = self.forward(
            user_interesting, user_masking, mem, processes_length, init_state, s_token, s_hit)
        
        # Convert to numpy for compatibility
        hit_np = hit.detach().cpu().numpy()
        sample_ids_np = sample_ids.detach().cpu().numpy()
        final_history_masking_np = final_history_masking.detach().cpu().numpy()
        
        # One-hot encoding of targets
        onehot_target = F.one_hot(targets, num_classes=user_interesting.size(-1)).float()
        
        # Compute supervised loss
        rnn_output = logits.rnn_output
        log_probs = torch.log(rnn_output.clamp(min=1e-8))
        loss = -torch.mean(torch.sum(log_probs * onehot_target, dim=-1))
        
        # Backward pass and optimize
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        
        self.optimizer.step()
        
        return loss.item(), None, hit_np, final_state, final_history_masking_np, sample_ids_np
    
    def inference(self, user_interesting, user_masking, mem, processes_length, init_state=None, s_t=0):
        self.eval()
        
        # Convert inputs to tensors
        user_interesting = torch.tensor(user_interesting, dtype=torch.float, device=self.device)
        user_masking = torch.tensor(user_masking, dtype=torch.float, device=self.device)
        mem = torch.tensor(mem, dtype=torch.float, device=self.device)
        processes_length = torch.tensor([processes_length], dtype=torch.long, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, sample_ids, final_state, final_history_masking, hit = self.forward(
                user_interesting, user_masking, mem, processes_length, init_state, s_t, 1.0)
            
            # Compute rewards
            imediate_reward, cumsum_reward = self.compute_rewards(hit, processes_length)
        
        # Convert to numpy for compatibility
        user_item_probs = logits.rnn_output.cpu().numpy()
        user_selected_items = sample_ids.cpu().numpy()
        user_final_masking = final_history_masking.cpu().numpy()
        user_hit = hit.cpu().numpy()
        user_imediate_reward = imediate_reward.cpu().numpy()
        user_cumsum_reward = cumsum_reward.cpu().numpy()
        
        return (user_item_probs, user_selected_items, user_final_masking,
                user_hit, user_imediate_reward, user_cumsum_reward)


def get_cum_interesting(y, n):
    """Convert history items to one-hot representation of interests"""
    interest = np.zeros((1, n), dtype=np.float32)
    for item in y:
        interest[0, item] = 1.0
    return interest


def get_initial_masking(n):
    """Create initial masking with all items available"""
    return np.ones((1, n), dtype=np.float32)


def get_masking(n, known_items):
    """Create masking with known items excluded"""
    masking = get_initial_masking(n)
    for item in known_items:
        masking[0, item] = 0.0
    return masking