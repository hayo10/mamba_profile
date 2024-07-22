import torch

def beam_search(model, tokenizer, input_ids, beam_size=2, max_length=50):

    finished_beams = []
    _, last_state = model(last_state=None, input_ids=input_ids[:,:-1])
    running_beam = [(0, input_ids[:, -1], last_state)]
    

    
    while len(finished_beams) < beam_size and running_beam:
        beam_score, input_ids, cur_last_state = running_beam.pop(0)
        
        outputs = model(last_state=cur_last_state, input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        new_last_state = outputs.last_state

        # Choose top 2 (beam_size) tokens
        top_k_values, top_k_indices = torch.topk(logits, beam_size, dim=-1)
 

        for i in range(beam_size):
            score = top_k_values[:,i]
            token = top_k_indices[:,i]
            
            # Add the new token and update attention_mask
            new_input_ids = token

            if token == tokenizer.eos_token_id or new_input_ids.shape[1] == max_length+14:
                finished_beams.append((beam_score + score, new_input_ids, new_last_state))
            else:
                running_beam.append((beam_score + score, new_input_ids, new_last_state))
                
        # Sort the running beams by score
        running_beam.sort(key=lambda x: x[0], reverse=True)
    
    # Return the highest scoring finished beam
    return max(finished_beams, key=lambda x: x[0])[1]