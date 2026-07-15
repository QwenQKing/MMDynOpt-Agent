
import copy
import pickle
import re
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from copy import deepcopy
from typing import Any, List, Union
import os
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tqdm import tqdm
from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils import hf_processor
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _repeat_interleave,
    vLLMRollout,
)
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs
from MMDynOpt.utils.torch_functional import get_final_eos_mask
from MMDynOpt.utils.tools.mm_LM_env import mm_llm_env


def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor: 
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(
            torch.cat([t, torch.tensor([pad_token_id] * (max_len - t.size(0)), device=t.device, dtype=t.dtype)], dim=0)
        )
    return torch.stack(padded_tensor_list, dim=dim)


class vLLMRollout_MultiTurn_MMSearch_R1(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):

        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        self.tokenizer = tokenizer
        self.processor = hf_processor(model_path)


        try:
            with open(config.search.user_prompt_after_image_search, 'rb') as file:
                self.user_prompt_after_image_search = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_after_image_search default to None") 
            self.user_prompt_after_image_search = None
        try:
            with open(config.search.user_prompt_after_text_search, 'rb') as file:
                self.user_prompt_after_text_search = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_after_text_search default to None")
            self.user_prompt_after_text_search = None

        print(f"[Prompt Set] user_prompt_after_text_search: {self.user_prompt_after_text_search}")
        print(f"[Prompt Set] user_prompt_after_image_search: {self.user_prompt_after_image_search}")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        print(f">>> vllm_rollout_spmd Rollout Starts ...")




        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:  
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        batch_size = idx.size(0)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        input_prompt_generation_mask = torch.zeros_like(
            idx, dtype=attention_mask.dtype, device=attention_mask.device
        )

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1,
            }

        n = 1 if prompts.meta_info.get('validate', False) else self.config.n

        vllm_inputs = (
            []
        )
        multi_turn_response_mask = []
        prefix_prompt_lengths = []
        search_tool_return_images = []
        mm_llm_history = [] 
        llm_call_stats = [] 

        if 'multi_modal_data' in non_tensor_batch:
            _multi_modal_data_list = non_tensor_batch['multi_modal_data']
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), _multi_modal_data_list):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    vllm_inputs.append(
                        {
                            'prompt_token_ids': deepcopy(raw_prompt_ids), 
                            'multi_modal_data': deepcopy(multi_modal_data),
                        }
                    )
                    multi_turn_response_mask.append(
                        [
                            torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                        ]
                    )
                    prefix_prompt_lengths.append(prefix_length)
                    search_tool_return_images.append([])
                    mm_llm_history.append([])
                    llm_call_stats.append([])
        else:
            for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids'): 
                prefix_length = len(raw_prompt_ids) 
                for _ in range(n): 
                    vllm_inputs.append( 
                        {
                            'prompt_token_ids': deepcopy(raw_prompt_ids), 
                        }
                    )
                    multi_turn_response_mask.append( 
                        [
                            torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                        ]
                    )
                    prefix_prompt_lengths.append(prefix_length) 
                    search_tool_return_images.append([]) 
                    mm_llm_history.append([]) 
                    llm_call_stats.append([]) 

        if 'image_urls' in non_tensor_batch.keys() and not prompts.meta_info.get('validate', False):
            non_tensor_batch['image_urls'] = _repeat_interleave(non_tensor_batch['image_urls'], self.config.n)

        to_generate = list(range(batch_size * n))
        worker_trajs_count = len(to_generate)
        agent_input_stats = [[] for _ in range(batch_size * n)]
        agent_output_stats = [[] for _ in range(batch_size * n)]

        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            max_iterations = self.config.max_gen_round
            while current_iteration < max_iterations and len(to_generate) > 0:
                _max_model_len = int(
                    self.config.get("max_model_len") or (self.config.prompt_length + self.config.response_length)
                )
                _max_prompt_len = _max_model_len - self.config.response_length
                _image_placeholder_token_id = 151655

                for _i in to_generate:
                    _prompt_ids = vllm_inputs[_i]["prompt_token_ids"]
                    _prefix_len = prefix_prompt_lengths[_i]
                    if len(_prompt_ids) <= _max_prompt_len:
                        continue

                    if _prefix_len >= _max_prompt_len:
                        _new_prompt_ids = _prompt_ids[-_max_prompt_len:]
                        _n_total_images_before = sum(1 for _t in _prompt_ids if _t == _image_placeholder_token_id)
                        _n_total_images_after = sum(1 for _t in _new_prompt_ids if _t == _image_placeholder_token_id)
                        _n_images_dropped = _n_total_images_before - _n_total_images_after

                        vllm_inputs[_i]["prompt_token_ids"] = _new_prompt_ids
                        prefix_prompt_lengths[_i] = 0

                        if _n_images_dropped > 0:
                            if "multi_modal_data" in vllm_inputs[_i] and "image" in vllm_inputs[_i]["multi_modal_data"]:
                                _img_list = vllm_inputs[_i]["multi_modal_data"]["image"]
                                vllm_inputs[_i]["multi_modal_data"]["image"] = _img_list[_n_images_dropped:]
                            if len(search_tool_return_images[_i]) >= _n_images_dropped:
                                search_tool_return_images[_i] = search_tool_return_images[_i][_n_images_dropped:]

                        _all_masks = torch.cat(multi_turn_response_mask[_i], dim=0)
                        _new_all_mask = _all_masks[-_max_prompt_len:]
                        multi_turn_response_mask[_i] = [_new_all_mask.clone()]
                        continue

                    _rest = _prompt_ids[_prefix_len:]
                    _keep_rest = _max_prompt_len - _prefix_len
                    if _keep_rest <= 0:
                        _new_prompt_ids = _prompt_ids[-_max_prompt_len:]
                        _n_total_images_before = sum(1 for _t in _prompt_ids if _t == _image_placeholder_token_id)
                        _n_total_images_after = sum(1 for _t in _new_prompt_ids if _t == _image_placeholder_token_id)
                        _n_images_dropped = _n_total_images_before - _n_total_images_after

                        vllm_inputs[_i]["prompt_token_ids"] = _new_prompt_ids
                        prefix_prompt_lengths[_i] = 0

                        if _n_images_dropped > 0:
                            if "multi_modal_data" in vllm_inputs[_i] and "image" in vllm_inputs[_i]["multi_modal_data"]:
                                _img_list = vllm_inputs[_i]["multi_modal_data"]["image"]
                                vllm_inputs[_i]["multi_modal_data"]["image"] = _img_list[_n_images_dropped:]
                            if len(search_tool_return_images[_i]) >= _n_images_dropped:
                                search_tool_return_images[_i] = search_tool_return_images[_i][_n_images_dropped:]

                        _all_masks = torch.cat(multi_turn_response_mask[_i], dim=0)
                        _new_all_mask = _all_masks[-_max_prompt_len:]
                        multi_turn_response_mask[_i] = [_new_all_mask.clone()]
                        continue

                    _new_rest = _rest[-_keep_rest:]
                    _new_prompt_ids = _prompt_ids[:_prefix_len] + _new_rest
                    vllm_inputs[_i]["prompt_token_ids"] = _new_prompt_ids

                    _dropped = _rest[: len(_rest) - _keep_rest]
                    _n_image_in_dropped = sum(1 for _t in _dropped if _t == _image_placeholder_token_id)
                    _n_image_in_prefix = sum(1 for _t in _prompt_ids[:_prefix_len] if _t == _image_placeholder_token_id)

                    if _n_image_in_dropped > 0:
                        if "multi_modal_data" in vllm_inputs[_i] and "image" in vllm_inputs[_i]["multi_modal_data"]:
                            _img_list = vllm_inputs[_i]["multi_modal_data"]["image"]
                            _start = _n_image_in_prefix
                            _end = _n_image_in_prefix + _n_image_in_dropped
                            vllm_inputs[_i]["multi_modal_data"]["image"] = _img_list[:_start] + _img_list[_end:]

                        if len(search_tool_return_images[_i]) >= _n_image_in_dropped:
                            search_tool_return_images[_i] = search_tool_return_images[_i][_n_image_in_dropped:]

                    _all_masks = torch.cat(multi_turn_response_mask[_i], dim=0)
                    _new_all_mask = torch.cat(
                        [_all_masks[:_prefix_len], _all_masks[-_keep_rest:]],
                        dim=0,
                    )
                    multi_turn_response_mask[_i] = [
                        _new_all_mask[:_prefix_len].clone(),
                        _new_all_mask[_prefix_len:].clone(),
                    ]
                idx_to_gen = []
                for i in to_generate:
                    idx_to_gen.append(vllm_inputs[i])
                    agent_input_stats[i].append(len(vllm_inputs[i]['prompt_token_ids']))

                print(
                    f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ..."
                )

                with self.update_sampling_params(n=1):
                    outputs = self.inference_engine.generate(
                        prompts=idx_to_gen, sampling_params=self.sampling_params, use_tqdm=False
                    )

                response = []
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                        if 151645 not in filtered_token_ids:
                            filtered_token_ids[-1] = 151645

                        response.append(filtered_token_ids)

                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_search_query_mapping = {}
                for i_gen, response_ in zip(to_generate, response):
                    response_ = list(response_)
                    agent_output_stats[i_gen].append(len(response_)) 
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_   
                    multi_turn_response_mask[i_gen].append(
                        torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device) 
                    )
                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                    match = re.search(r'<prompt>(.*?)</prompt>\s*$', decoded_resp_, re.DOTALL)
                    if match:
                        prompt_text = match.group(1).strip()
                        if current_iteration == max_iterations - 1:
                            idx_to_remove.append(i_gen)
                            print(f"{i_gen} has reached max_iterations {max_iterations}")
                            continue
                        assert str(i_gen) not in id_search_query_mapping.keys()
                        id_search_query_mapping[str(i_gen)] = {"prompt": prompt_text}

                    else:
                        idx_to_remove.append(i_gen)

                print(
                    f"[Round #{current_iteration} Rollout Interaction Trigger] For THIS round, we need to conduct reasoning for: {id_search_query_mapping} ..."
                )

                for x in idx_to_remove:
                    to_generate.remove(x)

                print(
                    f"[Round #{current_iteration} Rollout END] For THIS round, We hava completed {len(idx_to_remove)} trajs ..."
                )
                print(
                    f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ..."
                )

                search_result = []

                if not self.config.search.parallel_tool_call:
                    for i_todo in tqdm(to_generate, desc=f"[Round #{current_iteration} Searching Progress]"):
                        tool_returned_images = []
                        assert str(i_todo) in id_search_query_mapping.keys()
                        prompt_text = id_search_query_mapping[str(i_todo)]["prompt"]

                        is_first_call = (len(mm_llm_history[i_todo]) == 0)
                        if is_first_call and "image_urls" in non_tensor_batch:
                            img_raw = non_tensor_batch["image_urls"][i_todo]
                            if img_raw is None:
                                image_urls_this = None
                            elif isinstance(img_raw, (list, np.ndarray)):
                                image_urls_this = list(img_raw) if hasattr(img_raw, "__iter__") and not isinstance(img_raw, str) else [img_raw]
                            else:
                                image_urls_this = [img_raw]
                        else:
                            image_urls_this = None

                        reply_text, new_history, tool_stat = mm_llm_env(
                            prompt=prompt_text,
                            image_urls=image_urls_this,
                            history=mm_llm_history[i_todo],
                        )
                        mm_llm_history[i_todo] = new_history
                        tool_returned_str = reply_text
                        search_result.append((tool_returned_str, tool_returned_images, tool_stat))

                        llm_call_stats[i_todo].append({
                            "prompt_len": len(prompt_text),
                            "response_len": len(reply_text),
                            "prompt_tokens": int(tool_stat.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(tool_stat.get("completion_tokens", 0) or 0),
                        })

                else:
                    def tool_helper(i_todo):
                        tool_returned_images = []
                        assert str(i_todo) in id_search_query_mapping.keys()
                        prompt_text = id_search_query_mapping[str(i_todo)]["prompt"]

                        is_first_call = (len(mm_llm_history[i_todo]) == 0)
                        if is_first_call and "image_urls" in non_tensor_batch:
                            img_raw = non_tensor_batch["image_urls"][i_todo]
                            if img_raw is None:
                                image_urls_this = None
                            elif isinstance(img_raw, (list, np.ndarray)):
                                image_urls_this = list(img_raw) if hasattr(img_raw, "__iter__") and not isinstance(img_raw, str) else [img_raw]
                            else:
                                image_urls_this = [img_raw]
                        else:
                            image_urls_this = None

                        reply_text, new_history, tool_stat = mm_llm_env(
                            prompt=prompt_text,
                            image_urls=image_urls_this,
                            history=mm_llm_history[i_todo],
                        )
                        mm_llm_history[i_todo] = new_history
                        tool_returned_str = reply_text

                        llm_call_stats[i_todo].append({
                            "prompt_len": len(prompt_text),
                            "response_len": len(reply_text),
                            "prompt_tokens": int(tool_stat.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(tool_stat.get("completion_tokens", 0) or 0),
                        })

                        return (tool_returned_str, tool_returned_images, tool_stat)

                    search_call_futures = []
                    with ThreadPoolExecutor(self.config.search.parallel_tool_call_threads) as pool:
                        for i_todo in to_generate:
                            assert str(i_todo) in id_search_query_mapping.keys()
                            search_call_futures.append(pool.submit(tool_helper, i_todo))
                        for _ in tqdm(
                            as_completed(search_call_futures),
                            desc=f"[MT][Round #{current_iteration} Searching Progress]",
                        ):
                            pass
                    search_result = [f.result() for f in search_call_futures]

                to_generate_ = to_generate.copy()
                assert len(to_generate_) == len(
                    search_result
                ), f"Current Itr: {current_iteration} | len(to_generate_): {len(to_generate_)} | len(search_result): {len(search_result)}"
                for i_gen_, search_result_ in zip(to_generate_, search_result):

                    search_result_txt, search_result_img, tool_stat = search_result_
                    response_message = "<response>" + search_result_txt + "</response>"
                    search_result_message = (
                        "<|im_start|>user\n" + response_message + "<|im_end|>\n<|im_start|>assistant\n"
                    )

                    next_turn_prompt_ids = self.tokenizer.encode(search_result_message)

                    vllm_inputs[i_gen_][
                        'prompt_token_ids'
                    ] += next_turn_prompt_ids
                    if search_result_img:
                        vllm_inputs[i_gen_]['multi_modal_data']['image'] += search_result_img
                        search_tool_return_images[
                            i_gen_
                        ] += search_result_img
                    multi_turn_response_mask[i_gen_].append(
                        torch.zeros(len(next_turn_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                    )

                pbar.update(worker_trajs_count - len(to_generate))

                current_iteration += 1

        response = []
        response_generation_mask = []
        for i_ in range(batch_size * n):
            if len(multi_turn_response_mask[i_]) > 1:
                all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
            else:
                all_response_masks = multi_turn_response_mask[i_][0]
            resp_mask_device = all_response_masks.device

            first_round_prompt_length = prefix_prompt_lengths[i_]
            response_after_prompt = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]

            if search_tool_return_images[i_]:
                searched_image_inputs = self.processor.image_processor(
                    search_tool_return_images[i_], return_tensors='pt'
                )
                searched_image_grid_thw = searched_image_inputs['image_grid_thw']
                if searched_image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index, image_pad_token, magic_num = 0, 151655, 654321
                    all_response_masks = all_response_masks.tolist()
                    while image_pad_token in response_after_prompt:
                        pos = response_after_prompt.index(image_pad_token)
                        replicate_count = searched_image_grid_thw[index].prod() // merge_length
                        response_after_prompt[pos : pos + 1] = [magic_num] * replicate_count
                        all_response_masks[pos : pos + 1] = [0] * replicate_count
                        index += 1
                    response_after_prompt = [image_pad_token if x == magic_num else x for x in response_after_prompt]
                    all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=resp_mask_device)

            response_generation_mask.append(all_response_masks)
            all_response = torch.tensor(response_after_prompt, device=idx.device, dtype=idx.dtype)
            response.append(all_response)
            assert (
                response[i_].shape[0] == response_generation_mask[i_].shape[0]
            ), f"shape mismatched | response[i_]: {response[i_].shape[0]} | response_generation_mask[i_]: {response_generation_mask[i_].shape[0]}"
        assert len(response) == len(
            response_generation_mask
        ), "length mismatched between response and response_generation_mask!"

        response = pad_to_max_stack(    
            response, self.pad_token_id, dim=0
        )
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        if response.shape[1] > self.config.response_length_total:
            response = response[:, : self.config.response_length_total]
            response_generation_mask = response_generation_mask[:, : self.config.response_length_total]
        elif response.shape[1] < self.config.response_length_total:
            response = pad_sequence_to_length(response, self.config.response_length_total, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(
                response_generation_mask, self.config.response_length_total, 0
            )

        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            if 'multi_modal_data' in non_tensor_batch.keys():
                repeated = []
                _index_br = 0
                for item in non_tensor_batch['multi_modal_data']:
                    for _ in range(self.config.n):
                        new_item = copy.deepcopy(item)
                        if search_tool_return_images[_index_br]:
                            new_item['image'] += search_tool_return_images[_index_br]
                        repeated.append(new_item)
                        _index_br += 1
                non_tensor_batch['multi_modal_data'] = np.array(repeated)
            input_prompt_generation_mask = _repeat_interleave(
                input_prompt_generation_mask, self.config.n
            )

            for key in list(non_tensor_batch.keys()):
                if key not in ['multi_modal_data', 'image_urls']:
                    if isinstance(non_tensor_batch[key], np.ndarray):
                        non_tensor_batch[key] = _repeat_interleave(non_tensor_batch[key], self.config.n)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_final_eos_mask(
            response_id=response, eos_token=[151645], dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
            },
            batch_size=batch_size,
        )

        if 'extra_info' not in non_tensor_batch:
            non_tensor_batch['extra_info'] = np.array([None] * batch_size, dtype=object)
        elif not isinstance(non_tensor_batch['extra_info'], np.ndarray):
            non_tensor_batch['extra_info'] = np.array([None] * batch_size, dtype=object)

        for i in range(batch_size):
            if non_tensor_batch['extra_info'][i] is None:
                non_tensor_batch['extra_info'][i] = {}
            elif not isinstance(non_tensor_batch['extra_info'][i], dict):
                non_tensor_batch['extra_info'][i] = {}

            stats = llm_call_stats[i]
            if stats:
                total_prompt_len = sum(s['prompt_len'] for s in stats)
                total_response_len = sum(s['response_len'] for s in stats)
                n_llm_calls = len(stats)

                non_tensor_batch['extra_info'][i]['llm_prompt_len'] = total_prompt_len
                non_tensor_batch['extra_info'][i]['llm_response_len'] = total_response_len
                non_tensor_batch['extra_info'][i]['n_llm_calls'] = n_llm_calls
                llm_in_pr = [int(s.get('prompt_tokens', 0)) for s in stats]
                llm_out_pr = [int(s.get('completion_tokens', 0)) for s in stats]
                non_tensor_batch['extra_info'][i]['llm_input_tokens'] = sum(llm_in_pr)
                non_tensor_batch['extra_info'][i]['llm_output_tokens'] = sum(llm_out_pr)
                non_tensor_batch['extra_info'][i]['llm_per_round_input'] = llm_in_pr
                non_tensor_batch['extra_info'][i]['llm_per_round_output'] = llm_out_pr

            a_in_pr = [int(x) for x in agent_input_stats[i]]
            a_out_pr = [int(x) for x in agent_output_stats[i]]
            non_tensor_batch['extra_info'][i]['agent_input_tokens'] = sum(a_in_pr)
            non_tensor_batch['extra_info'][i]['agent_output_tokens_sum'] = sum(a_out_pr)
            non_tensor_batch['extra_info'][i]['agent_per_round_input'] = a_in_pr
            non_tensor_batch['extra_info'][i]['agent_per_round_output'] = a_out_pr
            non_tensor_batch['extra_info'][i]['n_agent_rounds'] = len(a_out_pr)

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        print(f">>> vllm_rollout_spmd Rollout Ends ...")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
