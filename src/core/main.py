from __future__ import print_function, division

import collections
import copy
import ctypes
import os
import queue
import random
import signal
import sys
import time
import warnings
import json

import numpy as np
import torch
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
from torch import nn
from typing import Optional

from src.core.experiments.config import get_experiment
from src.core.learning.optimizer import SharedAdam, SharedRMSprop
from src.core.learning.valid import valid
from src.core.learning.train import train
from src.core.learning.test import test
from src.core.utils.arguments import parse_arguments
from src.core.utils.misc import save_project_state_in_log
from src.core.utils.net import (
    ScalarMeanTracker,
    load_model_from_state_dict,
    TensorConcatTracker,
)
from src.core.utils.constants import PROJECT_ROOT_DIR

np.set_printoptions(threshold=10000000, suppress=True, linewidth=100000)

os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":

    metrics_to_record = {
        'train': {
            'ep_length': [],
            'accuracy': [],
            'pickupable_but_not_picked': [],
            'picked_but_not_pickupable': [],
            'reward': [],
            'dist_to_low_rank': [],
            'invalid_prob_mass': []
        },
        'valid': {
            'ep_length': [],
            'accuracy': [],
            'pickupable_but_not_picked': [],
            'picked_but_not_pickupable': [],
            'reward': []
        },
        'test': {
            'accuracy': [],
            'dist_to_low_rank': [],
            'ep_length': [],
            'final_manhattan_distance_from_target': [],
            'initial_manhattan_steps': [],
            'invalid_prob_mass': [],
            'loss': [],
            'n_frames': [],
            'picked_but_not_pickupable_distance': [],
            'picked_but_not_pickupable': [],
            'picked_but_not_pickupable_visibility': [],
            'pickupable_but_not_picked': [],
            'reward': [],
            'spl_manhattan': []
        }
    }
    max_accuracy = 0
    save_best = False
    args = parse_arguments()
    with open(os.path.join(PROJECT_ROOT_DIR, 'resources/project/runtime_args.properties'), 'w+', encoding='utf-8') as f:
        f.write('%s=%s\n' % ('runtime_configurations', 'resources/project/runtime_args.properties'))
        for key, value in vars(args).items():
            f.write('%s=%s\n' % (key, value))

    experiment = get_experiment()
    experiment.init_train_agent.env_args = args
    experiment.init_valid_agent.env_args = args
    experiment.init_test_agent.env_args = args
    experiment.checkpoints_dir = args.checkpoints_dir
    experiment.use_checkpoint = args.use_checkpoint

    start_time = time.time()
    local_start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))

    if args.enable_logging:
        # Caching current state of the project
        log_file_path = save_project_state_in_log(sys.argv,
            local_start_time_str,
            experiment.checkpoints_dir,
            experiment.use_checkpoint,
            args.log_dir
        )
        # Create a tensorboard logger
        log_writer = SummaryWriter(log_file_path)

        for arg in vars(args):
            log_writer.add_text("call/" + arg, str(getattr(args, arg)), 0)
        s = ""
        for arg in vars(args):
            s += "--" + arg + " "
            s += str(getattr(args, arg)) + " "
        log_writer.add_text("call/full", s, 0)
        log_writer.add_text("log-path/", log_file_path, 0)
        if experiment.checkpoints_dir is not None:
            log_writer.add_text("model-load-path", experiment.checkpoints_dir, 0)

    # Seed (hopefully) all sources of randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
    mp = mp.get_context("spawn")

    if any(gpu_id >= 0 for gpu_id in args.gpu_ids):
        assert torch.cuda.is_available(), (
            f"You have specified gpu_ids=={args.gpu_ids} but no GPUs are available."
            " Please check that your machine has GPUs installed and that the correct (GPU compatible)"
            " version of torch has been installed."
        )

    shared_model: nn.Module = experiment.create_model()

    optimizer_state = None
    restarted_from_episode = None
    if experiment.use_checkpoint != '':
        path = os.path.join(experiment.checkpoints_dir, experiment.use_checkpoint) 
        saved_state = torch.load(path, map_location=lambda storage, loc: storage)
        print("Loading pretrained weights from {}...".format(path))
        if "model_state" in saved_state:
            load_model_from_state_dict(shared_model, saved_state["model_state"])
            optimizer_state = saved_state["optimizer_state"]
            restarted_from_episode = saved_state["episodes"]
        else:
            load_model_from_state_dict(shared_model, saved_state)
        print("Done.")

    shared_model.share_memory()
    pytorch_total_params = sum(p.numel() for p in shared_model.parameters())
    pytorch_total_trainable_params = sum(
        p.numel() for p in shared_model.parameters() if p.requires_grad
    )
    print("pytorch_total_params:" + str(pytorch_total_params))
    print("pytorch_total_trainable_params:" + str(pytorch_total_trainable_params))

    if not args.shared_optimizer:
        raise NotImplementedError("Must use shared optimizer.")

    optimizer: Optional[torch.optim.Optimizer] = None
    if args.shared_optimizer:
        if args.optimizer == "RMSprop":
            optimizer = SharedRMSprop(
                filter(lambda param: param.requires_grad, shared_model.parameters()),
                lr=args.lr,
                saved_state=optimizer_state,
            )
        elif args.optimizer == "Adam":
            optimizer = SharedAdam(
                filter(lambda param: param.requires_grad, shared_model.parameters()),
                lr=args.lr,
                amsgrad=args.amsgrad,
                saved_state=optimizer_state,
            )
        else:
            raise NotImplementedError(
                "Must choose a shared optimizer from 'RMSprop' or 'Adam'."
            )

    processes = []

    if args.enable_test_agent:
        average_metrics = {
            'accuracy': [],
            'ep_length': [],
            'pickupable_but_not_picked': [],
            'picked_but_not_pickupable': [],
            'reward': []
        }

        # Creating the basic test configuration
        end_flag = mp.Value(ctypes.c_bool, False)
        test_res_queue = mp.Queue()
        test_total_ep = mp.Value(ctypes.c_int32, 0)
        save_data_queue = None if not args.save_extra_data else mp.Queue()

        # Creating the test experiment and the testing process
        test_experiment = copy.deepcopy(experiment)
        p = mp.Process(
            target=test,
            args=(args, shared_model, test_experiment, test_res_queue, end_flag, 0, 1),
        )

        # Starting with the testing scenarios
        p.start()
        processes.append(p)
        time.sleep(0.2)

        # TODO: Test Metrics Logging
        test_thin = 1
        n_frames = 0
        try:
            while (
                (not experiment.stopping_criteria_reached())
                and any(p.is_alive() for p in processes)
                and test_total_ep.value < args.max_ep
            ):
                try:
                    while not test_res_queue.empty():
                        test_result = test_res_queue.get()
                        if len(test_result) == 0:
                            continue
                        ep_length = sum(
                            test_result[k] for k in test_result if "ep_length" in k
                        )
                        n_frames += ep_length

                        if test_total_ep.value % args.save_freq == 0:
                            metrics_to_record['test']['ep_length'].append([test_total_ep.value, ep_length])
                            average_metrics['ep_length'].append(ep_length)
                            metrics_to_record['test']['accuracy'].append(
                                [test_total_ep.value, test_result['accuracy']])
                            average_metrics['accuracy'].append(test_result['accuracy'])
                            metrics_to_record['test']['pickupable_but_not_picked'].append(
                                [test_total_ep.value, test_result['pickupable_but_not_picked']])
                            average_metrics['pickupable_but_not_picked'].append(test_result['pickupable_but_not_picked'])
                            metrics_to_record['test']['picked_but_not_pickupable'].append(
                                [test_total_ep.value, test_result['picked_but_not_pickupable']])
                            average_metrics['picked_but_not_pickupable'].append(test_result['picked_but_not_pickupable'])
                            metrics_to_record['test']['reward'].append([test_total_ep.value, test_result['reward']])
                            average_metrics['reward'].append(test_result['reward'])

                        key = list(test_result.keys())[0].split("/")[0]
                        test_total_ep.value += 1
                        if test_total_ep.value % test_thin == 0:
                            for k in test_result:
                                if np.isscalar(test_result[k]):
                                    log_writer.add_scalar(
                                        k + "/test",
                                        np.mean(average_metrics[k]),
                                        test_total_ep.value,
                                    )
                                elif isinstance(test_result[k], collections.Iterable):
                                    log_writer.add_histogram(
                                        k + "/test",
                                        np.mean(average_metrics[k]),
                                        test_total_ep.value,
                                    )
                    # Saving extra data (if any)
                    if save_data_queue is not None:
                        while True:
                            try:
                                experiment.save_episode_summary(
                                    data_to_save=save_data_queue.get(timeout=0.2)
                                )
                            except queue.Empty as _:
                                break
                except queue.Empty as _:
                    pass
        finally:
            end_flag.value = True
            print(
                "Stopping criteria reached: {}".format(
                    experiment.stopping_criteria_reached()
                ),
                flush=True,
            )
            print(
                "Any workers still alive: {}".format(any(p.is_alive() for p in processes)),
                flush=True,
            )

            with open(os.path.join(PROJECT_ROOT_DIR,
                                   'src/core/output/json_logs/test_metrics_as_json_' + str(time.time()) + '.json'), 'w+',
                      encoding='UTF-8') as f:
                f.write(json.dumps(metrics_to_record))

            if args.enable_logging:
                log_writer.close()
            for p in processes:
                p.join(0.1)
                if p.is_alive():
                    os.kill(p.pid, signal.SIGTERM)

            end_time = time.time()
            local_end_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(end_time))
            print(local_start_time_str + " > " + local_end_time_str)
            print("All done.", flush=True)

    else:
        end_flag = mp.Value(ctypes.c_bool, False)
        train_scalars = ScalarMeanTracker()
        train_tensors = TensorConcatTracker()
        train_total_ep = mp.Value(
            ctypes.c_int32, restarted_from_episode if restarted_from_episode else 0
        )

        train_res_queue = mp.Queue()

        save_data_queue = None if not args.save_extra_data else mp.Queue()
        episode_init_queue = (
            None
            if not args.use_episode_init_queue
            else experiment.create_episode_init_queue(mp_module=mp)
        )

        if experiment.stopping_criteria_reached():
            warnings.warn("Stopping criteria reached before any computations started!")
            print("All done.")
            sys.exit()

        for rank in range(0, args.workers):
            train_experiment = copy.deepcopy(experiment)                            # each worker get's his own experiment
            train_experiment.init_train_agent.seed = random.randint(0, 10 ** 10)
            p = mp.Process(
                target=train,
                args=(
                    rank,
                    args,
                    shared_model,
                    train_experiment,
                    optimizer,
                    train_res_queue,
                    end_flag,
                    train_total_ep,
                    None,  # Update lock
                    save_data_queue,
                    episode_init_queue,
                ),
            )
            p.start()
            processes.append(p)
            time.sleep(0.2)

        time.sleep(5)

        valid_res_queue = mp.Queue()
        valid_total_ep = mp.Value(
            ctypes.c_int32, restarted_from_episode if restarted_from_episode else 0
        )
        if args.enable_val_agent:
            valid_experiment = copy.deepcopy(experiment)
            p = mp.Process(
                target=valid,
                args=(args, shared_model, valid_experiment, valid_res_queue, end_flag, 0, 1),
            )
            p.start()
            processes.append(p)
            time.sleep(0.2)

        time.sleep(1)

        train_thin = 500
        valid_thin = 1
        n_frames = 0
        try:
            while (
                (not experiment.stopping_criteria_reached())
                and any(p.is_alive() for p in processes)
                and train_total_ep.value < args.max_ep
            ):
                try:
                    train_result = train_res_queue.get(timeout=10)
                    if len(train_result) != 0:
                        train_scalars.add_scalars(train_result)
                        train_tensors.add_tensors(train_result)
                        ep_length = sum(
                            train_result[k] for k in train_result if "ep_length" in k
                        )
                        train_total_ep.value += 1
                        n_frames += ep_length

                        # Saving metrics to json file for analysis purpose
                        if train_total_ep.value % args.save_freq == 0:
                            metrics_to_record['train']['ep_length'].append([train_total_ep.value, ep_length])
                            metrics_to_record['train']['accuracy'].append([train_total_ep.value, train_result['accuracy']])
                            metrics_to_record['train']['pickupable_but_not_picked'].append(
                                [train_total_ep.value, train_result['pickupable_but_not_picked']])
                            metrics_to_record['train']['picked_but_not_pickupable'].append(
                                [train_total_ep.value, train_result['picked_but_not_pickupable']])
                            metrics_to_record['train']['reward'].append([train_total_ep.value, train_result['reward']])
                            metrics_to_record['train']['dist_to_low_rank'].append(
                                [train_total_ep.value, train_result['dist_to_low_rank']])
                            metrics_to_record['train']['invalid_prob_mass'].append(
                                [train_total_ep.value, train_result['invalid_prob_mass']])

                        if train_total_ep.value > 10000:
                            old_max_accuracy = max_accuracy
                            max_accuracy = max(max_accuracy, train_result['accuracy'])
                            if old_max_accuracy != max_accuracy:
                                save_best = True

                        if args.enable_logging and train_total_ep.value % train_thin == 0:
                            tracked_means = train_scalars.pop_and_reset()
                            for k in tracked_means:
                                log_writer.add_scalar(
                                    k + "/train", tracked_means[k], train_total_ep.value
                                )
                            if train_total_ep.value % (20 * train_thin) == 0:
                                tracked_tensors = train_tensors.pop_and_reset()
                                for k in tracked_tensors:
                                    log_writer.add_histogram(
                                        k + "/train",
                                        tracked_tensors[k],
                                        train_total_ep.value,
                                    )

                        if args.enable_logging and train_total_ep.value % (10 * train_thin):
                            log_writer.add_scalar(
                                "n_frames", n_frames, train_total_ep.value
                            )

                    if args.enable_logging and args.enable_val_agent:
                        while not valid_res_queue.empty():
                            valid_result = valid_res_queue.get()
                            if len(valid_result) == 0:
                                continue

                            if valid_total_ep.value % args.save_freq == 0:
                                metrics_to_record['valid']['ep_length'].append([valid_total_ep.value, ep_length])
                                metrics_to_record['valid']['accuracy'].append(
                                    [valid_total_ep.value, valid_result['accuracy']])
                                metrics_to_record['valid']['pickupable_but_not_picked'].append(
                                    [valid_total_ep.value, valid_result['pickupable_but_not_picked']])
                                metrics_to_record['valid']['picked_but_not_pickupable'].append(
                                    [valid_total_ep.value, valid_result['picked_but_not_pickupable']])
                                metrics_to_record['valid']['reward'].append([valid_total_ep.value, valid_result['reward']])

                            key = list(valid_result.keys())[0].split("/")[0]
                            valid_total_ep.value += 1
                            if valid_total_ep.value % valid_thin == 0:
                                for k in valid_result:
                                    if np.isscalar(valid_result[k]):
                                        log_writer.add_scalar(
                                            k + "/valid",
                                            valid_result[k],
                                            train_total_ep.value,
                                        )
                                    elif isinstance(valid_result[k], collections.Iterable):
                                        log_writer.add_histogram(
                                            k + "/train",
                                            valid_result[k],
                                            train_total_ep.value,
                                        )

                    # Saving extra data (if any)
                    if save_data_queue is not None:
                        while True:
                            try:
                                experiment.save_episode_summary(
                                    data_to_save=save_data_queue.get(timeout=0.2)
                                )
                            except queue.Empty as _:
                                break

                    # Checkpoints
                    if (
                        train_total_ep.value == args.max_ep
                        or (train_total_ep.value % args.save_freq) == 0 or save_best
                    ):
                        if not os.path.exists(args.checkpoints_dir):
                            os.makedirs(args.checkpoints_dir, exist_ok=True)

                        state_to_save = shared_model.state_dict()

                        if save_best:
                            save_path = os.path.join(
                                args.checkpoints_dir,
                                "best_{}_{}.dat".format(
                                    train_total_ep.value, local_start_time_str
                                ),
                            )
                            save_best = False
                        else:
                            save_path = os.path.join(
                                args.checkpoints_dir,
                                "{}_{}.dat".format(
                                    train_total_ep.value, local_start_time_str
                                ),
                            )
                        torch.save(
                            {
                                "model_state": shared_model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "episodes": train_total_ep.value,
                            },
                            save_path,
                        )
                except queue.Empty as _:
                    pass
        finally:
            end_flag.value = True
            print(
                "Stopping criteria reached: {}".format(
                    experiment.stopping_criteria_reached()
                ),
                flush=True,
            )
            print(
                "Any workers still alive: {}".format(any(p.is_alive() for p in processes)),
                flush=True,
            )
            print(
                "Reached max episodes: {}".format(train_total_ep.value >= args.max_ep),
                flush=True,
            )

            with open(os.path.join(PROJECT_ROOT_DIR,
                                   'src/core/output/json_logs/train_valid_metrics_as_json_' + str(time.time()) + '.json'), 'w+',
                      encoding='UTF-8') as f:
                f.write(json.dumps(metrics_to_record))

            if args.enable_logging:
                log_writer.close()
            for p in processes:
                p.join(0.1)
                if p.is_alive():
                    os.kill(p.pid, signal.SIGTERM)

            end_time = time.time()
            local_end_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(end_time))
            print(local_start_time_str + " > " + local_end_time_str)
            print("All done.", flush=True)
