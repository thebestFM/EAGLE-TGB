import math
import os
import pickle
import sys
import time
import warnings
import torch
import numba as nb
import numpy as np
from pathlib import Path
from numba import typed
from numba.experimental import jitclass
from utils.args_loader import load_args
from utils.data_processing import get_data_tgb
from utils.model import Mixer_per_node
from utils.util import EarlyStopMonitor, NegEdgeSampler, tppr_node_finder, compute_metrics, set_random_seed, move_to_cpu

warnings.filterwarnings("ignore")

args = load_args()

device = torch.device(f"cuda:{args.gpu}")
set_random_seed(args.seed)
NUM_NEG_TRAIN = 1
k_list = [10]

# Settings from TGB
if args.dataset_name == "tgbl-wiki":
    NUM_NEG_VAL = 999
    NUM_NEG_TEST = 999
elif args.dataset_name == "tgbl-review":
    NUM_NEG_VAL = 100
    NUM_NEG_TEST = 100
elif args.dataset_name == "tgbl-coin" or args.dataset_name == "tgbl-comment" or args.dataset_name == "tgbl-flight":
    NUM_NEG_VAL = 20
    NUM_NEG_TEST = 20


full_data, train_data, val_data, test_data, n_nodes, n_edges, metric, tgb_eval_neg_sampler = get_data_tgb(args.dataset_name, root="data", use_validation=True)

n_train = train_data.n_interactions
n_val = val_data.n_interactions
n_test = test_data.n_interactions
print(f"#Edge: train {n_train}, val {n_val}, test {n_test}")

train_neg_edge_sampler = NegEdgeSampler(destinations=train_data.destinations, full_destinations=train_data.destinations, num_neg=NUM_NEG_TRAIN, device=device, seed=args.seed)

def load_neg_samples(mode, batch_idx, batch_size, num_instance, data, neg_edge_sampler, neg_filepath):
    start_idx = batch_idx * batch_size
    end_idx = min(num_instance, start_idx + batch_size)
    sample_idxs = np.array(list(range(start_idx, end_idx)))

    if mode == "Train":
        destinations_batch = data.destinations[sample_idxs]
        negatives_batch = neg_edge_sampler.sample(destinations_batch)
    else: # tgb-prepared neg_edge_sampler for Val and Test
        sources_batch = data.sources[sample_idxs]
        destinations_batch = data.destinations[sample_idxs]
        timestamps_batch = data.timestamps[sample_idxs]
        
        negatives_batch_list = neg_edge_sampler.query_batch(sources_batch, destinations_batch, timestamps_batch, split_mode=mode.lower())

        # Some rows lack samples in tgb's sampler
        max_len = max([len(row) for row in negatives_batch_list])
        padded_negatives_batch_list = []
        for row in negatives_batch_list:
            if len(row) < max_len:
                repeat_count = max_len - len(row)
                repeat_items = (row * ((repeat_count // len(row)) + 1))[:repeat_count]
                new_row = row + repeat_items
            else:
                new_row = row
            padded_negatives_batch_list.append(new_row)

        negatives_batch = torch.tensor(padded_negatives_batch_list, device='cpu') # [num_edges, num_neg]

    negatives_batch = move_to_cpu(negatives_batch)

    os.makedirs(os.path.dirname(neg_filepath), exist_ok=True)
    with open(neg_filepath, "wb") as f:
        pickle.dump(negatives_batch, f)
    
    return negatives_batch


# EAGLE-Structure
def cal_tppr_stats(mode, tppr_finder, data, neg_sampler, filepath, DATA, bs, num_neg):
    sources_all = data.sources
    destinations_all = data.destinations
    timestamps_all = data.timestamps

    num_instance = data.n_interactions
    num_batch = math.ceil(num_instance / bs)

    negatives_all = None
    torch.cuda.reset_max_memory_allocated()

    for batch_idx in range(0, num_batch):
        neg_filepath = (
            f"data/batchneg/{DATA}/{mode}_neg{num_neg}_bs{bs}_batch{batch_idx}.pkl"
        )
        if os.path.exists(neg_filepath):
            with open(neg_filepath, "rb") as f:
                negatives_batch = pickle.load(f)
        else:
            negatives_batch = load_neg_samples(mode, batch_idx, bs, num_instance, data, neg_sampler, neg_filepath)
        
        if negatives_all is None:
            negatives_all = negatives_batch
        else:
            negatives_all = torch.cat((negatives_all, negatives_batch), dim=0)


    negatives_all = negatives_all.t().flatten() # arr[dst(1)_neg1, dst(2)_neg1, ..., dst(n_edge)_neg1, dst(1)_neg2, ...]
    
    if isinstance(negatives_all, torch.Tensor):
        negatives_all = negatives_all.cpu().numpy()

    source_nodes = np.concatenate(
        [sources_all, destinations_all, negatives_all], dtype=np.int32
    )

    t1 = time.time()
    scores = tppr_finder.precompute_link_prediction(
        source_nodes, num_neg
    ) # concat[pos_score*SIZE, neg_score_1*SIZE, ..., neg_score_num_neg*SIZE]
    t_cal_tppr_score = time.time() - t1
    allocated_memory = torch.cuda.max_memory_allocated() / (1024**2) # /MB

    noise4zeros = np.random.uniform(0, 1e-8, scores.shape)
    scores[scores == 0.0] += noise4zeros[scores == 0.0]

    data = source_nodes, timestamps_all, scores, t_cal_tppr_score, allocated_memory

    data = move_to_cpu(data)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"Structure-{mode} scores have been saved at {filepath}")

    return data, t_cal_tppr_score, allocated_memory


def get_cached_tppr_status(finder, DATA, train_data, val_data, test_data, train_neg_sampler, eval_neg_sampler, filename):
    train_file = os.path.join(f"structure_score_cache/{DATA}/train_{filename}")
    if not os.path.exists(train_file):
        train_stats, t_cal_train_tppr, mem_train = cal_tppr_stats(
            "Train",
            finder,
            train_data,
            train_neg_sampler,
            train_file,
            DATA,
            args.batch_size,
            num_neg=NUM_NEG_TRAIN
        )
    else:
        f = open(train_file, "rb")
        train_stats = pickle.load(f)
        t_cal_train_tppr = train_stats[3]
        mem_train = train_stats[4]
        print(f"Loading Structure-Train scores from {train_file}")

    val_file = os.path.join(f"structure_score_cache/{DATA}/val_{filename}")
    if not os.path.exists(val_file):
        val_stats, t_cal_val_tppr, mem_val = cal_tppr_stats(
            "Val",
            finder,
            val_data,
            eval_neg_sampler,
            val_file,
            DATA,
            args.batch_size,
            num_neg=NUM_NEG_VAL
        )
    else:
        f = open(val_file, "rb")
        val_stats = pickle.load(f)
        t_cal_val_tppr = val_stats[3]
        mem_val = val_stats[4]
        print(f"Loading Structure-Val scores from {val_file}")

    test_file = os.path.join(f"structure_score_cache/{DATA}/test_{filename}")
    if not os.path.exists(test_file):
        test_stats, t_cal_test_tppr, mem_test = cal_tppr_stats(
            "Test",
            finder,
            test_data,
            eval_neg_sampler,
            test_file,
            DATA,
            args.batch_size,
            num_neg=NUM_NEG_TEST
        )
    else:
        f = open(test_file, "rb")
        test_stats = pickle.load(f)
        t_cal_test_tppr = test_stats[3]
        mem_test = test_stats[4]
        print(f"Loading Structure-Test scores from {test_file}")

    return val_stats, test_stats, t_cal_train_tppr, t_cal_val_tppr, t_cal_test_tppr, mem_train, mem_val, mem_test


def get_scores(data, tppr_stats, cached_neg_samples):
    tppr_scores = tppr_stats[2]
    num_instance = data.n_interactions
    sample_inds = np.array(list(range(0, num_instance)))

    neg_sample_inds = np.concatenate(
        [sample_inds + i * num_instance for i in range(1, 1 + cached_neg_samples)]
    )
    pos_score_structure = tppr_scores[sample_inds]
    neg_score_structure = tppr_scores[neg_sample_inds]
    pos_score_structure = (
        torch.from_numpy(pos_score_structure).unsqueeze(-1).type(torch.float)
    )
    neg_score_structure = (
        torch.from_numpy(neg_score_structure).unsqueeze(-1).type(torch.float)
    )

    return pos_score_structure, neg_score_structure


def train_structure(structure_saved_name):
    tppr_finder = tppr_node_finder(n_nodes + 1, args.topk_struct, args.alpha, args.beta)
    tppr_finder.reset_tppr()

    val_stats, test_stats, t_train, t_val, t_test, mem_train, mem_val, mem_test = get_cached_tppr_status(
        tppr_finder,
        args.dataset_name,
        train_data,
        val_data,
        test_data,
        train_neg_edge_sampler,
        tgb_eval_neg_sampler,
        structure_saved_name,
    )

    with torch.no_grad():
        pos_score_val, neg_score_val = get_scores(
            val_data, val_stats, cached_neg_samples=NUM_NEG_VAL
        )
        with torch.no_grad():
            val_ap, val_mrr, val_hr_list = compute_metrics(
                pos_score_val, neg_score_val, device, k_list=k_list
            )
        print(f"Structure-Val: ap = {val_ap:.4f}, mrr = {val_mrr:.4f}, "
              + ", ".join([f"hr@{k} = {hr:.4f}" for k, hr in zip(k_list, val_hr_list)]))
        sys.stdout.flush()

        pos_score_test, neg_score_test = get_scores(
            test_data, test_stats, cached_neg_samples=NUM_NEG_TEST
        )
        
        test_ap, test_mrr, test_hr_list = compute_metrics(
            pos_score_test, neg_score_test, device, k_list=k_list
        )

        print(f"Structure-Test: ap = {test_ap:.4f}, mrr = {test_mrr:.4f}, "
              + ", ".join([f"hr@{k} = {hr:.4f}" for k, hr in zip(k_list, test_hr_list)]))
        
        print(f"train_time: {t_train:4f}, val_time: {t_val:4f}, test_time: {t_test:4f}, memory_train: {mem_train}, memory_val: {mem_val}, memory_test: {mem_test}\n")
            
        sys.stdout.flush()

    return val_ap, val_mrr, val_hr_list, test_ap, test_mrr, test_hr_list


# EAGLE-Time
l_int = typed.List()
l_float = typed.List()
a_int = np.array([1, 2], dtype=np.int32)
a_float = np.array([1, 2], dtype=np.float64)
l_int.append(a_int)
l_float.append(a_float)
spec = [
    ("node_to_neighbors", nb.typeof(l_int)),
    ("node_to_edge_idxs", nb.typeof(l_int)),
    ("node_to_edge_timestamps", nb.typeof(l_float)),
]

@jitclass(spec)
class NeighborFinder:
    def __init__(self, node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps):
        self.node_to_neighbors = node_to_neighbors
        self.node_to_edge_idxs = node_to_edge_idxs
        self.node_to_edge_timestamps = node_to_edge_timestamps

    def find_before(self, src_idx, cut_time):
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return (
            self.node_to_neighbors[src_idx][:i],
            self.node_to_edge_idxs[src_idx][:i],
            self.node_to_edge_timestamps[src_idx][:i],
        )

    def get_clean_delta_times(self, source_nodes, timestamps, n_neighbors, topk_sample="last"):
        if topk_sample not in ["last", "early", "random"]:
            raise ValueError("TopK sample strategy must be in [last, early, random]")

        if topk_sample == "random":
            np.random.seed(2024)

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        delta_times = np.zeros(len(source_nodes) * tmp_n_neighbors, dtype=np.float32)
        n_edges = np.zeros(len(source_nodes), dtype=np.int32)
        cum_sum = 0
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            _, _, edge_times = self.find_before(source_node, timestamp)
            n_ngh = len(edge_times)
            if n_ngh > 0:
                if topk_sample == "last":
                    selected_times = edge_times[-n_neighbors:][
                        ::-1
                    ] # delta time from last to early
                elif topk_sample == "early":
                    selected_times = edge_times[
                        :n_neighbors
                    ] # delta time from early to last
                elif topk_sample == "random":
                    if n_ngh <= n_neighbors:
                        selected_times = edge_times
                    else:
                        selected_indices = np.random.choice(
                            n_ngh, n_neighbors, replace=False
                        )
                        selected_times = edge_times[selected_indices]
                        selected_times = np.sort(
                            selected_times
                        ) # delta time from early to last

                n_ngh = len(selected_times)
                delta_times[cum_sum : cum_sum + n_ngh] = timestamp - selected_times

            n_edges[i] = n_ngh
            cum_sum += n_ngh
        return delta_times, n_edges, cum_sum


def get_neighbor_finder(data):
    max_node_idx = max(data.sources.max(), data.destinations.max())
    adj_list = [[] for _ in range(max_node_idx + 1)]

    for source, destination, edge_idx, timestamp in zip(
        data.sources, data.destinations, data.edge_idxs, data.timestamps
    ):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    node_to_neighbors = typed.List()
    node_to_edge_idxs = typed.List()
    node_to_edge_timestamps = typed.List()

    for neighbors in adj_list:
        sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
        node_to_neighbors.append(
            np.array([x[0] for x in sorted_neighbors], dtype=np.int32)
        )
        node_to_edge_idxs.append(
            np.array([x[1] for x in sorted_neighbors], dtype=np.int32)
        )
        node_to_edge_timestamps.append(
            np.array([x[2] for x in sorted_neighbors], dtype=np.float64)
        )

    return NeighborFinder(node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps)


def process_time_data(mode, finder, data, bs, num_neg, neg_sampler, filepath=None):
    print(f"Processing {mode} data")
    num_instance = data.n_interactions
    num_batch = math.ceil(num_instance / bs)

    delta_times_list = []
    all_inds_list = []
    batch_size_list = []

    for batch_idx in range(0, num_batch):
        start_idx = batch_idx * bs
        end_idx = min(num_instance, start_idx + bs)
        sample_inds = np.array(list(range(start_idx, end_idx)))

        sources_batch = data.sources[sample_inds]
        destinations_batch = data.destinations[sample_inds]
        timestamps_batch = data.timestamps[sample_inds]

        neg_filepath = f"data/batchneg/{args.dataset_name}/{mode}_neg{num_neg}_bs{bs}_batch{batch_idx}.pkl"

        if os.path.exists(neg_filepath):
            with open(neg_filepath, "rb") as f:
                negatives_batch = (pickle.load(f)).t().flatten()
        else:
            negatives_batch = load_neg_samples(mode, batch_idx, bs, num_instance, data, neg_sampler, neg_filepath)

        if isinstance(negatives_batch, torch.Tensor):
            negatives_batch = negatives_batch.cpu().numpy()

        source_nodes = np.concatenate(
            [sources_batch, destinations_batch, negatives_batch], dtype=np.int32
        )
        timestamps = np.tile(timestamps_batch, num_neg + 2)

        delta_times, n_neighbors, total_edges = finder.get_clean_delta_times(
            source_nodes, timestamps, args.topk_time, args.time_sample_strategy
        )
        delta_times = delta_times[:total_edges]
        delta_times = torch.from_numpy(delta_times).to(device).unsqueeze(-1)

        all_inds = []
        for i, n_ngh in enumerate(n_neighbors):
            all_inds.extend([(args.topk_time * i + j) for j in range(n_ngh)])

        all_inds = torch.tensor(all_inds, device=device)
        batch_size = len(n_neighbors)

        delta_times = move_to_cpu(delta_times)
        all_inds = move_to_cpu(all_inds)

        delta_times_list.append(delta_times)
        all_inds_list.append(all_inds)
        batch_size_list.append(batch_size)

    if filepath:
        with open(filepath, "wb") as f:
            pickle.dump(
                (num_batch, delta_times_list, all_inds_list, batch_size_list), f
            )
    print(f"Processed time data has been saved at {filepath}")

    return num_batch, delta_times_list, all_inds_list, batch_size_list


def load_precessed_time_data(mode, finder, data, batch_size, num_neg, neg_sampler):
    filepath = f"time_processed_data/{args.dataset_name}/{mode}_bs_{args.batch_size}_topk_{args.topk_time}_sample_{args.time_sample_strategy}_seed_{args.seed}.pkl"
    
    if filepath and os.path.exists(filepath):
        print(f"Loading cached Time-{mode} data from {filepath}")
        with open(filepath, "rb") as f:
            num_batch, delta_times_list, all_inds_list, batch_size_list = pickle.load(f)
    else:
        num_batch, delta_times_list, all_inds_list, batch_size_list = process_time_data(
            mode, finder, data, batch_size, num_neg, neg_sampler, filepath
        )

    return num_batch, delta_times_list, all_inds_list, batch_size_list


def time_forward(model, mode, epoch, optimizer, criterion, num_neg, num_batch, features_list=None, delta_times_list=None, all_inds_list=None, batch_size_list=None):
    if mode == "Train":
        model = model.train()
    else:
        model = model.eval()

    ap_list, mrr_list, hit_list = [], [], []
    t_epo = 0.0
    allocated_memory = 0.0

    all_pos_score = []
    all_neg_score = []

    for batch_idx in range(num_batch):
        t1 = time.time()
        torch.cuda.reset_max_memory_allocated(device)
        no_neighbor_flag = False

        if delta_times_list[batch_idx].numel() == 0:
            no_neighbor_flag = True

            num_pos_sc = batch_size_list[batch_idx] // (num_neg + 2)
            num_neg_sc = num_pos_sc * num_neg
            pos_score = torch.zeros(num_pos_sc, 1)
            neg_score = torch.zeros(num_neg_sc, 1)
        else:
            pos_score, neg_score = model(
                delta_times_list[batch_idx].to(device),
                all_inds_list[batch_idx].to(device),
                batch_size_list[batch_idx],
                num_neg,
            )

        mem = torch.cuda.max_memory_allocated(device) / (1024**2) # /MB
        allocated_memory = max(allocated_memory, mem)

        t_epo += time.time() - t1

        if mode == "Train" and no_neighbor_flag == False:
            t2 = time.time()
            optimizer.zero_grad()
            predicts = torch.cat([pos_score, neg_score], dim=0).to(device)
            labels = torch.cat(
                [torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0
            ).to(device)
            loss = criterion(input=predicts, target=labels)
            loss.backward()
            optimizer.step()
            t_epo += time.time() - t2
        
        else:
            with torch.no_grad():
                pos_cpu = pos_score.sigmoid().cpu()
                neg_cpu = neg_score.sigmoid().cpu()
                all_pos_score.append(pos_cpu)
                all_neg_score.append(neg_cpu)
                del pos_score, neg_score
                torch.cuda.empty_cache()

                ap, mrr, hr_list = compute_metrics(pos_cpu, neg_cpu, 'cpu', k_list=k_list)
                ap_list.append(ap)
                mrr_list.append(mrr)
                hit_list.append(hr_list)
        
        torch.cuda.empty_cache()

    if mode == "Train":
        print(
            f"\nTime-Train-Epoch{epoch+1}-Neg{num_neg}: loss = {loss.item():.5f}, time: {t_epo}, memory used: {allocated_memory}"
        )
        sys.stdout.flush()
        return t_epo, allocated_memory

    else:
        ap = np.mean(ap_list)
        mrr = np.mean(mrr_list)
        hit_array = np.array(hit_list)
        mean_hr = np.mean(hit_array, axis=0)
        
        if mode == "Val":
            print(
                f"Time-Val-Epoch{epoch+1}-Neg{num_neg}: ap = {ap:.4f}, mrr = {mrr:.4f}, "
                + ", ".join([f"hr@{k} = {hr:.4f}" for k, hr in zip(k_list, mean_hr)])
                + f", time: {t_epo}, memory used: {allocated_memory}"
            )

        else:
            print(
                f"Time-Test-Neg{num_neg}: ap = {ap:.4f}, mrr = {mrr:.4f}, "
                + ", ".join([f"hr@{k} = {hr:.4f}" for k, hr in zip(k_list, mean_hr)])
                + f", time: {t_epo}, memory used: {allocated_memory}"
            )

        sys.stdout.flush()

    return ap, mrr, mean_hr, t_epo, all_pos_score, all_neg_score, allocated_memory


def train_time(time_saved_name):
    best_model_path = f"saved_time_models/{args.dataset_name}/{time_saved_name}.pth"

    finder = get_neighbor_finder(full_data)

    edge_predictor_configs = {
        "dim": args.hidden_dims,
    }

    mixer_configs = {
        "per_graph_size": args.topk_time,
        "time_channels": args.hidden_dims,
        "num_layers": args.num_layers,
        "device": device,
    }

    model = Mixer_per_node(mixer_configs, edge_predictor_configs).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    early_stopper = EarlyStopMonitor(max_round=args.patience)

    train_num_batch, train_delta_times_list, train_all_inds_list, train_batch_size_list = (
        load_precessed_time_data(
            "Train",
            finder,
            train_data,
            args.batch_size,
            NUM_NEG_TRAIN,
            train_neg_edge_sampler
        )
    )
    val_num_batch, val_delta_times_list, val_all_inds_list, val_batch_size_list = (
        load_precessed_time_data(
            "Val",
            finder,
            val_data,
            args.batch_size,
            NUM_NEG_VAL,
            tgb_eval_neg_sampler
        )
    )
    test_num_batch, test_delta_times_list, test_all_inds_list, test_batch_size_list = (
        load_precessed_time_data(
            "Test",
            finder,
            test_data,
            args.batch_size,
            NUM_NEG_TEST,
            tgb_eval_neg_sampler
        )
    )

    t_train = 0.0
    t_val = 0.0

    val_time_score_filepath = f"time_score_cache/{args.dataset_name}/val_{time_saved_name}"
    test_time_score_filepath = f"time_score_cache/{args.dataset_name}/test_{time_saved_name}"
    
    for epoch in range(args.num_epochs):
        t_train_epo, mem_train = time_forward(
            model,
            "Train",
            epoch,
            optimizer,
            criterion,
            num_neg=NUM_NEG_TRAIN,
            num_batch=train_num_batch,
            delta_times_list=train_delta_times_list,
            all_inds_list=train_all_inds_list,
            batch_size_list=train_batch_size_list,
        )
        t_train += t_train_epo

        with torch.no_grad():
            val_ap, val_mrr, val_hr_list, t_val_epo, val_all_pos_score, val_all_neg_score, mem_val = time_forward(
                model,
                "Val",
                epoch,
                None,
                None,
                num_neg=NUM_NEG_VAL,
                num_batch=val_num_batch,
                delta_times_list=val_delta_times_list,
                all_inds_list=val_all_inds_list,
                batch_size_list=val_batch_size_list,
            )
            t_val += t_val_epo

            if early_stopper.early_stop_check(val_ap):
                print(f"\nLoading the best model params from {best_model_path}")
                sys.stdout.flush()
                model_parameters = torch.load(best_model_path, map_location=device)
                model.load_state_dict(model_parameters)
                model.eval()
                break
            else:
                if epoch == early_stopper.best_epoch:
                    torch.save((model.state_dict()), best_model_path)
                    print(f"Saving the best model params at {best_model_path}")
                    val_time_score = val_all_pos_score, val_all_neg_score
                    val_time_score = move_to_cpu(val_time_score)
                    os.makedirs(os.path.dirname(val_time_score_filepath), exist_ok=True)
                    with open(val_time_score_filepath, "wb") as vf:
                        pickle.dump(val_time_score, vf)
                    print(f"Time-Val scores in epoch {epoch+1} have been saved at {val_time_score_filepath}")
                    sys.stdout.flush()

    with torch.no_grad():
        test_ap, test_mrr, test_hr_list, t_test, test_all_pos_score, test_all_neg_score, mem_test = time_forward(
            model,
            "Test",
            early_stopper.best_epoch,
            None,
            None,
            num_neg=NUM_NEG_TEST,
            num_batch=test_num_batch,
            delta_times_list=test_delta_times_list,
            all_inds_list=test_all_inds_list,
            batch_size_list=test_batch_size_list,
        )
        test_time_score = test_all_pos_score, test_all_neg_score
        test_time_score = move_to_cpu(test_time_score)
        os.makedirs(os.path.dirname(test_time_score_filepath), exist_ok=True)
        with open(test_time_score_filepath, "wb") as tf:
            pickle.dump(test_time_score, tf)
        print(f"Time-Test scores have been saved at {test_time_score_filepath}")

        print(
            f"\nBest_epoch: {early_stopper.best_epoch+1}, total_train_time: {t_train:4f}, val_time: {t_val:4f}, test_time: {t_test:4f}, memory_train: {mem_train}, memory_val: {mem_val}, memory_test: {mem_test}"
        )

    return val_ap, val_mrr, val_hr_list, test_ap, test_mrr, test_hr_list


# EAGLE-Hybrid
def mix_scores(mode, batch_size, data, time_pos_score, time_neg_score, structure_score, delta_times_list, all_inds_list, num_neg_per_pos, time_topk, yita, device):
    num_pos_edge = data.n_interactions
    BATCH_SIZE = batch_size if batch_size != -1 else num_pos_edge
    num_batch = math.ceil(num_pos_edge / BATCH_SIZE)

    ap_list, mrr_list, hit_list = [], [], []

    for batch_idx in range(0, num_batch):
        batch_time_pos_score = time_pos_score[batch_idx].to(device)
        batch_time_neg_score = time_neg_score[batch_idx].to(device)

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_pos_edge, start_idx + BATCH_SIZE)
        pos_ids = np.array(list(range(start_idx, end_idx)))

        cur_batch_size = min(BATCH_SIZE, num_pos_edge - start_idx)

        neg_ids = np.concatenate([pos_ids + i*num_pos_edge for i in range(1, 1 + num_neg_per_pos)])
        batch_skc_pos_score = structure_score[pos_ids]
        batch_skc_neg_score = structure_score[neg_ids]


        delta_times, all_inds = delta_times_list[batch_idx].to(device), all_inds_list[batch_idx].to(device)


        total_groups = (2 + num_neg_per_pos) * cur_batch_size
        # max_delta = delta_times.max()

        groups_avg_dts = []

        delta_times = delta_times.squeeze(1) # [2906]

        all_cur_full_ids = torch.arange(total_groups * time_topk, device=device).reshape(total_groups, time_topk)

        groups_avg_dts = []

        max_delta_value = delta_times.max()

        for group_id in range(total_groups):
            cur_full_ids = all_cur_full_ids[group_id]
            
            # check existed cur_ids in all_inds
            mask = (all_inds.unsqueeze(1) == cur_full_ids.unsqueeze(0)) # [2906, time_topk]

            matched = mask.any(dim=1) # [2906], in-True, not in-False

            cur_top_ids = torch.nonzero(matched, as_tuple=False).squeeze(1) # [num_matched]

            if cur_top_ids.numel() > 0:
                group_deltas = delta_times[cur_top_ids] # [num_matched]
            else:
                group_deltas = torch.tensor([], dtype=delta_times.dtype, device=device)

            num_matched = group_deltas.size(0)
            if num_matched < time_topk:
                padding = torch.full((time_topk - num_matched,), max_delta_value, dtype=delta_times.dtype, device=device)
                group_deltas = torch.cat([group_deltas, padding], dim=0) # [time_topk]

            avg_dt = group_deltas.mean()
            groups_avg_dts.append(avg_dt)

        avg_dts_tensor = torch.stack(groups_avg_dts) # [total_groups]
        avg_dts_tensor = avg_dts_tensor / avg_dts_tensor.mean() - 1

        src_dts = avg_dts_tensor[:cur_batch_size]
        pos_dst_dts = avg_dts_tensor[cur_batch_size : 2*cur_batch_size]
        neg_dst_dts = avg_dts_tensor[2*cur_batch_size:] # [num_neg_per_pos * BATCH_SIZE]

        batch_skc_pos_score_tensor = torch.tensor(batch_skc_pos_score, dtype=torch.float32, device=device) # [BATCH_SIZE]
        batch_skc_neg_score_tensor = torch.tensor(batch_skc_neg_score, dtype=torch.float32, device=device) # [num_neg_per_pos*BATCH_SIZE]

        batch_time_pos_score = batch_time_pos_score.squeeze(1) # [BATCH_SIZE]
        batch_time_neg_score = batch_time_neg_score.squeeze(1) # [num_neg_per_pos*BATCH_SIZE]
        
        hy_pos_weight = yita * ((torch.exp(-src_dts) + torch.exp(-pos_dst_dts))/2) # [BATCH_SIZE]
        batch_hy_pos_score = hy_pos_weight * batch_time_pos_score + batch_skc_pos_score_tensor # [BATCH_SIZE]

        hy_neg_weight = yita * ((torch.exp(-src_dts.repeat(num_neg_per_pos)) + torch.exp(-neg_dst_dts))/2) # [num_neg_per_pos*BATCH_SIZE]
        batch_hy_neg_score = hy_neg_weight * batch_time_neg_score + batch_skc_neg_score_tensor  # [num_neg_per_pos * BATCH_SIZE]

        ap, mrr, hr_list = compute_metrics(batch_hy_pos_score, batch_hy_neg_score, device, k_list=k_list)
        ap_list.append(ap)
        mrr_list.append(mrr)
        hit_list.append(hr_list)

    ap = np.mean(ap_list)
    mrr = np.mean(mrr_list)
    hit_array = np.array(hit_list)
    all_hr = np.mean(hit_array, axis=0)
    print(f"Hybrid-{mode}-yita[{yita}]: ap = {ap:.4f}, mrr = {mrr:.4f}, " + ", ".join([f"hr@{k} = {hr:.4f}" for k, hr in zip(k_list, all_hr)]))
    sys.stdout.flush()

    return ap, mrr, all_hr


def train_hybrid(structure_saved_name, time_saved_name):
    val_structure_score_filepath = f"structure_score_cache/{args.dataset_name}/val_{structure_saved_name}"
    val_time_score_filepath = f"time_score_cache/{args.dataset_name}/val_{time_saved_name}"

    test_structure_score_filepath = f"structure_score_cache/{args.dataset_name}/test_{structure_saved_name}"
    test_time_score_filepath = f"time_score_cache/{args.dataset_name}/test_{time_saved_name}"

    with open(val_structure_score_filepath, 'rb') as vsf:
        val_structure_data = pickle.load(vsf)
    val_structure_score = val_structure_data[2]
    
    with open(val_time_score_filepath, 'rb') as vtf:
        val_time_score = pickle.load(vtf)
    val_time_pos_score, val_time_neg_score = val_time_score
    
    with open(test_structure_score_filepath, 'rb') as tsf:
        test_structure_data = pickle.load(tsf)
    test_structure_score = test_structure_data[2]
    
    with open(test_time_score_filepath, 'rb') as ttf:
        test_time_score = pickle.load(ttf)
    test_time_pos_score, test_time_neg_score = test_time_score

    val_time_data_filepath = f"time_processed_data/{args.dataset_name}/Val_bs_{args.batch_size}_topk_{args.topk_time}_sample_{args.time_sample_strategy}_seed_{args.seed}.pkl"
    with open(val_time_data_filepath, 'rb') as vtdf:
        _, val_delta_times_list, val_all_inds_list, _ = pickle.load(vtdf)

    test_time_data_filepath = f"time_processed_data/{args.dataset_name}/Test_bs_{args.batch_size}_topk_{args.topk_time}_sample_{args.time_sample_strategy}_seed_{args.seed}.pkl"
    with open(test_time_data_filepath, 'rb') as ttdf:
        _, test_delta_times_list, test_all_inds_list, _ = pickle.load(ttdf)


    with torch.no_grad():
        val_ap, val_mrr, val_hr_list = mix_scores("Val", args.batch_size, val_data, val_time_pos_score, val_time_neg_score, val_structure_score, val_delta_times_list, val_all_inds_list, NUM_NEG_VAL, args.topk_time, yita=args.yita, device=device)
        test_ap, test_mrr, test_hr_list = mix_scores("Test", args.batch_size, test_data, test_time_pos_score, test_time_neg_score, test_structure_score, test_delta_times_list, test_all_inds_list, NUM_NEG_TEST, args.topk_time, yita=args.yita, device=device)

    return val_ap, val_mrr, val_hr_list, test_ap, test_mrr, test_hr_list



if args.model == "structure" or args.model == "hybrid":
    structure_saved_name = (
        "bs_" + str(args.batch_size) + 
        "_seed_" + str(args.seed) + 
        "_topk_" + str(args.topk_struct) + 
        "_alpha_" + str(args.alpha) + 
        "_beta_" + str(args.beta)
    )
    
    Path(f"structure_score_cache/{args.dataset_name}").mkdir(parents=True, exist_ok=True)
    
if args.model == "time" or args.model == "hybrid":
    time_saved_name = (
        "bs_" + str(args.batch_size) + 
        "_seed_" + str(args.seed) + 
        "_topk_" + str(args.topk_time) + 
        "_sample_" + args.time_sample_strategy + 
        "_nl_" + str(args.num_layers) + 
        "_hd_" + str(args.hidden_dims) + 
        "_lr_" + str(args.lr) + 
        "_wd_" + str(args.weight_decay)
    )

    Path(f"time_processed_data/{args.dataset_name}").mkdir(parents=True, exist_ok=True)
    Path(f"saved_time_models/{args.dataset_name}").mkdir(parents=True, exist_ok=True)
    Path(f"time_score_cache/{args.dataset_name}").mkdir(parents=True, exist_ok=True)


val_aps, val_mrrs, val_hr_lists, test_aps, test_mrrs, test_hr_lists = [], [], [], [], [], []

# no randomness in EAGLE-Structure calculation process
if args.model == "structure" or args.model == "hybrid":
    structure_val_ap, structure_val_mrr, structure_val_hr_list, structure_test_ap, structure_test_mrr, structure_test_hr_list = train_structure(structure_saved_name)

for run in range(args.num_runs):
    print(f"Run {run+1}/{args.num_runs} begins:")
    running_time_saved_name = time_saved_name + f"_run_{run+1}_in_{args.num_runs}"
    
    if args.model == "time" or args.model == "hybrid":
        time_val_ap, time_val_mrr, time_val_hr_list, time_test_ap, time_test_mrr, time_test_hr_list = train_time(running_time_saved_name)

    if args.model == "hybrid":
        torch.cuda.empty_cache()
        hybrid_val_ap, hybrid_val_mrr, hybrid_val_hr_list, hybrid_test_ap, hybrid_test_mrr, hybrid_test_hr_list = train_hybrid(structure_saved_name, running_time_saved_name)

    print(f"\n")
    
    val_ap, val_mrr, val_hr_list, test_ap, test_mrr, test_hr_list = [globals()[f"{args.model}_{metric}"] for metric in ["val_ap", "val_mrr", "val_hr_list", "test_ap", "test_mrr", "test_hr_list"]]
    val_aps.append(val_ap)
    val_mrrs.append(val_mrr)
    val_hr_lists.append(val_hr_list)
    test_aps.append(test_ap)
    test_mrrs.append(test_mrr)
    test_hr_lists.append(test_hr_list)

def calculate_stats(values):
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

def calculate_hr_stats(hr_lists):
    hr_arr = np.asarray(hr_lists, dtype=float)
    hr_by_k = hr_arr.T
    hr_mean_list = []
    for k, vals in zip(k_list, hr_by_k):
        mean = vals.mean()
        std  = vals.std(ddof=1) if vals.size > 1 else 0.0
        hr_mean_list.append(f"hr@{k} = {mean:.4f}±{std:.4f}")
    
    return hr_mean_list

print("============================== Final Results ==============================\n")
print(f"Val:\nap = {calculate_stats(val_aps)}\nmrr = {calculate_stats(val_mrrs)}")
print("\n".join(calculate_hr_stats(val_hr_lists)))
print(f"\nTest:\nap = {calculate_stats(test_aps)}\nmrr = {calculate_stats(test_mrrs)}")
print("\n".join(calculate_hr_stats(test_hr_lists)))
