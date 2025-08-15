import numpy as np
import random
import pandas as pd
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.tbatch = None
        self.n_batch = 0

    def sample(self, ratio):
        data_size = self.n_interactions
        sample_size = int(ratio * data_size)
        sample_inds = random.sample(range(data_size), sample_size)
        sample_inds = np.sort(sample_inds)
        sources = self.sources[sample_inds]
        destination = self.destinations[sample_inds]
        timestamps = self.timestamps[sample_inds]
        edge_idxs = self.edge_idxs[sample_inds]
        labels = self.labels[sample_inds]
        return Data(sources, destination, timestamps, edge_idxs, labels)


def TGB2Data(tgb_slice):
    # TGB Data -> Data
    sources = tgb_slice.src.cpu().numpy()
    destination = tgb_slice.dst.cpu().numpy()
    timestamps = tgb_slice.t.cpu().numpy().astype(np.float64)
    edge_idxs = np.arange(len(sources), dtype=np.int64)
    labels = np.ones_like(sources, dtype=np.int32)
    return Data(sources, destination, timestamps, edge_idxs, labels)

def get_data_tgb(name="tgbl-wiki", root="data", use_validation=True):
    dataset = PyGLinkPropPredDataset(name=name, root=root)
    data_tgb = dataset.get_TemporalData()
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    metric = dataset.eval_metric
    neg_sampler = dataset.negative_sampler

    train_data = TGB2Data(data_tgb[train_mask])
    val_data = TGB2Data(data_tgb[val_mask])
    test_data  = TGB2Data(data_tgb[test_mask])
    full_data  = TGB2Data(data_tgb)

    node_set = set(full_data.sources) | set(full_data.destinations)
    n_total_unique_nodes = len(node_set)
    n_edges = len(full_data.sources)

    dataset.load_val_ns()
    dataset.load_test_ns()

    return full_data, train_data, val_data, test_data, n_total_unique_nodes, n_edges, metric, neg_sampler


# transductive setting
def get_data_transductive(dataset_name, use_validation=False):
    used_datasets = ["Contacts", "lastfm", "wikipedia", "reddit", "superuser", "askubuntu", "wikitalk"]
    other_datasets = [
        "mooc",
        "enron",
        "SocialEvo",
        "uci",
        "CollegeMsg",
        "TaobaoSmall",
        "CanParl",
        "Flights",
        "UNtrade",
        "USLegis",
        "UNvote",
        "Taobao",
        "DGraphFin",
        "TaobaoLarge",
        "YoutubeReddit",
        "YoutubeRedditLarge",
    ]
    if dataset_name in used_datasets:
        dir = "data"
    elif dataset_name in other_datasets:
        dir = "benchtemp_datasets"

    graph_df = pd.read_csv(f"../{dir}/{dataset_name}/ml_{dataset_name}.csv")

    # edge_features = np.load('../data/{}/ml_{}.npy'.format(dataset_name,dataset_name))
    # node_features = np.load('../data/{}/ml_{}_node.npy'.format(dataset_name,dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values.astype(np.float64)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)
    n_edges = len(sources)

    random.seed(2024)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time

    val_mask = (
        np.logical_and(timestamps <= test_time, timestamps > val_time)
        if use_validation
        else test_mask
    )

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )

    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )

    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            full_data.n_interactions, full_data.n_unique_nodes
        )
    )
    print(
        "The training dataset has {} interactions, involving {} different nodes".format(
            train_data.n_interactions, train_data.n_unique_nodes
        )
    )
    print(
        "The validation dataset has {} interactions, involving {} different nodes".format(
            val_data.n_interactions, val_data.n_unique_nodes
        )
    )
    print(
        "The test dataset has {} interactions, involving {} different nodes".format(
            test_data.n_interactions, test_data.n_unique_nodes
        )
    )

    return full_data, train_data, val_data, test_data, n_total_unique_nodes, n_edges
