"""
Copyright 2021 Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT
Copyright 2022 Marcin Kowiel.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import scipy
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction


def kmeans_clustering(kmeans_dataset, num_centers, n_init=10):
    kmeans = KMeans(init='k-means++', n_init=n_init, n_clusters=num_centers).fit(kmeans_dataset)
    return kmeans.cluster_centers_


def select_closest_to_centers(options, centers):
    dist_ctr = pairwise_distances(options, centers)
    min_dist_indices = np.argmin(dist_ctr, axis=0)
    return list(min_dist_indices)


def kmeans_acquisition(
    dataset_x: AbstractDataSource,
    batch_size: int,
    available_indices: List[AnyStr],
    last_selected_indices: List[AnyStr],
    last_model: AbstractBaseModel,
    use_embedding=False,
    n_init=10
) -> List:
    if use_embedding and hasattr(last_model, 'get_embedding'):
        kmeans_dataset = last_model.get_embedding(dataset_x.subset(available_indices)).numpy()
    else:
        kmeans_dataset = np.squeeze(dataset_x.subset(available_indices), axis=1)

    centers = kmeans_clustering(kmeans_dataset, batch_size, n_init=n_init)
    chosen = select_closest_to_centers(kmeans_dataset, centers)
    return [available_indices[idx] for idx in chosen]


def select_most_distant(options, previously_selected, num_samples):
    num_options, num_selected = len(options), len(previously_selected)
    if num_selected == 0:
        min_dist = np.tile(float("inf"), num_options)
    else:
        dist_ctr = pairwise_distances(options, previously_selected)
        min_dist = np.amin(dist_ctr, axis=1)

    indices = []
    for i in range(num_samples):
        idx = min_dist.argmax()
        dist_new_ctr = pairwise_distances(options, options[[idx], :])
        for j in range(num_options):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
        indices.append(idx)
    return indices


def core_set_acquisition(
    dataset_x: AbstractDataSource,
    batch_size: int,
    available_indices: List[AnyStr],
    last_selected_indices: List[AnyStr],
    last_model: AbstractBaseModel) -> List:
    if hasattr(last_model, 'get_embedding'):
        topmost_hidden_representation = last_model.get_embedding(dataset_x.subset(available_indices)).numpy()
        selected_hidden_representations = last_model.get_embedding(dataset_x.subset(last_selected_indices)).numpy()
    else:
        topmost_hidden_representation = np.squeeze(dataset_x.subset(available_indices), axis=1)
        selected_hidden_representations = np.squeeze(dataset_x.subset(last_selected_indices), axis=1)
    chosen = select_most_distant(topmost_hidden_representation, selected_hidden_representations, batch_size)
    return [available_indices[idx] for idx in chosen]


def top_uncertain_acquisition(
    dataset_x: AbstractDataSource,
    select_size: int,
    available_indices: List[AnyStr],
    last_selected_indices: List[AnyStr] = None,
    model: AbstractBaseModel = None,
) -> List:
    avail_dataset_x = dataset_x.subset(available_indices)
    model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=True)

    if len(model_pedictions) != 3:
        raise TypeError("The provided model does not output uncertainty.")

    pred_mean, pred_uncertainties, _ = model_pedictions

    if len(pred_mean) < select_size:
        raise ValueError("The number of query samples exceeds"
                            "the size of the available data.")

    numerical_selected_indices = np.flip(
        np.argsort(pred_uncertainties)
    )[:select_size]
    selected_indices = [available_indices[i] for i in numerical_selected_indices]

    return selected_indices


def kmeans_initialise(gradient_embedding, k):
    ind = np.argmax([np.linalg.norm(s, 2) for s in gradient_embedding])
    mu = [gradient_embedding[ind]]
    indsAll = [ind]
    centInds = [0.] * len(gradient_embedding)
    cent = 0
    while len(mu) < k:
        if len(mu) == 1:
            D2 = pairwise_distances(gradient_embedding, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(gradient_embedding, [mu[-1]]).ravel().astype(float)
            for i in range(len(gradient_embedding)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = scipy.stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(gradient_embedding[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(gradient_embedding[indsAll], gradient_embedding[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll


def badge_acquisition(dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
    if hasattr(last_model, 'get_gradient_embedding'):
        gradient_embedding = last_model.get_gradient_embedding(dataset_x.subset(available_indices)).numpy()
    else:
        gradient_embedding = np.squeeze(dataset_x.subset(available_indices), axis=1)
    chosen = kmeans_initialise(gradient_embedding, batch_size)
    selected = [available_indices[idx] for idx in chosen]
    return selected


def combined_core_set_top_uncertain_kmeans_badge_acquisition(
    dataset_x: AbstractDataSource,
    batch_size: int,
    available_indices: List[AnyStr],
    last_selected_indices: List[AnyStr] = None,
    last_model: AbstractBaseModel = None,
) -> List:
    badge_ids = badge_acquisition(dataset_x, batch_size, available_indices, last_selected_indices, last_model)
    core_set_ids = core_set_acquisition(dataset_x, batch_size, available_indices, last_selected_indices, last_model)
    kmeans_ids = kmeans_acquisition(dataset_x, batch_size, available_indices, last_selected_indices, last_model)
    top_uncertain_ids = top_uncertain_acquisition(dataset_x, batch_size, available_indices, last_selected_indices, last_model)
    combined_ids = set(core_set_ids).union(set(kmeans_ids))
    combined_ids = combined_ids.union(set(top_uncertain_ids))
    combined_ids = combined_ids.union(set(badge_ids))
    selected = np.random.choice(list(combined_ids), size=batch_size, replace=False)
    return selected

class CustomAcquisitionFunction(BaseBatchAcquisitionFunction):
    def __call__(self, dataset_x: AbstractDataSource, batch_size: int, available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr], last_model: AbstractBaseModel) -> List:
        return combined_core_set_top_uncertain_kmeans_badge_acquisition(dataset_x, batch_size, available_indices, last_selected_indices, last_model)
