from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN ASSIGN5_1_1
        data_index = self.index[index]
        return self.data[data_index]
        # END ASSIGN5_1_1

class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN ASSIGN5_1_1
        indices = list(range(len(self.data)))
        rng.shuffle(indices)

        total_size = len(indices)
        lengths = [int(size * total_size) for size in sizes]
        assigned = sum(lengths)
        remainder = total_size - assigned

        for i in range(remainder):
            lengths[i % len(lengths)] += 1

        start = 0
        for length in lengths:
            end = start + length
            self.partitions.append(indices[start:end])
            start = end
        # END ASSIGN5_1_1

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN ASSIGN5_1_1
        return Partition(self.data, self.partitions[partition])
        # END ASSIGN5_1_1

def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN ASSIGN5_1
    if world_size <= 0:
        raise ValueError("world_size must be a positive integer")

    partition_batch = batch_size // world_size
    if partition_batch <= 0:
        raise ValueError("partitioned batch size must be positive")

    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partitioner = DataPartitioner(dataset, sizes=partition_sizes)
    partition_dataset = partitioner.use(rank)

    return DataLoader(
        partition_dataset,
        batch_size=partition_batch,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # END ASSIGN5_1
