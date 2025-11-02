from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    if num_batches <= 0 or num_partitions <= 0:
        return

    total_steps = num_batches + num_partitions - 1
    for k in range(total_steps):
        step = []
        for partition_idx in range(num_partitions):
            batch_idx = k - partition_idx
            if 0 <= batch_idx < num_batches:
                step.append((batch_idx, partition_idx))
        if step:
            yield step
    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.

        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2
        if self.split_size < 1:
            raise ValueError("split_size must be at least 1")

        num_partitions = len(self.partitions)
        if num_partitions == 0:
            return x

        micro_batches = list(torch.split(x, self.split_size, dim=0))

        batches: List[List[Optional[Tensor]]] = [
            [None] * (num_partitions + 1) for _ in range(len(micro_batches))
        ]

        first_device = self.devices[0]
        for idx, micro_batch in enumerate(micro_batches):
            if micro_batch.device != first_device:
                micro_batch = micro_batch.to(first_device)
                micro_batches[idx] = micro_batch
            batches[idx][0] = micro_batch

        for step in _clock_cycles(len(micro_batches), num_partitions):
            self.compute(batches, step)

        outputs = [batches[idx][num_partitions] for idx in range(len(micro_batches))]
        result = torch.cat(outputs, dim=0)
        return result.to(self.devices[-1])
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        pending = []
        for micro_idx, partition_idx in schedule:
            partition = partitions[partition_idx]
            device = devices[partition_idx]
            input_batch = batches[micro_idx][partition_idx]
            if input_batch is None:
                raise RuntimeError("Pipeline stage received empty input batch")
            if input_batch.device != device:
                input_batch = input_batch.to(device)
                batches[micro_idx][partition_idx] = input_batch

            def compute_fn(partition=partition, input_batch=input_batch):
                return partition(input_batch)

            task = Task(compute_fn)
            in_queue = self.in_queues[partition_idx]
            out_queue = self.out_queues[partition_idx]
            in_queue.put(task)
            pending.append((micro_idx, partition_idx, out_queue))

        for micro_idx, partition_idx, out_queue in pending:
            success, payload = out_queue.get()
            if not success:
                _, exc_info = payload
                raise exc_info[1].with_traceback(exc_info[2])

            _, output = payload
            batches[micro_idx][partition_idx + 1] = output
        # END ASSIGN5_2_2
