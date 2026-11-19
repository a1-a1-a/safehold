"""Pure training schedule used by the GPU trainer."""


def submitted_indices(order, updates):
    """Repeat the fixed seeded ordering until the requested update budget."""
    if updates <= 0 or not order:
        raise ValueError("positive update budget and nonempty submitted corpus required")
    for step in range(updates):
        yield order[step % len(order)]


def cyclic_batch(step, pool_size, batch_size=8):
    """Return the next consecutive probe batch, wrapping at the pool end."""
    if pool_size <= 0 or batch_size <= 0:
        raise ValueError("positive pool and batch sizes required")
    start = (step * batch_size) % pool_size
    return [(start + i) % pool_size for i in range(batch_size)]
