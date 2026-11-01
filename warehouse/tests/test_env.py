"""The grid, the orders and the distances."""

import numpy as np
import pytest

from warehouse import env as W


@pytest.mark.parametrize("name", list(W.SIZES))
def test_layout_is_connected_and_every_shelf_is_reachable(name):
    wh = W.Warehouse(W.SIZES[name])
    d = wh.dist_to(wh.parking[0])
    assert all(d[c] >= 0 for c in wh.free)
    assert all(d[a] >= 0 for a in wh.access.values())
    assert len(wh.parking) >= W.SIZES[name].robots


def test_corridor_two_fits():
    lay = W.Layout("wide", 24, 40, 4, 4, corridor=2, stations=2, robots=4)
    wh = W.Warehouse(lay)
    assert all(wh.dist_to(wh.stations[0])[c] >= 0 for c in wh.free)


def test_blocks_that_do_not_fit_are_refused():
    with pytest.raises(ValueError):
        W.Warehouse(W.Layout("x", 10, 10, 9, 9))


def test_access_cell_touches_its_shelf():
    wh = W.Warehouse(W.SIZES["small"])
    for k, (r, c) in enumerate(wh.shelves):
        a, b = wh.xy(wh.access[k])
        assert abs(a - r) + abs(b - c) == 1


def test_orders():
    wh = W.Warehouse(W.SIZES["small"])
    cat = W.Catalogue(wh, 500, 0)
    orders = W.batch_orders(wh, cat, 200, 0)
    assert all(3 <= len(o.products) <= 12 and len(set(o.products)) == len(o.products) for o in orders)
    assert all(o.station in wh.stations and o.shelves and o.arrival == 0 for o in orders)
    a = W.poisson_orders(wh, cat, 0.05, 2000, 0)
    assert 60 <= len(a) <= 140 and all(x.arrival <= y.arrival < 2000 for x, y in zip(a, a[1:]))
    assert [o.products for o in a] == [o.products for o in W.poisson_orders(wh, cat, 0.05, 2000, 0)]
