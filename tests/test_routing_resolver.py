import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from topos_train import resolve_routing_for_batch


def _args(topology):
    return argparse.Namespace(topology=topology)


def test_explicit_topology_bypasses_auto_logic():
    topology, chi = resolve_routing_for_batch(_args("volumetric"), {"chi": torch.tensor([2.0])})
    assert topology == "volumetric"
    assert chi is None


def test_auto_prefers_batch_chi():
    batch = {"chi": torch.tensor([0.0])}
    topology, chi = resolve_routing_for_batch(_args("auto"), batch)
    assert topology == "auto"
    assert chi == 0.0


def test_auto_reads_meta_euler_chi():
    batch = {"meta": {"euler_chi": [2.0]}}
    topology, chi = resolve_routing_for_batch(_args("auto"), batch)
    assert topology == "auto"
    assert chi == 2.0


def test_auto_falls_back_to_explicit_topology_label():
    batch = {"topology": ["toroidal"]}
    topology, chi = resolve_routing_for_batch(_args("auto"), batch, default_chi=None)
    assert topology == "toroidal"
    assert chi is None
