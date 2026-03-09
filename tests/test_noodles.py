"""Tests for `noodles` package."""

import pytest

from noodles.comfy import LTXImg2VidInplaceNood


@pytest.fixture
def example_node():
    """Fixture to create an Example node instance."""
    return LTXImg2VidInplaceNood()


def test_example_node_initialization(example_node):
    """Test that the node can be instantiated."""
    assert isinstance(example_node, LTXImg2VidInplaceNood)


def test_return_types():
    """Test the node's metadata."""
    assert LTXImg2VidInplaceNood.RETURN_TYPES == ("IMAGE",)
    assert LTXImg2VidInplaceNood.FUNCTION == "test"
    assert LTXImg2VidInplaceNood.CATEGORY == "Example"
