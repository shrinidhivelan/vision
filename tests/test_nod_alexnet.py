from brainscore_vision import model_registry

def test_alexnet_registered():
    assert "alexnet_nod" in model_registry

def test_alexnet_layers_nonempty():
    model = model_registry["alexnet_nod"]()
    assert hasattr(model, "layers")
    assert len(model.layers) > 0