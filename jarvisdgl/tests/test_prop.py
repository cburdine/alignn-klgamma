from jarvisdgl.main import train_property_model


def test_prop():
    train_property_model(epochs=2, maxrows=16, batch_size=8)
