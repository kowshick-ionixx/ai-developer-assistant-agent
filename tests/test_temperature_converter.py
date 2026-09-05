import pytest

from temperature_converter import celsius_to_fahrenheit, fahrenheit_to_celsius


@pytest.mark.parametrize(
    "celsius,expected",
    [
        (0.0, 32.0),
        (100.0, 212.0),
        (-40.0, -40.0),
        (25.0, 77.0),
        (-273.15, -459.67),
    ],
)
def test_celsius_to_fahrenheit(celsius, expected):
    assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)


@pytest.mark.parametrize(
    "fahrenheit,expected",
    [
        (32.0, 0.0),
        (212.0, 100.0),
        (-40.0, -40.0),
        (77.0, 25.0),
        (-459.67, -273.15),
    ],
)
def test_fahrenheit_to_celsius(fahrenheit, expected):
    assert fahrenheit_to_celsius(fahrenheit) == pytest.approx(expected)
