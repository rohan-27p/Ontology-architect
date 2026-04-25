from ontology_architect.config import UniverseConfig
from ontology_architect.universe import ProceduralAlienUniverse, format_sensor_log, window_has_drift


def test_universe_is_deterministic_for_seed():
    config = UniverseConfig(seed=123, family="thermal_split", anomaly_rate=0.2)
    first = ProceduralAlienUniverse(config).generate(16)
    second = ProceduralAlienUniverse(config).generate(16)

    assert [row.to_json() for row in first] == [row.to_json() for row in second]


def test_drift_is_hidden_from_sensor_log_but_available_to_scorer():
    config = UniverseConfig(
        seed=5,
        family="coupled_oscillator",
        future_window=2,
        drift_interval=1,
        anomaly_rate=0.0,
    )
    records = ProceduralAlienUniverse(config).generate(8)
    log = format_sensor_log(records)

    assert window_has_drift(records)
    assert "drift" not in log.lower()
