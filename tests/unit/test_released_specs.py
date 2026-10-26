"""A built-in spec that was released keeps its content for its id@version.

The registry refuses an id@version it already holds with different content, so changing a released
spec without bumping `strategy.version` stops the app on the user's PC (build-26, AUTO-LEV). Add
every newly released id@version here; change a spec only together with a new version."""

from pathlib import Path

import pytest

from quantlab.strategies.spec import load_spec

SPECS = Path(__file__).resolve().parents[2] / "src" / "quantlab" / "builtin" / "specs"

RELEASED = {
    "AUTO-DEVA@1.0.0": "f6abd70b8ed5f3e6b66b1f74886e6edd4f65b7d074d559165960b0576c0bc084",
    "AUTO-DEVB@1.0.0": "e80ce83e8c87a96a7083a08353df471871b501dffe086df347728b3d51422022",
    "AUTO-EM@1.0.0": "6649c8b3c9a95d34a3d228a0e7d6afd4d580caff4b7ab14dde160a0bdcd07cf8",
    "AUTO-REAL@1.0.0": "7ae81f3202e4265969a1e709878a52fb2621f0996304bbcbab9ec30da01b2b9c",
    "AUTO-LEV@1.0.0": "d4d7896e1ae9d0859f117cfac031cc63f0548c83e55937f3320dde2516609c64",  # build-24
    "AUTO-LEV@1.1.0": "2bc713bd791fb959b3edefb395e7c38ea42349b1b8dfa9b03a2e8ba04c77a118",
    "AUTO-LEVP@1.0.0": "3c866e577c21c3e148dc1d9e5f0bb942613b7e4a7a8f13dac64c26762bd2f963",  # build-26
    "N4X-INTL-1X@1.0.0": "7a39f786f0e985f38a98e7e9519e253ce5d9d5ef5a2e410def8209c571ad3239",
    "N5X-INTL-DEEP@1.0.0": "24b7112349fb3212cae744cab5a209205eecfa3a54cd226582753db0f9678af0",
    "OSR-SECT3X@1.0.0": "ac89fce7c933cd0ab492ae8d0c89ce023402140a274c8d5bf8ba9ce5c0cdeb73",
    "INS-CLUSTER@1.0.0": "946cb15349f6a81038c42d819b7778df70167713e16f5cb5bc94c97a991adce7",
    "PEAD-SP500@1.0.0": "11c393b3f224f7a9ece64e55e3e5a78969910a635b70781793b054841b8313d5",
    "AUTO-SP500@1.0.0": "aa59a58f66b16ca360b8a2d23480dcae3563f0156c1cc13af1ed3aa87b0d8280",
    "GAP-EARN-SP500@1.0.0": "8d0d36c12f8b011b1d5b1c3a3b706c553699aa8d25d69ffaa77429f49bead90b",
    "CAL-TOM-SECT3X@1.0.0": "de902400e6b66c8f460ef3762e89feb04806d60ab2fc7b928e9c1439ab2cd2cc",
    "AUTO-SP500X@1.0.0": "a4e373bb4f567eb0cdac35d8c100fee1ad2d737f4f70b6e3c9ee2d5267083360",
    "AUTO-SP500Y@1.0.0": "49f4a0ed495521f010abd064561052b719ae3b34806857b18b495df6b876f7b0",
    "AUTO-KR2X@1.0.0": "dc3e93a8648ab0ae01f8ddecdab970a132d71294f02dbf038fcaa597b482131d",
    "TT-SPY-SMA@1.0.0": "10ae1faed92e16a69efffa877888adc35970b809c381945faa91225fe48c7ce9",
    "LOWVOL-SP500@1.0.0": "8d7dbc547af450ca0faae6f41ca8a38cd5fa2062629104fa25c6df0478a76eb1",
    "INS-HOLD6M@1.0.0": "5bd9a241422d7abd4ab550019c2a6c081777238dedd211475f9554817616b57d",
    "VALQ-SP500@1.0.0": "ca27b5afcc3e144cdb6f1f6509aadcf13bfbed828e7ac87865cd26272f4f4c00",
    "AUTO-KRSTK@1.0.0": "5f609b0ae3440a3961b4cf3ef4f01da019e2990efed0e0aebbd25d7d0fc15eda",
    "AUTO-KRDART@1.0.0": "cb05d4f72293aeea90c35b685808545b86328c33ee92c4657eb31bf0c7394e88",
    "PAIR-KRSECT@1.0.0": "2c18ccf55860c6184f0fa767e192d73d64834c152a64b2fb393f5538a0cac8d8",
}


@pytest.mark.parametrize("path", sorted(SPECS.glob("*.qls.yaml")), ids=lambda p: p.name)
def test_released_spec_content_is_unchanged(path):
    loaded = load_spec(path)
    key = f"{loaded.strategy_id}@{loaded.version}"
    assert key in RELEASED, f"{key} is new: add its hash to RELEASED when it ships"
    assert loaded.strategy_hash == RELEASED[key], f"{key} changed after release: bump strategy.version"
