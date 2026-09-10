import json
import os

import pytest

from tsm.config import load_config, parse_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("name", ["paris4.json", "paris4_roi.json"])
def test_load_shipped_configs(name):
    cfg = load_config(os.path.join(ROOT, "configs", name))
    assert cfg.volume.url.startswith("https://")
    assert cfg.volume.alt_url.startswith("s3://")
    assert cfg.volume.level == 0 and cfg.volume.voxel_um == 2.4
    assert cfg.region.start_zyx[0] == 34176
    assert cfg.region.size_zyx[0] in (128, 256)
    assert cfg.budget.ram_bytes == 8 << 30
    assert cfg.out_dir.startswith("/home/forrest/tsm-output/")


def test_region_stop():
    cfg = parse_config(
        {
            "volume": {"url": "/tmp/x"},
            "region": {"start_zyx": [1, 2, 3], "size_zyx": [4, 5, 6]},
            "out_dir": "/tmp/o",
        }
    )
    assert cfg.region.stop_zyx == (5, 7, 9)
    assert cfg.extra == {}


def test_unknown_top_level_key_rejected():
    with pytest.raises(ValueError, match="unknown top-level"):
        parse_config(
            {
                "volume": {"url": "/tmp/x"},
                "region": {"start_zyx": [0, 0, 0], "size_zyx": [1, 1, 1]},
                "out_dir": "/tmp/o",
                "bogus": 1,
            }
        )


def test_bad_types_rejected():
    base = {
        "volume": {"url": "/tmp/x"},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [1, 1, 1]},
        "out_dir": "/tmp/o",
    }
    with pytest.raises(ValueError):
        parse_config({**base, "region": {"start_zyx": [0, 0], "size_zyx": [1, 1, 1]}})
    with pytest.raises(ValueError):
        parse_config({**base, "volume": {"url": "/tmp/x", "level": -1}})
    with pytest.raises(ValueError):
        parse_config({**base, "volume": {"url": "/tmp/x", "zzz": 1}})
    with pytest.raises(ValueError):
        parse_config({**base, "out_dir": ""})


def test_load_config_from_file(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "volume": {"url": "/tmp/x", "voxel_um": 9.6},
                "region": {"start_zyx": [0, 0, 0], "size_zyx": [2, 2, 2]},
                "out_dir": "/tmp/o",
                "extra": {"teacher": {"patch": 256}},
            }
        )
    )
    cfg = load_config(str(p))
    assert cfg.volume.voxel_um == 9.6
    assert cfg.extra["teacher"]["patch"] == 256
