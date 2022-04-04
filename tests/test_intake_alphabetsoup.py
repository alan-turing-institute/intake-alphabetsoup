import intake
from typing import Callable

def test_default_usage():
    assert isinstance(intake.open_synthetic_alphabetsoup, Callable)

    # default font
    ds = intake.open_synthetic_alphabetsoup()

    images, boxes, labels = ds.read()

    assert len(images) == 1
    assert len(boxes) == 1
    assert len(labels) == 1


def test_usage1():
    assert isinstance(intake.open_synthetic_alphabetsoup, Callable)
    ds = intake.open_synthetic_alphabetsoup(
        image_count = 10,
        datashape = [512, 512],
        ctf_defocus=5000,
        ctf_box_size=512,
        font_face="Verdana",
        font_size=20
    )

    data0 = ds.read_partition(0)
    data1 = ds.read_partition(1)
    data00 = ds.read_partition(0)

    # check that data is returned consistently
    assert (data0[0] == data00[0]).all()
    assert (data0[0] != data1[0]).any()


def test_seed():
    assert isinstance(intake.open_synthetic_alphabetsoup, Callable)

    ds1 = intake.open_synthetic_alphabetsoup(image_count=2)
    ds2 = intake.open_synthetic_alphabetsoup(image_count=2)

    data1_part0 = ds1.read_partition(0)

    # second data source: access partition 1, then partition 0
    data2_part1 = ds2.read_partition(1)
    data2_part0 = ds2.read_partition(0)

    assert (data1_part0[0] == data2_part0[0]).all()
    assert (data1_part0[0] != data2_part1[0]).any()
