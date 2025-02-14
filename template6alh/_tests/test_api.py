"""
tests the api
"""

import tempfile
from pathlib import Path
import shutil
import logging

import nrrd
import numpy as np
from sqlalchemy import Engine, Connection, select, MetaData
from sqlalchemy.orm import Session
from skimage.data import binary_blobs

from template6alh import api, template_creation, matplotlib_slice
from template6alh import sql_classes as sc
from template6alh.utils import get_engine, run_with_logging, get_cmtk_executable
from template6alh.sql_utils import get_path, select_most_recent, get_imgs, ConfigDict

logger = logging.getLogger()
logger.setLevel("DEBUG")

eg_coords = matplotlib_slice.CoordsSet(
    brain=(0, 0, 0), sez=(0, 1, 1), tip=(1, 0, 1), scale=np.array([1, 1, 1])
)

header = dict(
    [
        ("space", "right-anterior-superior"),
        ("space directions", [[2, 0.0, 0.0], [0.0, 2, 0.0], [0.0, 0.0, 2]]),
        ("labels", ["x", "y", "z"]),
    ]
)


def check_init(engine: Engine, root_dir: Path):
    api.init(
        engine, [Path("test")], root_dir, neuropil_chan=2, fasii_chan=1, eve_chan=None
    )
    dirs = list(root_dir.iterdir())
    dir_names = set(d.name for d in dirs)
    assert dir_names == {"001", "002"}
    file_names = set(d.name for d in dirs[0].iterdir())
    assert file_names == set(("chan1.nrrd", "chan2.nrrd", "chan3.nrrd"))
    connection = Connection(engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    channels = connection.execute(select(metadata.tables["channel"])).fetchall()
    assert len(channels) == 6


def check_add_more_raw(session: Session, root_dir: Path):
    api.add_more_raw(
        session, [Path("test")], neuropil_chan=2, fasii_chan=1, eve_chan=None
    )
    dirs = list(root_dir.iterdir())
    dir_names = set(d.name for d in dirs)
    assert dir_names == {"001", "002", "003", "004"}
    channels = session.execute(select(sc.Channel)).scalars().all()
    assert len(channels) == 12
    unique_raw_file_ids = session.query(sc.Image.raw_file_id).distinct().all()
    assert len(unique_raw_file_ids) == 2
    images = session.execute(select(sc.Image)).scalars().all()
    assert images


def check_segment_neuropil(session: Session, root_dir: Path):
    images = session.execute(select(sc.Image)).scalars().all()
    assert len(images) != 0
    api.segment_neuropil(session, ["001"], 2.0, None, None)
    img_path = root_dir / "001"
    assert len(list(img_path.glob("*neuropil_mask.nrrd"))) == 1
    channels = (
        session.execute(select(sc.Channel).filter(sc.Channel.channel_type == "mask"))
        .scalars()
        .all()
    )
    assert len(channels) != 0
    for chan in channels:
        _, metadata = nrrd.read(str(get_path(session, chan)))
        spacings = np.diag(metadata["space directions"])
        assert np.array_equal(np.abs(spacings), [2, 2, 2])


def check_select_neuopil_fasii(session: Session, root_dir: Path):
    api.select_neuropil_fasii(session, None)
    assert list(root_dir.glob("**/*neuropil_fasii.nrrd"))
    images = get_imgs(session, None)
    path = None
    for image in images:
        channel = session.execute(
            select_most_recent("select-neuropil-fasii", image)
        ).scalar_one()
        assert channel.channel_type == "image"
        path = get_path(session, channel)
        assert path.exists()
    assert path is not None
    out = run_with_logging((get_cmtk_executable("describe"), "-m", path))
    response = out.stderr.decode() + out.stdout.decode()
    assert "(8bit unsigned)" in response


def check_clean(session: Session, root_dir: Path):
    channels = session.execute(select(sc.Channel)).scalars().all()
    assert len(channels) == 13
    assert channels[-1].channel_type == "mask"
    get_path(session, channels[-1]).unlink()
    api.clean(session)
    channels = session.execute(select(sc.Channel)).scalars().all()
    assert len(channels) == 12


def check_select_most_recent(session: Session):
    api.segment_neuropil(session, None, 2, None, None)
    images = get_imgs(session, None)
    assert len(images) == 4
    for image in images:
        channel = (
            session.execute(select_most_recent("make-neuropil-mask", image))
            .scalars()
            .first()
        )
        assert isinstance(channel, sc.Channel)
        assert channel.channel_type == "mask"
        print(get_path(session, channel))


def check_make_landmarks(session: Session, root_dir: Path):
    api.make_landmarks(session, None, True)
    images = get_imgs(session, None)
    assert len(images) != 0
    for image in images:
        channel = (
            session.execute(select_most_recent("make-landmarks", image))
            .scalars()
            .first()
        )
        assert isinstance(channel, sc.Channel)
        assert channel.channel_type == "landmarks"
        get_path(session, channel).write_text(eg_coords.to_cmtk())


def check_landmark_register(session: Session, root_dir: Path):
    api.landmark_register(session, None)
    assert list(root_dir.glob("**/*landmark.xform"))
    images = get_imgs(session, None)
    assert len(images) == 4
    for image in images:
        channel = (
            session.execute(select_most_recent("landmark-register", image))
            .scalars()
            .first()
        )
        assert isinstance(channel, sc.Channel)
        assert channel.channel_type == "xform"


def check_mask_register(session: Session, root_dir: Path):
    api.mask_register(session, ["002"])
    assert len(list(root_dir.glob("**/*affine_mask.xform"))) != 0
    assert len(list(root_dir.glob("**/*warp_mask.xform"))) != 0
    (image,) = get_imgs(session, ["002"])
    results = (
        session.execute(
            select_most_recent("mask-register", image)
            .join(sc.ChannelMetadata, sc.Channel.mdata)
            .filter(
                sc.ChannelMetadata.key == "xform-type",
                sc.ChannelMetadata.value == "warp",
            )
        )
        .scalars()
        .all()
    )
    assert len(results) == 1
    results = (
        session.execute(
            select_most_recent("mask-register", image)
            .join(sc.ChannelMetadata, sc.Channel.mdata)
            .filter(
                sc.ChannelMetadata.key == "xform-type",
                sc.ChannelMetadata.value == "affine",
            )
        )
        .scalars()
        .all()
    )
    assert len(results) == 1
    results = (
        session.execute(
            select_most_recent("mask-register", image).filter(
                sc.Channel.channel_type == "aligned",
            )
        )
        .scalars()
        .all()
    )
    assert len(results) == 1


def test_api():
    with tempfile.TemporaryDirectory() as folder_str:
        folder = Path(folder_str)
        folder = Path().home() / "Documents/t6alh"
        shutil.rmtree(folder)
        folder.mkdir(exist_ok=False)
        root_dir = folder / "root_dir"
        db_path = folder / "test.db"
        mt_path = root_dir / "template/mask_template.nrrd"
        engine = get_engine(db_path=db_path, delete_previous=True)
        check_init(engine, root_dir)
        with Session(engine) as session:
            check_add_more_raw(session, root_dir)
            check_segment_neuropil(session, root_dir)
            check_clean(session, root_dir)
            check_select_most_recent(session)
            check_select_neuopil_fasii(session, root_dir)
            check_make_landmarks(session, root_dir)
            mt_path.parent.mkdir(exist_ok=True, parents=True)
            nrrd.write(
                str(mt_path),
                binary_blobs(length=20, n_dim=3).astype(np.uint8) * 254,
                header=header,
            )
            mt_path.with_suffix(".landmarks").write_text(eg_coords.to_cmtk())
            check_landmark_register(session, root_dir)
            check_mask_register(session, root_dir)


def check_landmark_align(session: Session, root_dir: Path):
    template_creation.landmark_align(session, None, None, target_grid="8,45,20:5,5,5")
    assert list(root_dir.glob("**/*landmark_reformat.nrrd"))
    images = get_imgs(session, None)
    assert len(images) != 0
    for image in images:
        channel = (
            session.execute(select_most_recent("landmark-align", image))
            .scalars()
            .first()
        )
        assert isinstance(channel, sc.Channel)
        assert channel.channel_type == "aligned-mask"


def check_groupwise_template(session: Session, root_dir: Path):
    template_path = root_dir / "template/mask_template.nrrd"
    template_path.parent.mkdir(exist_ok=True)
    random_data = binary_blobs(
        length=100, blob_size_fraction=0.1, n_dim=3, volume_fraction=0.1
    )
    nrrd.write(
        file=str(template_path), data=random_data.astype(np.float32), header=header
    )
    template_creation.iterative_mask_template(
        session, ["001", "002"], make_template=False
    )
    template_path = ConfigDict(session)["mask_template_path"]
    data, _ = nrrd.read(str(template_path))
    assert data.max() == 254
    assert data.dtype == np.uint8


def check_fasii_template(session: Session, root_dir: Path):
    template_creation.fasii_template(session, None)


def test_template():
    with tempfile.TemporaryDirectory() as folder_str:
        folder = Path(folder_str)
        folder = Path().home() / "Documents/t6alh"
        shutil.rmtree(folder)
        folder.mkdir(exist_ok=False)
        root_dir = folder / "root_dir"
        db_path = folder / "test.db"
        engine = get_engine(db_path=db_path, delete_previous=True)
        api.init(
            engine,
            [Path("test")],
            root_dir,
            neuropil_chan=2,
            fasii_chan=1,
            eve_chan=None,
        )
        with Session(engine) as session:
            api.segment_neuropil(session, None, None, None, None)
            check_make_landmarks(session, root_dir)
            check_landmark_align(session, root_dir)
            check_groupwise_template(session, root_dir)
            check_select_neuopil_fasii(session, root_dir)
            check_fasii_template(session, root_dir)
