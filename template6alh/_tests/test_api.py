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
from sqlalchemy.orm import Session, aliased
from skimage.data import binary_blobs

from template6alh import api, template_creation
from template6alh import sql_classes as sc
from template6alh.utils import get_engine
from template6alh.sql_utils import get_path

logger = logging.getLogger()
logger.setLevel("DEBUG")


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
    assert len(list(img_path.glob("*neuropil_mask_*.nrrd"))) == 8
    metadatas = (
        session.execute(
            select(sc.ChannelMetadata).where(sc.ChannelMetadata.key == "flip")
        )
        .scalars()
        .all()
    )
    assert len(metadatas) == 8
    unfliped_channel = next(m.channel for m in metadatas if m.value == "000")
    total_flipped_channel = next(m.channel for m in metadatas if m.value == "111")
    unflipped_data, _ = nrrd.read(str(get_path(session, unfliped_channel)))
    total_flipped_data, _ = nrrd.read(str(get_path(session, total_flipped_channel)))
    assert np.array_equal(unflipped_data, np.flip(total_flipped_data, (0, 1, 2)))

    for md in metadatas:
        _, metadata = nrrd.read(str(get_path(session, md.channel)))
        spacings = np.diag(metadata["space directions"])
        assert np.array_equal(np.abs(spacings), [2, 2, 2])


def check_clean(session: Session, root_dir: Path):
    channels = session.execute(select(sc.Channel)).scalars().all()
    assert len(channels) == 20
    assert channels[-1].channel_type == "mask"
    get_path(session, channels[-1]).unlink()
    api.clean(session)
    channels = session.execute(select(sc.Channel)).scalars().all()
    assert len(channels) == 19


def check_align_masks(session: Session, root_dir: Path):
    api.segment_neuropil(session, ["002"], None, None, None)
    api.mask_affine(session, ["002"])
    assert len(list(root_dir.glob("**/*affine_*.xform"))) != 0
    assert len(list(root_dir.glob("**/*reformat_*.nrrd"))) != 0
    session.commit()
    xforms = (
        session.execute(select(sc.Channel).filter(sc.Channel.channel_type == "xform"))
        .scalars()
        .all()
    )
    assert len(xforms) != 0
    for xform in xforms:
        path = get_path(session, xform)
        assert path.exists()
    unflip_producer = aliased(sc.AnalysisStep)
    xform_chan = aliased(sc.Channel)
    best_mask = aliased(sc.Channel)
    unflip_mask = aliased(sc.Channel)
    unflip_mask_mdata = aliased(sc.ChannelMetadata)
    best_00 = session.execute(
        select(sc.Channel, xform_chan, unflip_mask)
        .join(sc.ChannelMetadata, sc.Channel.mdata)
        .filter(
            sc.ChannelMetadata.key == "best-flip", sc.ChannelMetadata.value == "yes"
        )
        .join(sc.Image, sc.Channel.image)
        .filter(sc.Image.folder == "002")
        .join(sc.AnalysisStep, sc.Channel.producer)
        .join(xform_chan, sc.AnalysisStep.output_channels)
        .filter(xform_chan.channel_type == "xform")
        .join(best_mask, sc.AnalysisStep.input_channels)
        .join(unflip_producer, best_mask.producer)
        .join(unflip_mask, unflip_producer.output_channels)
        .join(unflip_mask_mdata, unflip_mask.mdata)
        .filter(unflip_mask_mdata.key == "flip", unflip_mask_mdata.value == "000")
        .order_by(sc.AnalysisStep.runtime.desc())
    ).all()
    assert len(best_00) == 1
    b = best_00[0]
    assert "neuropil_mask_000" in b[2].path
    assert "affine" in b[1].path
    bad_00 = session.execute(
        select(sc.Channel)
        .join(sc.ChannelMetadata, sc.Channel.mdata)
        .filter(sc.ChannelMetadata.key == "best-flip", sc.ChannelMetadata.value == "no")
    ).all()
    assert len(bad_00) == 7


def check_align_to_mask(session: Session, root_dir):
    api.align_to_mask(session, ["002"])
    (warped,) = (root_dir / "002").glob("*warp_mask.xform")
    ((_, chan),) = session.execute(
        select(sc.AnalysisStep, sc.Channel)
        .join(sc.Channel, sc.AnalysisStep.output_channels)
        .filter(sc.AnalysisStep.function == "align-to-mask")
    ).all()
    assert isinstance(chan, sc.Channel)
    assert chan.image.folder == "002"
    assert chan.channel_type == "xform"
    assert warped.exists()



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
            mt_path.parent.mkdir(exist_ok=True, parents=True)
            nrrd.write(
                str(mt_path),
                binary_blobs(length=20, n_dim=3).astype(np.uint8) * 254,
                header=header,
            )
            check_align_masks(session, root_dir)
            check_align_to_mask(session, root_dir)


def check_select_images(session: Session, root_dir: Path):
    template_creation.select_images(session, None, "001")
    image_dir = root_dir / "001"
    new_file_list = list(image_dir.glob("*for_template.nrrd"))
    assert len(new_file_list) == 1
    new_file_channel = (
        session.execute(select(sc.Channel).order_by(sc.Channel.id.desc()))
        .scalars()
        .first()
    )
    assert new_file_channel is not None
    assert new_file_channel.producer.function == "select_images"


def check_groupwise_template(session: Session, root_dir: Path):
    template_path = root_dir / "template/mask_template.nrrd"
    template_path.parent.mkdir(exist_ok=True)
    random_data = binary_blobs(
        length=100, blob_size_fraction=0.1, n_dim=3, volume_fraction=0.1
    )
    nrrd.write(
        file=str(template_path), data=random_data.astype(np.float32), header=header
    )
    template_creation.iterative_mask_template(session, make_template=False)
    data, _ = nrrd.read(str(template_path))
    assert data.max() == 254
    assert data.dtype == np.uint8
    path_from_db = (
        session.execute(
            select(sc.GlobalConfig).filter(sc.GlobalConfig.key == "mask_template_path")
        )
        .scalar_one()
        .value
    )
    assert Path(path_from_db).resolve() == template_path.resolve()


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
            api.segment_neuropil(session, ["001"], None, None, None)
            check_select_images(session, root_dir)
            check_groupwise_template(session, root_dir)
            api.mask_affine(session, None)
            api.align_to_mask(session, None)
