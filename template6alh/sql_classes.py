"""
This code contains the classes that are used for sqlqlqhenmy
"""

from typing import Literal
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    String,
    Integer,
    create_engine,
    ForeignKey,
    select,
    Float,
    DateTime,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

ChannelType = Literal["mask", "raw", "aligned", "aligned-mask", "xform", "image"]


class Base(DeclarativeBase):
    pass


class GlobalConfig(Base):
    """
    Global configuration data
    """

    __tablename__ = "global_config"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String)


class ChannelMetadata(Base):
    """
    unstructured metadata for each channel
    """

    __tablename__ = "channel_metadata"
    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channel.id"), nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[str] = mapped_column(String, nullable=False)
    channel: Mapped["Channel"] = relationship("Channel", back_populates="mdata")


class RawFile(Base):
    """
    represents a czi file

    path: Mapped[str]
    neuropil_chan: Mapped[int | None]
    fasii_chan: Mapped[int | None]
    eve_chan: Mapped[int | None]
    relationships
        data_chans: Mapped[list["DataChannel"]]
        images: Mapped[list["Image"]]
    """

    __tablename__ = "raw_file"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    neuropil_chan: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fasii_chan: Mapped[int | None] = mapped_column(Integer, nullable=True)
    eve_chan: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # relationships
    data_chans: Mapped[list["DataChannel"]] = relationship(
        back_populates="raw_file", cascade="all, delete-orphan"
    )
    images: Mapped[list["Image"]] = relationship(
        back_populates="raw_file", cascade="all, delete-orphan"
    )


class DataChannel(Base):
    """
    represents a channel number for data

    chan: Mapped[int]
    relationships:
        raw_file: Mapped[RawFile]
    """

    __tablename__ = "data_channel"
    id: Mapped[int] = mapped_column(primary_key=True)
    chan: Mapped[int] = mapped_column(Integer, nullable=False)
    # relationships
    raw_file_id: Mapped[int] = mapped_column(ForeignKey("raw_file.id"))
    raw_file: Mapped[RawFile] = relationship(back_populates="data_chans")


class Channel(Base):
    """
    represents a data file stored as a nrrd

    path: Mapped[str]
    channel_type: Mapped[ChannelType]
    step_number: Mapped[int]
    number: Mapped[int]
    scalex: Mapped[int]
    scaley: Mapped[int]
    scalez: Mapped[int]
    relationships:
        image: Mapped["Image"]
        producer: Mapped["AnalysisStep"]
        mdata: Mapped[list["ChannelMetadata"]]
    """

    __tablename__ = "channel"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    channel_type: Mapped[ChannelType] = mapped_column(String, nullable=False)
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    scalex: Mapped[int] = mapped_column(Float, nullable=False)
    scaley: Mapped[int] = mapped_column(Float, nullable=False)
    scalez: Mapped[int] = mapped_column(Float, nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=True)
    # relationships
    image_id: Mapped[int] = mapped_column(ForeignKey("image.id"))
    image: Mapped["Image"] = relationship(back_populates="channels")
    producer_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_step.id"), nullable=True
    )

    # Relationships
    producer: Mapped["AnalysisStep"] = relationship(
        "AnalysisStep", back_populates="output_channels"
    )
    used_in_steps: Mapped[list["AnalysisStep"]] = relationship(
        "AnalysisStep", secondary="analysis_step_input", back_populates="input_channels"
    )
    mdata: Mapped[list["ChannelMetadata"]] = relationship(
        "ChannelMetadata", back_populates="channel"
    )


class Image(Base):
    """
    represents a CNS image

    folder: Mapped[str]
    progress: Mapped[int]
    relationships:
        channels: Mapped[list[Channel]]
        analysis_steps: Mapped[list["AnalysisStep"]]
        raw_file: Mapped[RawFile]
    """

    __tablename__ = "image"
    id: Mapped[int] = mapped_column(primary_key=True)
    folder: Mapped[str] = mapped_column(String, nullable=False)
    progress: Mapped[int] = mapped_column(Integer, nullable=False)
    # relationships
    channels: Mapped[list[Channel]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )
    raw_file_id: Mapped[int] = mapped_column(ForeignKey("raw_file.id"))
    raw_file: Mapped[RawFile] = relationship(back_populates="images")


class AnalysisStep(Base):
    """
    a python function called for analysis

    function: Mapped[str]
    kwargs: Mapped[str]
    runtime: Mapped[datetime]
    relationships:
        image: Mapped["Image"]
    """

    __tablename__ = "analysis_step"
    id: Mapped[int] = mapped_column(primary_key=True)
    function: Mapped[str] = mapped_column(String, nullable=False)
    kwargs: Mapped[str] = mapped_column(String, nullable=False)
    runtime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # relationships
    output_channels: Mapped[list["Channel"]] = relationship(
        "Channel", back_populates="producer"
    )
    input_channels: Mapped[list["Channel"]] = relationship(
        "Channel", secondary="analysis_step_input", back_populates="used_in_steps"
    )


class AnalysisStepInput(Base):
    """
    Utility table to map many images to many analysis steps
    """

    __tablename__ = "analysis_step_input"
    analysis_step_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_step.id"), primary_key=True
    )
    channel_id: Mapped[int] = mapped_column(ForeignKey("channel.id"), primary_key=True)


def test():
    Path("db.db").unlink()
    engine = create_engine("sqlite:///db.db", echo=True)
    session = Session(engine)
    Base.metadata.create_all(engine)
    channels = select(DataChannel)
    session.scalars(channels)
    dirs = list(Path("/mnt/g/peter/eg_imgs").iterdir())
    raw_data = RawFile(
        path="raw.czi",
        fasii_chan=1,
        neuropil_chan=2,
        images=[
            Image(
                progress=0,
                folder=str(d.relative_to("/mnt/g/peter/eg_imgs")),
                channels=[
                    Channel(
                        path=p.name,
                        number=p.name[4],
                        step_number=0,
                        channel_type="raw",
                        scalex=0.44,
                        scaley=0.28231951,
                        scalez=0.28231951,
                    )
                    for p in d.iterdir()
                ],
            )
            for d in dirs
        ],
    )
    global_config = GlobalConfig(key="prefix_dir", value="/mnt/g/peter/eg_imgs")
    with Session(engine) as session:
        session.add_all([raw_data, global_config])
        session.commit()
