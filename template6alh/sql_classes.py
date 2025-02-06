"""
This code contains the classes that are used for sqlqlqhenmy
"""

from typing import Literal
from datetime import datetime

from sqlalchemy import (
    String,
    Integer,
    ForeignKey,
    Float,
    DateTime,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

ChannelType = Literal["mask", "raw", "aligned", "aligned-mask", "xform", "landmarks"]
FunctionName = Literal[
    "make-landmarks",
    "landmark-register",
    "mask-register",
    "landmark-align",
    "reformat-fasii",
]


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
    scalex: Mapped[float] = mapped_column(Float, nullable=False)
    scaley: Mapped[float] = mapped_column(Float, nullable=False)
    scalez: Mapped[float] = mapped_column(Float, nullable=False)
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
    function: Mapped[FunctionName] = mapped_column(String, nullable=False)
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
