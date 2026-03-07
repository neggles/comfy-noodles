from enum import StrEnum
from math import ceil
from typing import Any

import numpy as np
import torch
from comfy_api.latest import AudioInput, io, ui
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchaudio import transforms as AT

from noodles.utils import get_input_dir_path

VIDEO_EXTNS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


class StringIntAddNood(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="noodles-StringIntAddNood",
            display_name="String Int Add",
            category="noodles/convert",
            inputs=[
                io.String.Input(
                    "in_a",
                    default="0",
                    multiline=False,
                    display_name="A",
                ),
                io.Int.Input(
                    "in_b",
                    default=0,
                    display_name="B",
                ),
            ],
            outputs=[
                io.Int.Output("result", display_name="Out"),
            ],
        )

    @classmethod
    def execute(cls, *, in_a: str, in_b: int):  # ty:ignore[invalid-method-override]
        try:
            int_a = int(in_a)
        except ValueError as e:
            raise ValueError(f"Could not convert '{in_a}' to an integer.") from e

        return io.NodeOutput(result=int_a + in_b)  # ty:ignore[unknown-argument]


# WIP, not loaded yet
class LoadVideoForAudioNood(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        input_dir = get_input_dir_path()
        files = [f.relative_to(input_dir).as_posix() for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTNS]

        return io.Schema(
            node_id="noodles-LoadVideoForAudioNood",
            display_name="Load Video for Audio",
            category="noodles/audio",
            is_experimental=True,
            inputs=[
                io.Combo.Input(
                    "video",
                    display_name="Video Path",
                    tooltip="Path to the video file to load audio from.",
                    options=[str(f) for f in files],
                ),
                io.Float.Input(
                    "framerate",
                    display_name="Framerate",
                    default=30.0,
                    min=1,
                    max=240,
                    tooltip="Framerate of your generation, used to snip segments for partial loads.",
                ),
                io.Int.Input(
                    "start_frame",
                    display_name="Start Frame",
                    default=0,
                    min=0,
                    tooltip="Starting frame for this audio segment, at the given framerate.",
                ),
                io.Int.Input(
                    "max_frames",
                    display_name="Max Frames",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    tooltip="Number of frames worth of audio to output for this segment, at the given framerate.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="Audio"),
                io.Int.Output(display_name="Frame Count"),
                io.Image.Output(display_name="First Frame"),
            ],
        )

    @classmethod
    def execute(
        cls,
        *,
        video_path: str,
        framerate: float,
        start_frame: int,
        max_frames: int,
    ):
        pass


def plot_waveform(waveform: torch.Tensor, sr: int, title: str = "Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram: torch.Tensor, title: str = None, ylabel: str = "freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    power_to_db = AT.AmplitudeToDB("power", 80.0)
    ax.imshow(power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


class AudioPreviewMelSpectrogramNood(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioPreviewMelSpectrogramNood",
            display_name="Audio Preview Mel Spectrogram",
            category="noodles/audio",
            is_experimental=True,
            inputs=[
                io.Audio.Input("audio", display_name="Audio"),
                io.Int.Input("fft_size", display_name="FFT Size", default=2048, min=256, max=16384),
                io.Int.Input("n_mels", display_name="Mel Bands", default=128, min=16, max=512),
                io.Float.Input("power", display_name="Magnitude Exp.", default=2.0, min=0, max=4, step=0.1),
                io.Boolean.Input(
                    "normalized",
                    display_name="Normalize",
                    default=False,
                    label_on="Yes",
                    label_off="No",
                    tooltip="Whether to normalize by magnitude after STFT.",
                ),
                io.Int.Input(
                    "width_px",
                    display_name="Width (px)",
                    default=1280,
                    min=64,
                    max=4096,
                    step=8,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    "height_px",
                    display_name="Height (px)",
                    default=960,
                    min=64,
                    max=4096,
                    step=8,
                    display_mode=io.NumberDisplay.slider,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="Spectrogram"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        *,
        audio: AudioInput,
        fft_size: int,
        n_mels: int,
        power: float,
        normalized: bool,
        width_px: int,
        height_px: int,
    ):
        waveform = audio["waveform"]  # [B, C, T]
        sr = audio["sample_rate"]

        # create the mel spectrogram transform
        mel_spectrogram: AT.MelSpectrogram = AT.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=fft_size,
            win_length=None,
            hop_length=fft_size // 2,
            power=power,
            normalized=normalized,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        )

        # strip the batch dim if present
        if waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)

        melspec = mel_spectrogram(waveform)

        # make sure we're in the right backend mode
        prev_backend = plt.get_backend()
        plt.switch_backend("Agg")

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(width_px / 96, height_px / 96),
            dpi=96,
        )

        plot_waveform(waveform, sr, title="Audio waveform", ax=axs[0])
        plot_spectrogram(melspec[0], title="Mel Spectrogram", ax=axs[1], ylabel="mel freq")
        plt.tight_layout()

        # get canvas
        canvas: FigureCanvasAgg = fig.canvas
        # make the renderer render
        canvas.draw()

        # get the canvas pixels as a numpy array, convert to float32 in [0, 1]
        rgba = np.asarray(canvas.buffer_rgba(), copy=True).astype(np.float32) / 255.0

        # convert to torch and drop the alpha channel
        image = torch.from_numpy(rgba[:, :, :3]).clone().unsqueeze(0)  # [1, H, W, C]

        # close figure
        plt.close(fig)

        # restore the previous backend if it was different
        if prev_backend != plt.get_backend():
            plt.switch_backend(prev_backend)

        return io.NodeOutput(
            image,
            ui=ui.PreviewImage(image, cls=cls),
        )


class AspectRatioOption(StrEnum):
    Square = "1:1"
    OldPC = "4:3"
    SemiWide = "3:2"
    Landscape = "8:5"
    Widescreen = "16:9"
    UltraWide = "21:9"
    ThreeFour = "3:4"
    SemiTall = "2:3"
    Portrait = "5:8"
    Tall = "9:16"
    UltraTall = "9:21"

    def get_width_height(self, side_length: int) -> tuple[int, int]:
        match self:
            case AspectRatioOption.Square:
                return side_length, side_length
            case AspectRatioOption.OldPC:
                return side_length, side_length * 3 // 4
            case AspectRatioOption.SemiWide:
                return side_length, side_length * 2 // 3
            case AspectRatioOption.Landscape:
                return side_length, side_length * 5 // 8
            case AspectRatioOption.Widescreen:
                return side_length, side_length * 9 // 16
            case AspectRatioOption.UltraWide:
                return side_length, side_length * 9 // 21
            case AspectRatioOption.ThreeFour:
                return side_length * 3 // 4, side_length
            case AspectRatioOption.SemiTall:
                return side_length * 2 // 3, side_length
            case AspectRatioOption.Portrait:
                return side_length * 5 // 8, side_length
            case AspectRatioOption.Tall:
                return side_length * 9 // 16, side_length
            case AspectRatioOption.UltraTall:
                return side_length * 9 // 21, side_length


class VideoGenParamsNood(io.ComfyNode):
    """
    Convenience node to output basic parameters for LTX video generation.
    Width, height, number of frames, frames per second as both float and int
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoGenParamsNood",
            display_name="Video Generation Params",
            category="noodles/ltx",
            inputs=[
                io.DynamicCombo.Input(
                    id="res_mode",
                    options=[
                        io.DynamicCombo.Option(
                            "Aspect Ratio",
                            [
                                io.Combo.Input(
                                    "aspect_ratio",
                                    display_name="Aspect Ratio",
                                    options=AspectRatioOption,
                                    default=AspectRatioOption.Widescreen,
                                ),
                                io.Int.Input(
                                    "side_length_px",
                                    display_name="Side Length (px)",
                                    default=960,
                                    min=32,
                                    max=4096,
                                    step=32,
                                    display_mode=io.NumberDisplay.slider,
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "Custom",
                            [
                                io.Int.Input("width_px", display_name="Width", default=960, min=32, max=4096, step=32),
                                io.Int.Input("height_px", display_name="Height", default=544, min=32, max=4096, step=32),
                            ],
                        ),
                    ],
                    display_name="Resolution Mode",
                ),
                io.Int.Input(
                    "n_frames",
                    display_name="Frames",
                    default=121,
                    min=1,
                    max=2049,
                    step=8,
                ),
                io.Float.Input(
                    "framerate",
                    display_name="FPS",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=1.0,
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width_px"),
                io.Int.Output(display_name="height_px"),
                io.Int.Output(display_name="n_frames"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="fps_int"),
            ],
        )

    @classmethod
    def execute(
        cls,
        res_mode: dict[str, Any],
        n_frames: int,
        framerate: float,
    ) -> io.NodeOutput:
        match res_mode["res_mode"]:
            case "Aspect Ratio":
                aspect_ratio = AspectRatioOption(res_mode.get("aspect_ratio"))
                side_length = int(res_mode.get("side_length_px", -1))
                width_px, height_px = aspect_ratio.get_width_height(side_length)
            case "Custom":
                width_px = int(res_mode.get("width_px", -1))
                height_px = int(res_mode.get("height_px", -1))
            case _:
                raise ValueError(f"Invalid resolution mode: {res_mode['res_mode']}")

        if width_px < 32 or height_px < 32:
            raise ValueError(f"Width and height must be at least 32 pixels. Got {width_px}x{height_px}.")

        # round up shorter edge to nearest multiple of 32 if it's not already, to ensure compatibility with common video codecs and the LTX pipeline
        if width_px < height_px and width_px % 32 != 0:
            width_px = ((width_px + 31) // 32) * 32
        elif height_px < width_px and height_px % 32 != 0:
            height_px = ((height_px + 31) // 32) * 32

        # round fps number to int (upwards)
        framerate = ceil(framerate)
        return io.NodeOutput(
            width_px,
            height_px,
            n_frames,
            float(framerate),
            int(framerate),
        )
