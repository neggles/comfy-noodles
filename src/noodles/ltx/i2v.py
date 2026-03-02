import torch
from comfy.sd import VAE
from comfy.utils import common_upscale
from comfy_api.latest import LatentInput, io

from .common import MaskStrategy, get_mask_decay_curve


class LTXImg2VidInplaceNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXImg2VidInplaceNood",
            display_name="LTX Img2Vid Inplace Nood",
            category="noodles/ltx",
            inputs=[
                io.Vae.Input("vae"),
                io.Image.Input("images"),
                io.Latent.Input("latent"),
                io.Int.Input(
                    "num_frames",
                    default=1,
                    min=1,
                    max=1025,
                    step=8,
                    tooltip="Number of frames to encode and replace in the latent. Must be a multiple of the VAE time scale factor (e.g. 8).",
                ),
                io.Float.Input(
                    "strength_min",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Target strength for the final frame in a multi-frame overlap window.",
                ),
                io.Combo.Input(
                    "mask_strat",
                    options=MaskStrategy,
                    default=MaskStrategy.SolidMask,
                    tooltip="The curve/window function used to calculate strength decay across the overlap window",
                ),
                io.Int.Input(
                    "decay_start",
                    default=0,
                    min=0,
                    max=1024,
                    tooltip="Number of frames to keep at full strength before decay starts",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        vae: VAE,
        images: torch.Tensor,
        latent: LatentInput,
        num_frames: int,
        strength_min: float,
        mask_strat: MaskStrategy,
        decay_start: int,
    ) -> io.NodeOutput:
        samples = latent["samples"]
        time_scale_factor, height_scale_factor, width_scale_factor = vae.downscale_index_formula

        batch, _, latent_frames, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        # print some debug info about the input shapes and parameters
        print(f"Input images shape: {images.shape}")

        # rescale input images if they are not the right resolution
        if images.shape[1] != height or images.shape[2] != width:
            # flip to [N, C, H, W] for rescaling, then flip back to [N, H, W, C]
            pixels = common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        else:
            pixels = images

        # strip alpha channel
        encode_pixels = pixels[:, :, :, :3]

        # Get existing noise mask if present, otherwise create new one
        if "noise_mask" in latent:
            conditioning_latent_frames_mask = latent["noise_mask"].clone()
        else:
            conditioning_latent_frames_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )

        # Get frame weights based on decay curve
        frame_weights = get_mask_decay_curve(mask_strat, num_frames, decay_start, w_min=strength_min)
        print(f"I2V frame weights: {frame_weights}")

        frame_1_lat = vae.encode(encode_pixels[0:1])
        if num_frames == 1:
            # If only one frame, just replace the first latent frame and return
            samples[:, :, 0 : frame_1_lat.shape[2]] = frame_1_lat
            conditioning_latent_frames_mask[:, :, 0 : frame_1_lat.shape[2]] = 1.0 - frame_weights[0]
            return io.NodeOutput({"samples": samples, "noise_mask": conditioning_latent_frames_mask})

        # make sure num_frames is a multiple of the time scale factor plus one

        # iterate over frames and apply weighted strengtht to the noise mask
        for idx in range(num_frames):
            latent_idx = idx // time_scale_factor

            # encode single frame to latent with VAE
            t = vae.encode(encode_pixels[idx : idx + 1])

            # clamp start_idx and end_idx to valid range
            latent_idx = max(0, min(latent_frames - 1, latent_idx))
            end_idx = min(latent_idx + t.shape[2], latent_frames)

            # Replace the corresponding frames in the samples with encoded image latents
            samples[:, :, latent_idx:end_idx] = t[:, :, : end_idx - latent_idx]

            # update the noise mask for the current frame range based on the frame weight
            # mask values are inverted relative to weights, so we subtract from 1.0 here
            conditioning_latent_frames_mask[:, :, latent_idx:end_idx] = 1.0 - frame_weights[idx]

        return io.NodeOutput({"samples": samples, "noise_mask": conditioning_latent_frames_mask})
