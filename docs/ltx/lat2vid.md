# LTX-2 Segment Continuation & Bootstrap Reference

Documenting the design of the overlap and masking process being used here so that I don't forget the fiddly
little details, of which there are quite a few. I have probably overcomplicated this a little but whatever.

Ultimately I'm trying to make a song-to-music-video pipeline. We'll see how that goes...

## Design Goals

- Try to retain flexibility to change between masking modes during generation of segments. Total frames added to the output video should not vary dependding on anything other than the number of latents in a batch/segment, and the overlap factor.
- Save all segment files for a given video in a single output directory, such that a standard lexical sort will sort by:
    1. Video ID
    2. Segment number
    3. Segment iteration (i.e. generation attempt)
- Do overlapped extended-length generation with as few VAE roundtrips as possible - ideally zero, but some crimes will be needed for that to work.
- Handle audio conditioning nice and easily - if I load up a segment I should be able to get the correct global frame indices to extract reference audio from, i.e. the global offset into the original audio (loaded as a video to make things easier) and the "frame count" of audio samples to extract.

I'll also generally be dropping the last latent out of each generated segment before saving, since the quality of that is usually pretty poor due to the model not having a future-context scratchpad to work in.

OK, now onto the fiddly bits.

# 1. LTX2 Latent Format

LTX2 uses temporally packed latents where latent[0] is one frame, and latent[1:] are 8 frames each. A batch with `L` latents decodes into `1 + 8 * (L - 1)` frames, e.g. 21 latents = 161 frames.

I will refer to the initial 1-frame latent as the "bootstrap latent" from here on in. The bootstrap latent makes it somewhat difficult to avoid a VAE roundtrip when doing segmented/overlapped generation, since the final latent from a segment will be an N-frame latent rather than a bootstrap latent, and they're not directly interchangeable.

So how do we solve this problem?

Simple. We do crimes. But I'll get to that.

# 2. Overlap Handling

For the sake of this documentation we'll just treat the latent as a list of opaque objects. The inner dims don't matter.

Let:
- `L` = number of latents saved after generation
- `L_gen` = number of latents in the batch, including the terminal latent even if dropped.
- `k` = number of latents to overlap, including bootstrap
- `d` = `1` if we're dropping the last latent before save
- `prev.lat[]` = the saved latent from the last segment
- `next.lat[]` = the (initially empty) latent for this segment.
- `start_frame` = Global video frame number for the start of this segment.
- `n_frames` = Number of good frames in this segment. Includes bootstrap frame for segment 0 only.

Minimum value for `k` is 2, ensuring:
- 1 overlap slot for the bootstrap latent/frame
- 1 or more overlap slots for actual frame latents.

This means we actually use `L - (k - 1) * 8` frames from each segment. Minimum value for `L` is 3, since if you have to overlap by at least 2, you probably also want to actually generate some more frames.

We can calculate the `end_frame` for a segment via `end_frame = start_frame + n_frames - overlap_frames` but i'll get to that below.

### Example

Given `L` = 21 (161 frames), `k` = 6, and `d` = 1, here's what the first 3 segments look like:

Segment 0:
```python
start_frame = 0
L_gen = L + d = 22  # add an extra since we're throwing it away later
n_frames = 1 + (8 * (L - 1)) = 161 # we keep the bootstrap frame here
end_frame = 161 # not actually saved
```
Don't need to do anything particularly fancy here.

Segment 1:
```python
prev.start_frame = 0 # start_frame from previous segment metadata
prev.n_frames = 161
overlap_frames = 8 * (k - 1) = 40  # 1 less than k 

start_frame = prev.start_frame + prev.n_frames - overlap_frames = 121
L_gen = L + d = 22  # as above
n_frames = 8 * (L - 1) = 160  # drop bootstrap frame
```

Segment 2:
```python
prev.start_frame = 121 # start_frame from previous segment metadata
prev.n_frames = 160
overlap_frames = 8 * (k - 1) = 40  # 1 less than k 

start_frame = prev.start_frame + prev.n_frames - overlap_frames = 241
L_gen = L + d = 22  # as above
n_frames = 8 * (L - 1) = 160  # drop bootstrap frame
```

And so on and so forth.

How the overlap zone is masked is handled further down the page in section 4.

## 2.1 Video Assembly

For segments after segment 0, the bootstrap latent's decoded frame **MUST** be dropped when assembling the video.

For segment 0, the bootstrap latent's decoded frame **MUST NOT** be dropped during video assembly.

Assembly is fairly simple.

1. Load first segment
    1. Decode saved latent to frames
    2. Store frames in buffer
2. Load next segment
    1. Decode saved latent to frames
    2. Get segment's start_frame
    3. Store frames into buffer, using `start_frame` as an offset. This will overwrite the tail end of the previous segment.
3. `GOTO STEP_2` until you run out of segments (or CPU OOM)
4. Encode video, mixing in audio, and save.

The other option would be to assemble a ginormous latent in the same fashion, then pray you have enough VRAM to decode multiple minutes of video & that the VAE will actually handle videos that long (I think it *should* but I don't know. Will have to try at some point.)

# 3. Bootstrap Modes

Bootstrap mode determines how `latent[0]` is constructed for a continuation segment.

This is the part where we start bullying poor, innocent, defenseless diffusion models into doing things they weren't trained for.

I've come up with three possible methods to try. They all share one common invariant: the first non-bootstrap latent is always a fully-noise-masked copy of the previous segment's `latent[-(k + 1)]`.

## 3.1 Bootstrap via Dummy Latent

This mode feels mean, but I suspect it'll work the best. 

1. We simply provide an empty, unmasked bootstrap latent,
2. and let the model diffuse the first frame as well.

This feels unhinged, and I suspect the decoded image from those frames will be incomprehensible in a visual sense. But there's at least one frozen known-good multi-frame latent following it, and the model should diffuse something that makes sense to it semantically even if it's visually meaningless, so it should still work okay, at least for sufficiently high values of `k` (at least 4 at 24fps?).

*Should.*

## 3.2 Bootstrap via VAE Roundtrip

This is the least-cursed but also kinda mid option, and is the main reason why `k` has to be at least 2.

1. Take the last decoded frame of the previous segment's `latent[-k]`
2. Encode it to a single-frame latent with the VAE
3. Use that.

Boring. Lame. Meh. Not even cursed. What's the point of doing something you know will work?

And I specifically set out to *not* do this due to generation loss from the repeated VAE roundtrips.

But it's an option I supposed.

## 3.3 Bootstrap via Raw Latent

Most cursed, least likely to work, still going to try it since the VAE roundtrip option means we have a spare latent anyway.

1. Take `latent[-k]` from the previous segment
2. Just use it as the bootstrap latent and damn the consequences.

If this works I'm going to be really mad tbh.

# 4. Mask Scheduling Strategies

When overlapping latents we can optionally apply a decaying curve to the noise mask strength, allowing the model to alter/partially denoise some of the existing latents. This should hopefully make the seams between segments less obvious. 

No idea which curve will work best though so we have a bunch of 'em:

- SolidMask: this just hard-masks the whole chunk. Can optionally mask the first kept overlap latent at 1.0 and the rest at a fixed value, though that's almost certainly a bad idea.
- LinearDecay: Does what it says on the box.
- CosineDecayV1: Does a cosine decay curve. 
- both Smoothstep and Smootherstep curves
- Half-gaussian decay with adjustable sigma.

Have a look at `get_mask_decay_curve()` for the details.

---

OK I think that's about everything I needed to have written down for now.
Will flesh this out later

Narrator: *"no, she won't"*
